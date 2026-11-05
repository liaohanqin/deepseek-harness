/** One standard ACP session's Agent, configuration, prompt, update, and teardown lifecycle. */

import type { Context } from '@deepseek-ai/cordis'
import {
  RequestError,
  type ContentBlock,
  type McpServer,
  type PromptRequest,
  type PromptResponse,
  type SessionConfigOption,
  type SessionNotification,
  type StopReason,
} from '@agentclientprotocol/sdk'
import type { Agent, AgentHandle, AgentOptions, AgentStatus } from '@deepseek-ai/dsh-agent'
import { installModelSelection } from '@deepseek-ai/dsh-agent'
import { createUserMessage, errorChain, type UserMessage } from '@deepseek-ai/dsh-llm'
import { type Session, type SessionEvent, type SessionId, type TurnEndReason } from '@deepseek-ai/dsh-session'
// Side-effect type import: declaration-merges the `workingDirectory` service read below.
import type {} from '@deepseek-ai/dsh-working-directory'
import { AcpContentError, admitAcpPrompt } from './content.ts'
import { turnEndToStopReason } from './codec.ts'
import { mountAcpMcpServers } from './mcp.ts'
import { AcpModelControl } from './model-control.ts'
import { assistantUpdates, toolCallUpdate, toolResultUpdate } from './updates.ts'

/** One registered human command, narrowed to the fields ACP advertises. */
interface AvailableCommandMetadata {
  /** Command name without the leading slash. */
  readonly name: string
  /** Human-readable summary used in discovery UI. */
  readonly description: string
  /** Optional free-form input hint advertised to capable clients. */
  readonly input?: { readonly hint: string }
}

/** Human-command registry without depending on the commands package. */
interface CommandRegistry {
  /** Effective descriptors for one agent, after scoped shadowing. */
  list(agent: Agent): readonly AvailableCommandMetadata[]
}

/** One skill reachable by the user-facing `/name` gesture. */
interface SkillSummary {
  /** Kebab-case skill name, addressed without the leading slash. */
  readonly name: string
  /** Short routing description shown by discovery consumers. */
  readonly description: string
  /** Resolved model and user invocation controls. */
  readonly invocation: { readonly userInvocable: boolean }
}

/** Skill registry without depending on the skill package. */
interface SkillRegistry {
  /** Catalog for one viewing scope and workspace. */
  list(options: {
    readonly cwd: string
    readonly scope: Agent
    readonly signal: AbortSignal
  }): Promise<readonly SkillSummary[]>
}

/** Subagent completion and teardown without depending on the subagent package. */
interface SubagentLifecycle {
  /** Dispose subagent descendants below exact host-owned parents child-first. */
  drainDescendants(parents: readonly Agent[]): Promise<void>
  /** Join progressing work in the root's subagent tree without closing future admission. */
  waitForTreeWork(root: Agent): Promise<boolean>
}

/** Inputs shared by fresh and resumed ACP session construction. */
interface AcpSessionBuildOptions {
  cwd: string
  mcpServers: readonly McpServer[]
  agentOptions: AgentOptions
  signal: AbortSignal
  notify: (notification: SessionNotification) => Promise<void>
}

/** Fresh ACP session construction inputs. */
export interface CreateAcpSessionOptions extends AcpSessionBuildOptions {
  sessionId: SessionId
}

/** Persisted ACP session construction inputs. */
export interface ResumeAcpSessionOptions extends AcpSessionBuildOptions {
  sessionId: SessionId
}

/**
 * Bound on workspace resolution and skill discovery for one catalog publish.
 * Discovery reads the filesystem, so a wedged provider root must not stall the
 * notification queue behind it.
 */
const COMMAND_DISCOVERY_TIMEOUT_MS = 10_000

interface InflightPrompt {
  resolve: (reason: StopReason) => void
  reject: (error: Error) => void
  messageId: string | undefined
  messageQueued: boolean
  turn: number | undefined
  endReason: TurnEndReason | undefined
  admissionDone: Promise<void>
  finishAdmission: () => void
  admissionController: AbortController
  cancellationDone: PromiseWithResolvers<void>
  cancelStartedIdle: boolean
  settlementStarted: boolean
  outputError: Error | undefined
  agentError: Error | undefined
}

/**
 * Steering outcomes shared by the Claude, Codex, and pi ACP adapters.
 *
 * `injected` means the prompt joined the turn already running. `promptRequired`
 * means no turn consumed it, so the caller should submit an ordinary
 * `session/prompt`; this adapter never reports `startedNewTurn` because it
 * leaves ownership of a new turn's lifecycle with the client.
 */
export type SteerOutcome = 'injected' | 'startedNewTurn' | 'promptRequired' | 'failed'

/** Request `_session/steering`: an ACP extension, not a core method. */
export interface SteerRequest {
  /** Wire session identifier; branded at the handler boundary. */
  sessionId: string
  prompt: readonly ContentBlock[]
  /** Steering options; `idleBehavior` selects what an idle session reports. */
  _meta?: { steering?: { idleBehavior?: 'promptRequired' } }
}

/** Response `_session/steering`. */
export interface SteerResponse {
  outcome: SteerOutcome
  /** Machine-readable detail, present when `outcome` is not `injected`. */
  reason?: string
}

/** Standard invalid-parameter failure with protocol-safe detail. */
function invalidParams(detail: string): RequestError {
  return RequestError.invalidParams(undefined, detail)
}

/** Standard internal failure with protocol-safe detail. */
function internalError(detail: string): RequestError {
  return RequestError.internalError(undefined, detail)
}

/**
 * Install switch notices and mount the requested MCP clients in the unpublished Agent scope.
 * @param agentCtx - unpublished Agent scope.
 * @param options - workspace and MCP servers requested for the session.
 */
async function composeAgent(agentCtx: Context, options: AcpSessionBuildOptions): Promise<void> {
  installModelSelection(agentCtx)
  await mountAcpMcpServers(agentCtx, options.mcpServers, options.cwd)
}

/**
 * Per-session ACP module. It owns the unpublished Agent composition, model
 * configuration view, one-prompt admission slot, ordered standard updates,
 * and memoized quiescent teardown.
 */
export class AcpSession {
  /** The exact top-level Agent owned by this ACP session. */
  readonly agent: Agent
  private readonly modelControl: AcpModelControl
  private outputTail = Promise.resolve()
  private inflight: InflightPrompt | undefined
  private closing: Promise<void> | undefined

  private constructor(
    private readonly ctx: Context,
    handle: AgentHandle,
    private readonly notify: (notification: SessionNotification) => Promise<void>,
  ) {
    this.agent = handle.agent
    this.modelControl = new AcpModelControl(ctx.llm, handle.agent.model)
    this.disposeAgent = () => handle.dispose()
  }

  private readonly disposeAgent: () => Promise<void>

  /**
   * Compose a fresh Agent and all requested MCP clients before publication.
   * @param ctx - ACP plugin context with Agent, LLM, and persistence services.
   * @param options - fresh session identity, workspace, route, MCP, and notifier.
   * @returns the fully composed per-session module.
   */
  static async create(ctx: Context, options: CreateAcpSessionOptions): Promise<AcpSession> {
    const handle = await ctx.agents.create({
      sessionId: options.sessionId,
      meta: { cwd: options.cwd },
      agentOptions: options.agentOptions,
      signal: options.signal,
      setup: agentCtx => composeAgent(agentCtx, options),
    })
    return new AcpSession(ctx, handle, options.notify)
  }

  /**
   * Restore a persisted Agent and compose the request's fresh MCP connections.
   * The Session's recorded model selection survives; deployment options supply
   * the route only when the Session records neither a selection nor a request.
   * @param ctx - ACP plugin context with Agent, LLM, and persistence services.
   * @param options - persisted identity, workspace, initial route, MCP, and notifier.
   * @returns the restored per-session module.
   */
  static async resume(ctx: Context, options: ResumeAcpSessionOptions): Promise<AcpSession> {
    const handle = await ctx.agents.resume({
      resumeSessionId: options.sessionId,
      agentOptions: options.agentOptions,
      signal: options.signal,
      setup: agentCtx => composeAgent(agentCtx, options),
    })
    return new AcpSession(ctx, handle, options.notify)
  }

  /**
   * Whether this module owns an exact Agent reference.
   * @param agent - Agent observed on a scoped runtime event.
   * @returns true only for this session's owned Agent.
   */
  owns(agent: Agent): boolean {
    return this.agent === agent
  }

  /**
   * Whether this module owns an exact Session reference.
   * @param session - Session observed on a durable event.
   * @returns true only for this session's owned Session.
   */
  ownsSession(session: Session): boolean {
    return this.agent.session === session
  }

  /**
   * Return the complete standard model configuration state.
   * @param signal - optional request cancellation.
   * @returns provider-grouped model and exact-model reasoning options.
   */
  configOptions(signal?: AbortSignal): Promise<SessionConfigOption[]> {
    this.assertActive()
    return this.modelControl.options(signal)
  }

  /**
   * Apply one standard configuration option from the next model step.
   * @param configId - advertised standard option id.
   * @param value - selected standard option value.
   * @param signal - optional request cancellation.
   * @returns the complete resulting option state.
   */
  setConfig(configId: string, value: unknown, signal?: AbortSignal): Promise<SessionConfigOption[]> {
    this.assertActive()
    return this.modelControl.set(configId, value, signal)
  }

  /** Resolve topology state off-chain, then serialize its notification without blocking execution updates. */
  topologyChanged(): void {
    if (this.closing !== undefined) return
    void this.modelControl.options()
      .then((configOptions) => {
        if (this.closing !== undefined) return
        const previous = this.outputTail
        this.outputTail = previous
          .then(() => this.notify({
            sessionId: this.agent.session.id,
            update: { sessionUpdate: 'config_option_update', configOptions },
          }))
          /* v8 ignore start -- the bridge notifier contains transport failure. */
          .catch((error: unknown) => {
            this.ctx.logger.warn(`acp: config-option update failed: ${errorChain(error)}`)
          })
        /* v8 ignore stop */
      })
      /* v8 ignore start -- option discovery contains per-provider failure. */
      .catch((error: unknown) => {
        this.ctx.logger.warn(`acp: config-option update failed: ${errorChain(error)}`)
      })
    /* v8 ignore stop */
  }

  /**
   * Publish the effective `/command` catalog, then serialize its notification
   * without blocking execution updates.
   *
   * User-invocable skills join the same list because `available_commands_update`
   * is the only discovery channel ACP offers. They stay plain-text gestures: the
   * host's pre-step boundary recognizes a leading `/name` and injects the skill
   * body, so a pick needs no command execution. `_meta.skill` keeps the two
   * apart for clients that care, and a command always wins a shared name —
   * matching the web client, which adjudicates the line before it becomes a
   * prompt.
   *
   * Both registries are optional: a composition without them advertises
   * nothing rather than failing the session.
   */
  commandsChanged(): void {
    if (this.closing !== undefined) return
    const commands = this.ctx.get('commands') as CommandRegistry | undefined
    const skills = this.ctx.get('skills') as SkillRegistry | undefined
    if (commands === undefined && skills === undefined) return
    const previous = this.outputTail
    this.outputTail = previous
      .then(async () => {
        const availableCommands: {
          name: string
          description: string
          input?: { hint: string }
          _meta?: { skill: true }
        }[] = []
        const claimed = new Set<string>()
        for (const command of commands?.list(this.agent) ?? []) {
          claimed.add(command.name)
          availableCommands.push({
            name: command.name,
            description: command.description,
            ...command.input === undefined
              ? {}
              : { input: { hint: command.input.hint } },
          })
        }
        // The agent's workspace selects project skill roots, so discovery wants
        // the live directory rather than the deployment default. A composition
        // without `workingDirectory` falls back to the session's recorded cwd.
        const workingDirectory = this.ctx.get('workingDirectory')
        const cwd = workingDirectory === undefined
          ? this.agent.session.header.cwd ?? process.cwd()
          : await workingDirectory.ensure(this.agent, AbortSignal.timeout(COMMAND_DISCOVERY_TIMEOUT_MS))
        const discovered = await skills?.list({
          cwd,
          scope: this.agent,
          signal: AbortSignal.timeout(COMMAND_DISCOVERY_TIMEOUT_MS),
        }) ?? []
        for (const skill of discovered) {
          if (!skill.invocation.userInvocable || claimed.has(skill.name)) continue
          claimed.add(skill.name)
          availableCommands.push({
            name: skill.name,
            description: skill.description,
            _meta: { skill: true },
          })
        }
        await this.notify({
          sessionId: this.agent.session.id,
          update: { sessionUpdate: 'available_commands_update', availableCommands },
        })
      })
      /* v8 ignore start -- the bridge notifier contains transport failure. */
      .catch((error: unknown) => {
        this.ctx.logger.warn(`acp: available-commands update failed: ${errorChain(error)}`)
      })
    /* v8 ignore stop */
  }

  /**
   * Admit one prompt and settle after the root, descendants, and updates finish.
   * @param params - standard ACP prompt request for this session.
   * @param imageEnabled - connection capability advertised at initialization.
   * @param requestSignal - JSON-RPC request cancellation signal.
   * @returns the correlated standard stop reason after ordered updates drain.
   */
  async prompt(
    params: PromptRequest,
    imageEnabled: boolean,
    requestSignal?: AbortSignal,
  ): Promise<PromptResponse> {
    this.assertActive()
    if (this.inflight !== undefined) throw invalidParams('a prompt is already in flight for this session')
    const completion = Promise.withResolvers<StopReason>()
    const admission = Promise.withResolvers<void>()
    const admissionController = new AbortController()
    const inflight: InflightPrompt = {
      resolve: completion.resolve,
      reject: completion.reject,
      messageId: undefined,
      messageQueued: false,
      turn: undefined,
      endReason: undefined,
      admissionDone: admission.promise,
      finishAdmission: admission.resolve,
      admissionController,
      cancellationDone: Promise.withResolvers<void>(),
      cancelStartedIdle: false,
      settlementStarted: false,
      outputError: undefined,
      agentError: undefined,
    }
    this.inflight = inflight
    const onRequestAbort = (): void => { this.cancelPrompt('ACP prompt request cancelled') }
    requestSignal?.addEventListener('abort', onRequestAbort, { once: true })
    /* v8 ignore next -- the SDK dispatches a live signal, then notifies abort through its listener. */
    if (requestSignal?.aborted === true) onRequestAbort()
    try {
      let admissionFailure: unknown
      const promptSelection = this.agent.model.current()
      try {
        if (this.ctx.agents.get(this.agent.id) !== this.agent) {
          throw internalError('prompt was not queued: the agent was disposed outside the bridge')
        }
        const content = await admitAcpPrompt(
          this.ctx,
          promptSelection,
          params.prompt,
          imageEnabled,
          admissionController.signal,
        )
        admissionController.signal.throwIfAborted()
        if (this.ctx.agents.get(this.agent.id) !== this.agent) {
          throw internalError('prompt was not queued: the agent was disposed outside the bridge')
        }
        const message = createUserMessage({
          content,
          source: { kind: 'user' },
        })
        inflight.messageId = message.id
        inflight.messageQueued = true
        try {
          this.agent.followup(message)
        } catch (error: unknown) {
          inflight.messageQueued = false
          throw error
        }
      } catch (error: unknown) {
        admissionFailure = error
      } finally {
        inflight.finishAdmission()
      }

      if (inflight.admissionController.signal.aborted) {
        this.settleAfterQuiescence(inflight)
        return { stopReason: await completion.promise }
      }
      if (admissionFailure !== undefined) {
        this.inflight = undefined
        if (admissionFailure instanceof AcpContentError) {
          throw admissionFailure.kind === 'invalid'
            ? invalidParams(admissionFailure.message)
            : internalError(admissionFailure.message)
        }
        if (admissionFailure instanceof RequestError) throw admissionFailure
        throw internalError(`prompt was not queued: ${(admissionFailure as Error).message}`)
      }

      this.settleAfterQuiescence(inflight)
      return { stopReason: await completion.promise }
    } finally {
      requestSignal?.removeEventListener('abort', onRequestAbort)
    }
  }

  /**
   * Join one prompt to the turn this session is already running.
   *
   * Steering is an operation on an in-flight turn, never an alternative way to
   * start one: with no live turn the content stays unconsumed and the caller
   * submits an ordinary `session/prompt`, which keeps ownership of that turn's
   * lifecycle. The steer therefore opens no admission slot and owns no
   * settlement — its output arrives on the original prompt's update stream.
   * @param params - `_session/steering` request for this session.
   * @param imageEnabled - connection capability advertised at initialization.
   * @param requestSignal - JSON-RPC request cancellation signal.
   * @returns the shared steering outcome vocabulary.
   */
  async steer(
    params: SteerRequest,
    imageEnabled: boolean,
    requestSignal?: AbortSignal,
  ): Promise<SteerResponse> {
    this.assertActive()
    const inflight = this.inflight
    if (inflight === undefined) return { outcome: 'promptRequired', reason: 'noRunningTurn' }
    const selection = this.agent.model.current()
    let content: Awaited<ReturnType<typeof admitAcpPrompt>>
    try {
      content = await admitAcpPrompt(
        this.ctx,
        selection,
        params.prompt,
        imageEnabled,
        requestSignal ?? new AbortController().signal,
      )
    } catch (error: unknown) {
      if (error instanceof AcpContentError) {
        throw error.kind === 'invalid'
          ? invalidParams(error.message)
          : internalError(error.message)
      }
      throw error
    }
    // Admission can await image persistence, during which the turn may well have
    // ended. Re-check rather than inject into a session that is no longer running.
    if (this.inflight !== inflight || this.agent.status === 'idle') {
      return { outcome: 'promptRequired', reason: 'noRunningTurn' }
    }
    const message = createUserMessage({
      content,
      source: { kind: 'user' },
    })
    try {
      this.agent.steer(message)
    } catch (error: unknown) {
      throw internalError(`steering was not queued: ${errorChain(error)}`)
    }
    return { outcome: 'injected' }
  }

  /** Cancel the active prompt, or autonomous work when no ACP prompt exists. */
  cancel(): void {
    const inflight = this.inflight
    this.cancelPrompt('ACP prompt cancelled')
    if (inflight === undefined) this.agent.cancel({ kind: 'user' })
  }

  /**
   * Process one durable event and enqueue its standard ACP projections.
   * @param session - exact event-owning Session.
   * @param event - committed durable event.
   */
  onSessionEvent(session: Session, event: SessionEvent): void {
    if (event.type === 'turn/start' && this.inflight?.turn !== undefined) {
      this.inflight.turn = event.data.turn
      this.inflight.endReason = undefined
    }
    try {
      if (event.type === 'assistant/message') {
        const inflight = this.inflight?.turn === event.data.turn ? this.inflight : undefined
        const previous = this.outputTail
        const delivery = previous.then(async () => {
          for (const update of await assistantUpdates(this.ctx, session, event)) {
            await this.notify({ sessionId: this.agent.session.id, update })
          }
        })
        this.outputTail = delivery.catch((error: unknown) => {
          const failure = error as Error
          if (inflight !== undefined) inflight.outputError ??= failure
          this.ctx.logger.warn(`acp: assistant output conversion failed: ${errorChain(error)}`)
        })
      } else if (event.type === 'tool/call') {
        const previous = this.outputTail
        this.outputTail = previous
          .then(() => this.notify({ sessionId: this.agent.session.id, update: toolCallUpdate(event) }))
          /* v8 ignore start -- the bridge notifier contains transport rejection. */
          .catch((error: unknown) => {
            this.ctx.logger.warn(`acp: tool-call update delivery failed: ${errorChain(error)}`)
          })
        /* v8 ignore stop */
      } else if (event.type === 'tool/result') {
        const previous = this.outputTail
        this.outputTail = previous
          .then(async () => this.notify({
            sessionId: this.agent.session.id,
            update: await toolResultUpdate(this.ctx, event),
          }))
          /* v8 ignore start -- supplemental-content conversion failure is contained and cannot fail Agent work. */
          .catch((error: unknown) => {
            this.ctx.logger.warn(`acp: tool-result update delivery failed: ${errorChain(error)}`)
          })
        /* v8 ignore stop */
      }
    } finally {
      const inflight = this.inflight
      if (inflight !== undefined && event.type === 'turn/end' && inflight.turn === event.data.turn) {
        inflight.endReason = event.data.reason
        inflight.agentError = undefined
      }
    }
  }

  /**
   * Correlate an accepted user message with its Agent turn.
   * @param message - claimed durable inbox message.
   * @param turn - allocated Agent turn.
   */
  onInboxClaimed(message: UserMessage, turn: number): void {
    if (this.inflight !== undefined && this.inflight.messageId === message.id) this.inflight.turn = turn
  }

  /**
   * Correlate an Agent interval failure with the active ACP prompt.
   * @param error - original same-process failure.
   */
  onAgentError(error: unknown): void {
    const inflight = this.inflight
    if (inflight === undefined || !inflight.messageQueued) return
    // A later committed turn/end supersedes this live failure. Failed turn
    // start/end appends have no terminal record and keep the failure observable.
    inflight.agentError = new Error(errorChain(error))
    this.settleAfterQuiescence(inflight)
  }

  /**
   * Complete cancelled activity without following a later root driver.
   * @param status - the emitted transition, which may precede a reentrant wake.
   */
  onAgentStatus(status: AgentStatus): void {
    const inflight = this.inflight
    if (inflight === undefined || !inflight.messageQueued || !inflight.admissionController.signal.aborted) return
    // Maintenance remains publicly idle; its next running transition belongs
    // to work released after the maintenance task finished.
    if (status === 'idle' || inflight.cancelStartedIdle) inflight.cancellationDone.resolve(this.outputTail)
  }

  /** Await every update queued before this call. */
  drainUpdates(): Promise<void> {
    return this.outputTail
  }

  /**
   * Cancel, drain, flush, and dispose this session once.
   * @param detail - cancellation detail for any prompt still in admission.
   * @returns the shared quiescent teardown promise.
   */
  close(detail: string): Promise<void> {
    if (this.closing !== undefined) return this.closing
    this.closing = (async () => {
      const failures: unknown[] = []
      const inflight = this.inflight
      this.cancelPrompt(detail)
      if (inflight === undefined || !inflight.messageQueued) this.agent.cancel({ kind: 'user' })
      try {
        await inflight?.admissionDone
        await this.agent.whenIdle()
        await this.outputTail
      } catch (error: unknown) {
        failures.push(new Error('ACP session activity drain failed', { cause: error }))
      }
      const subagents = this.ctx.get('subagents') as SubagentLifecycle | undefined
      try {
        await subagents?.drainDescendants([this.agent])
      } catch (error: unknown) {
        this.ctx.logger.warn(`acp: subagent teardown failed: ${errorChain(error)}`)
        failures.push(new Error('subagent teardown failed', { cause: error }))
      }
      try {
        await this.ctx.sessions.flush(this.agent.session)
      } catch (error: unknown) {
        failures.push(new Error('ACP session persistence flush failed', { cause: error }))
      }
      try {
        await this.disposeAgent()
      } catch (error: unknown) {
        failures.push(error)
      }
      if (failures.length === 1) throw failures[0]
      /* v8 ignore start -- independent teardown failures can aggregate only under multiple simultaneous provider faults. */
      if (failures.length > 1) {
        throw new AggregateError(failures, `ACP session teardown failed: ${failures.map(errorChain).join('; ')}`)
      }
      /* v8 ignore stop */
    })()
    return this.closing
  }

  private assertActive(): void {
    if (this.closing !== undefined) throw invalidParams(`session is closing: ${this.agent.session.id}`)
  }

  private cancelPrompt(detail: string): void {
    const inflight = this.inflight
    if (inflight === undefined) return
    const firstCancellation = !inflight.admissionController.signal.aborted
    if (firstCancellation) inflight.cancelStartedIdle = this.agent.status === 'idle'
    inflight.admissionController.abort(new Error(detail))
    this.settleAfterQuiescence(inflight)
    if (!inflight.messageQueued) {
      inflight.cancellationDone.resolve()
      return
    }
    this.agent.cancel({ kind: 'user' })
    if (firstCancellation) {
      const done = inflight.cancellationDone
      // Plugin disposal can detach status observers before this activity retires.
      void this.agent.whenIdle().then(() => { done.resolve(this.outputTail) }, done.reject)
    }
  }

  private settleAfterQuiescence(inflight: InflightPrompt): void {
    if (inflight.settlementStarted) return
    inflight.settlementStarted = true
    void (async () => {
      await inflight.admissionDone
      if (inflight.messageQueued) {
        const subagents = this.ctx.get('subagents') as SubagentLifecycle | undefined
        const cancelled = inflight.cancellationDone.promise
        const isCancelled = (): boolean => inflight.admissionController.signal.aborted
        const idleAt = (seq: Session['seq']): boolean => this.agent.status === 'idle' && this.agent.session.seq === seq
        while (true) {
          await Promise.race([this.agent.whenIdle(), cancelled])
          if (isCancelled()) break
          const idleSeq = this.agent.session.seq
          const children = await Promise.race([subagents?.waitForTreeWork(this.agent), cancelled])
          if (isCancelled()) break
          if (children || !idleAt(idleSeq)) continue
          await this.outputTail
          if (isCancelled() || idleAt(idleSeq)) break
        }
        if (isCancelled()) await cancelled
      }
      /* v8 ignore next -- this prompt owns the slot until this exact settlement clears it. */
      if (this.inflight !== inflight) return
      this.inflight = undefined
      if (inflight.admissionController.signal.aborted) {
        inflight.resolve('cancelled')
        return
      }
      if (inflight.outputError !== undefined) {
        inflight.reject(internalError(`assistant output delivery failed: ${inflight.outputError.message}`))
        return
      }
      if (inflight.agentError !== undefined) {
        inflight.reject(internalError(`turn failed: ${inflight.agentError.message}`))
        return
      }
      const end = inflight.endReason
      if (end === undefined) {
        inflight.resolve('cancelled')
      } else if (end.kind === 'error') {
        inflight.reject(internalError(`turn failed: ${end.error.message}`))
      } else {
        inflight.resolve(turnEndToStopReason(end))
      }
    })()
      /* v8 ignore start -- admissionDone only resolves; idle/output gates contain their own failures. */
      .catch((error: unknown) => {
        if (this.inflight !== inflight) return
        this.inflight = undefined
        inflight.reject(internalError(`prompt settlement failed: ${errorChain(error)}`))
      })
    /* v8 ignore stop */
  }
}
