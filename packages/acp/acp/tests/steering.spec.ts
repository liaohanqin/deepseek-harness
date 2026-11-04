import { PROTOCOL_VERSION } from '@agentclientprotocol/sdk'
import { SessionId } from '@deepseek-ai/dsh-session'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { textResponse } from './harness.ts'
import { makeBridgeHarness, type BridgeHarness } from './harness.ts'

async function newSession(harness: BridgeHarness): Promise<string> {
  await harness.client.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })
  return (await harness.client.newSession({ cwd: process.cwd(), mcpServers: [] })).sessionId
}

async function running(harness: BridgeHarness, sessionId: string): Promise<void> {
  await vi.waitFor(() => {
    expect(harness.ctx.agents.get(SessionId(sessionId))?.status).toBe('running')
  })
}

describe('acp steering', () => {
  let harness: BridgeHarness

  afterEach(async () => { await harness?.dispose() })

  it('advertises steering in the initialize response meta', async () => {
    harness = await makeBridgeHarness()
    const response = await harness.client.initialize({
      protocolVersion: PROTOCOL_VERSION,
      clientCapabilities: {},
    })
    expect(response._meta).toEqual({ steering: { supported: true } })
  })

  it('injects into the running turn instead of opening another', async () => {
    harness = await makeBridgeHarness({ script: ['hang'] })
    const sessionId = await newSession(harness)
    const first = harness.client.prompt({ sessionId, prompt: [{ type: 'text', text: 'one' }] })
    await running(harness, sessionId)

    await expect(harness.client.steer({ sessionId, prompt: [{ type: 'text', text: 'redirect' }] }))
      .resolves.toEqual({ outcome: 'injected' })

    // The steer owns no settlement: the original prompt is still the only
    // request in flight, so a second prompt is still refused.
    await expect(harness.client.prompt({ sessionId, prompt: [{ type: 'text', text: 'two' }] }))
      .rejects.toThrow(/already in flight/)

    await harness.client.cancel({ sessionId })
    await expect(first).resolves.toEqual({ stopReason: 'cancelled' })
  })

  it('parks the content at the next-step boundary rather than a new turn', async () => {
    harness = await makeBridgeHarness({ script: ['hang'] })
    const sessionId = await newSession(harness)
    const agent = harness.ctx.agents.get(SessionId(sessionId))!
    const first = harness.client.prompt({ sessionId, prompt: [{ type: 'text', text: 'one' }] })
    await running(harness, sessionId)

    await expect(harness.client.steer({ sessionId, prompt: [{ type: 'text', text: 'redirect' }] }))
      .resolves.toEqual({ outcome: 'injected' })

    // Steering targets the running turn's next step; a follow-up would have
    // landed in nextTurn and opened a turn of its own.
    await vi.waitFor(() => {
      expect(JSON.stringify(agent.inbox.nextStep)).toContain('redirect')
    })
    expect(JSON.stringify(agent.inbox.nextTurn)).not.toContain('redirect')

    await harness.client.cancel({ sessionId })
    await expect(first).resolves.toEqual({ stopReason: 'cancelled' })
  })

  it('reports promptRequired when the session is idle', async () => {
    harness = await makeBridgeHarness({ script: [] })
    const sessionId = await newSession(harness)

    await expect(harness.client.steer({ sessionId, prompt: [{ type: 'text', text: 'redirect' }] }))
      .resolves.toEqual({ outcome: 'promptRequired', reason: 'noRunningTurn' })
    // The content stayed unconsumed, so no model request was issued.
    expect(harness.adapter.requests).toHaveLength(0)
  })

  it('reports promptRequired after the running turn already settled', async () => {
    harness = await makeBridgeHarness({ script: [textResponse('done')] })
    const sessionId = await newSession(harness)
    await harness.client.prompt({ sessionId, prompt: [{ type: 'text', text: 'one' }] })
    await vi.waitFor(() => {
      expect(harness.ctx.agents.get(SessionId(sessionId))?.status).toBe('idle')
    })

    await expect(harness.client.steer({ sessionId, prompt: [{ type: 'text', text: 'redirect' }] }))
      .resolves.toEqual({ outcome: 'promptRequired', reason: 'noRunningTurn' })
  })

  it('rejects an unknown session', async () => {
    harness = await makeBridgeHarness()
    await harness.client.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} })
    await expect(harness.client.steer({ sessionId: 'missing', prompt: [{ type: 'text', text: 'x' }] }))
      .rejects.toThrow(/unknown session/)
  })
})
