/**
 * Fork boundary selection for the ACP bridge.
 *
 * `session/fork` carries no boundary on the wire, so the adapter picks the
 * cut itself. This module owns that single rule; the Host's Session
 * Controller applies the same one when a GUI client forks.
 * @module @deepseek-ai/dsh-acp/fork
 */

import type { SessionEvent, SessionSeq } from '@deepseek-ai/dsh-session/types'

/**
 * The inclusive seq of the source's latest completed-turn prefix, or
 * `undefined` when the log holds no completed turn.
 *
 * The cut starts at the last `turn/end` and then advances past every
 * trailing log-only event. It stops at the first event that opens or
 * contributes to later work — a new turn, an appended user message, or an
 * inbox splice — because forking through one of those would hand the child
 * a half-processed turn.
 *
 * @param events - a contiguous source log ordered by seq from zero.
 * @returns the inclusive boundary seq, or `undefined` when never completed.
 */
export function latestCompletedPrefixBoundary(events: readonly SessionEvent[]): SessionSeq | undefined {
  const lastTurnEnd = events.findLast(event => event.type === 'turn/end')
  if (lastTurnEnd === undefined) return undefined
  let boundary = lastTurnEnd.seq
  for (const next of events.slice(boundary + 1)) {
    if (next.type === 'turn/start' || (next.type === 'user/message' && next.surfaceOp === 'append')
      || next.type === 'agent/inbox/spliced') break
    boundary = next.seq
  }
  return boundary
}
