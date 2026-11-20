/**
 * Completed-turn cut selection shared by every Session fork consumer.
 * @module @deepseek-ai/dsh-agent/fork-boundary
 */

import type { SessionEvent, SessionSeq } from '@deepseek-ai/dsh-session'

/**
 * The inclusive seq of a source log's latest completed-turn prefix, or
 * `undefined` when the log holds no completed turn.
 *
 * The cut starts at the last `turn/end` and then advances past every trailing
 * log-only event. It stops at the first event that opens or contributes to
 * later work — a new turn, an appended user message, or an inbox splice —
 * because a fork through one of those would hand the child a half-processed
 * turn. Every fork consumer selects its boundary through this function so the
 * GUI and the automation surface cut at the same place.
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
