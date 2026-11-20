import { describe, expect, it } from 'vitest'
import { SessionSeq } from '@deepseek-ai/dsh-session'
import type { SessionEvent } from '@deepseek-ai/dsh-session'
import { latestCompletedPrefixBoundary } from '@deepseek-ai/dsh-agent'

/**
 * Build one log entry.
 *
 * The rule reads only `type`, `seq`, and a message's `surfaceOp`, so a fixture
 * places the discriminant it needs and leaves the payload to the shape the
 * rule never inspects.
 * @param type - event discriminant to place at this seq.
 * @param seq - contiguous position in the fixture log.
 * @param extra - fields the rule reads beyond the discriminant.
 * @returns one log entry for the fixture.
 */
function entry(
  type: SessionEvent['type'],
  seq: number,
  extra: { surfaceOp?: 'append' } = {},
): SessionEvent {
  return { type, seq: SessionSeq(seq), time: 1, data: {}, ...extra } as SessionEvent
}

describe('latestCompletedPrefixBoundary', () => {
  it('returns undefined for an empty log and for a log with no completed turn', () => {
    expect(latestCompletedPrefixBoundary([])).toBeUndefined()
    expect(latestCompletedPrefixBoundary([
      entry('turn/start', 0),
      entry('user/message', 1, { surfaceOp: 'append' }),
    ])).toBeUndefined()
  })

  it('returns the last turn/end when nothing follows it', () => {
    expect(latestCompletedPrefixBoundary([
      entry('turn/start', 0),
      entry('user/message', 1, { surfaceOp: 'append' }),
      entry('turn/end', 2),
    ])).toBe(2)
  })

  it('advances past trailing log-only events but stops at the next turn start', () => {
    expect(latestCompletedPrefixBoundary([
      entry('turn/start', 0),
      entry('turn/end', 1),
      entry('session/title', 2),
      entry('step/start', 3),
      entry('turn/start', 4),
      entry('user/message', 5, { surfaceOp: 'append' }),
    ])).toBe(3)
  })

  it('stops at an appended user message and at an inbox splice', () => {
    expect(latestCompletedPrefixBoundary([
      entry('turn/start', 0),
      entry('turn/end', 1),
      entry('session/title', 2),
      entry('user/message', 3, { surfaceOp: 'append' }),
    ])).toBe(2)
    expect(latestCompletedPrefixBoundary([
      entry('turn/start', 0),
      entry('turn/end', 1),
      entry('agent/inbox/spliced', 2),
    ])).toBe(1)
  })
})
