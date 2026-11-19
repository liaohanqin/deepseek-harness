import { describe, expect, it } from 'vitest'
import { SessionSeq } from '@deepseek-ai/dsh-session'
import type { SessionEvent, SessionEventType } from '@deepseek-ai/dsh-session/types'
import { latestCompletedPrefixBoundary } from '../src/fork.ts'

/** A log entry carrying only the fields the boundary rule reads. */
function pick<T extends SessionEventType>(
  type: T,
  seq: number,
  extra: Partial<SessionEvent<T>> = {},
): SessionEvent {
  return { type, seq: SessionSeq(seq), time: 1, data: {}, ...extra } as SessionEvent
}

describe('latestCompletedPrefixBoundary', () => {
  it('returns undefined for an empty log and for a log with no completed turn', () => {
    expect(latestCompletedPrefixBoundary([])).toBeUndefined()
    expect(latestCompletedPrefixBoundary([
      pick('turn/start', 0),
      pick('user/message', 1),
    ])).toBeUndefined()
  })

  it('returns the last turn/end when nothing follows it', () => {
    expect(latestCompletedPrefixBoundary([
      pick('turn/start', 0),
      pick('user/message', 1),
      pick('turn/end', 2),
    ])).toBe(2)
  })

  it('advances past trailing log-only events but stops at the next turn start', () => {
    expect(latestCompletedPrefixBoundary([
      pick('turn/start', 0),
      pick('turn/end', 1),
      pick('session/title', 2),
      pick('step/start', 3),
      pick('turn/start', 4),
      pick('user/message', 5),
    ])).toBe(3)
  })

  it('stops at an appended user message and at an inbox splice', () => {
    expect(latestCompletedPrefixBoundary([
      pick('turn/start', 0),
      pick('turn/end', 1),
      pick('session/title', 2),
      pick('user/message', 3, { surfaceOp: 'append' }),
    ])).toBe(2)
    expect(latestCompletedPrefixBoundary([
      pick('turn/start', 0),
      pick('turn/end', 1),
      pick('agent/inbox/spliced', 2),
    ])).toBe(1)
  })
})
