import { expect, it, vi } from 'vitest'
import { formatAnalyzedAt, timeZoneLabel } from './analyzedAt'

// A fixed zone that is not UTC, set before the formatters are built, so a hard-coded UTC fails on any machine: CI runs
// in UTC, where the browser zone and UTC print the same (Entry 147). The expected values are literals.
vi.hoisted(() => {
  vi.stubEnv('TZ', 'Africa/Tunis')
})

it('really runs in Africa/Tunis, so the literals below mean something', () => {
  expect(Intl.DateTimeFormat().resolvedOptions().timeZone).toBe('Africa/Tunis')
  expect(new Date('2026-11-03T09:14:05Z').getTimezoneOffset()).toBe(-60)
})

it('shows created_at in the browser zone, in the format of prototype line 645, with no offset', () => {
  expect(formatAnalyzedAt('2026-11-03T09:14:05Z')).toBe('3 Nov 2026, 10:14')
})

it('moves to the next day when the browser zone does', () => {
  expect(formatAnalyzedAt('2026-11-03T23:30:00Z')).toBe('4 Nov 2026, 00:30')
})

it('names the browser zone by its offset (prototype line 655)', () => {
  expect(timeZoneLabel()).toBe('GMT+1')
})
