import { expect, it } from 'vitest'
import { formatCheckedAt } from './checkedAt'

// The prototype's format (line 644) in the browser's time zone; the test runs in whatever zone the machine has.

const ISO = '2026-11-03T09:14:05Z'

it('uses the prototype line 644 options in the browser zone', () => {
  const expected = new Intl.DateTimeFormat('en-GB', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    timeZoneName: 'shortOffset',
  }).format(new Date(ISO))
  expect(formatCheckedAt(ISO)).toBe(expected)
})

it('is absolute: the date, the time and the offset', () => {
  expect(formatCheckedAt(ISO)).toMatch(/^\d{1,2} [A-Z][a-z]{2} 2026, \d{2}:\d{2} GMT([+-]\d{1,2}(:\d{2})?)?$/)
})

it('shows a value that is not a date as given', () => {
  expect(formatCheckedAt('not a date')).toBe('not a date')
})
