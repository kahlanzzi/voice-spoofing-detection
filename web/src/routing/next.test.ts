import { describe, expect, it } from 'vitest'
import { NEXT_MAX_LENGTH, safeNext, signInPath } from './next'

// safeNext is the open-redirect defence for the sign-in `next`: the path must be a protected app route exactly;
// the query is refused only for control characters, whitespace and '#', and is otherwise kept byte-for-byte.

const ID = 'req_' + '0123456789abcdef'.repeat(2)

describe('accepted, returned unchanged', () => {
  it.each([
    '/analyze',
    '/history',
    `/history/${ID}`,
    '/overview',
    '/status',
    '/history?verdict=synthetic&verdict=uncertain',
    '/history?verdict=a&verdict=b',
    '/history?x=%2F%2F',
    '/history?x=%252F',
    '/history?x=%5C',
    '/history?t=09:14',
    '/history?v=synthetic,uncertain',
    '/history?q=req_8fa9&page=2',
    '/analyze?',
  ])('%s', (value) => {
    expect(safeNext(value)).toBe(value)
  })

  it(`a value of exactly ${NEXT_MAX_LENGTH} characters`, () => {
    const value = '/history?q=' + 'a'.repeat(NEXT_MAX_LENGTH - '/history?q='.length)
    expect(value).toHaveLength(NEXT_MAX_LENGTH)
    expect(safeNext(value)).toBe(value)
  })
})

describe('rejected → /analyze', () => {
  it.each([
    ['null', null],
    ['empty', ''],
    ['sign-in itself', '/signin'],
    ['sign-in with its own next', '/signin?next=/analyze'],
    ['the root', '/'],
    ['absolute URL', 'https://evil.test/analyze'],
    ['absolute same-looking URL', 'http://localhost:8001/analyze'],
    ['scheme-relative', '//evil.test'],
    ['scheme-relative to a route', '//evil.test/analyze'],
    ['slash backslash', '/\\evil.test'],
    ['two backslashes', '\\\\evil.test'],
    ['backslash in a route', '/analyze\\'],
    ['javascript scheme', 'javascript:alert(1)'],
    ['data scheme', 'data:text/html,x'],
    ['relative path', 'analyze'],
    ['encoded double slash', '/%2F%2Fevil.test'],
    ['encoded backslash', '/%5Cevil.test'],
    ['double-encoded slashes', '/%252F%252Fevil.test'],
    ['encoded route', '/%61nalyze'],
    ['dot segment', '/history/../analyze'],
    ['short request ID', `/history/${ID.slice(0, -1)}`],
    ['long request ID', `/history/${ID}0`],
    ['uppercase prefix', `/history/REQ_${ID.slice(4)}`],
    ['uppercase hex', `/history/req_${ID.slice(4).toUpperCase()}`],
    ['not an ID', '/history/foo'],
    ['empty ID', '/history/'],
    ['uppercase route', '/Analyze'],
    ['trailing slash', '/analyze/'],
    ['unknown route', '/admin'],
    ['NUL in the path', '/analyze\x00'],
    ['NUL in the query', '/analyze?x=\x00'],
    ['US in the query', '/analyze?x=\x1f'],
    ['DEL in the query', '/analyze?x=\x7f'],
    ['newline in the query', '/analyze?x=\n'],
    ['tab in the query', '/analyze?x=\t'],
    ['space in the query', '/analyze?x= y'],
    ['no-break space in the query', '/analyze?x= '],
    ['fragment', '/analyze#x'],
    ['fragment after a query', '/analyze?x=1#y'],
  ])('%s', (_label, value) => {
    expect(safeNext(value)).toBe('/analyze')
  })

  it(`a value of ${NEXT_MAX_LENGTH + 1} characters`, () => {
    const value = '/history?q=' + 'a'.repeat(NEXT_MAX_LENGTH + 1 - '/history?q='.length)
    expect(value).toHaveLength(NEXT_MAX_LENGTH + 1)
    expect(safeNext(value)).toBe('/analyze')
  })
})

describe('signInPath → URLSearchParams.get → safeNext round trip', () => {
  // History filters built from docs/api_design.md §4.4 (repeated verdict, q, from/to, page), in mixed order.
  // §4.4 takes from/to as exactly YYYY-MM-DD; the ':' times here check that next keeps the query as it is,
  // not that the API accepts it.
  it.each([
    '',
    '?verdict=synthetic',
    '?verdict=synthetic&verdict=uncertain&verdict=authentic',
    '?page=2&verdict=uncertain&q=req_8fa9&verdict=synthetic',
    '?from=2026-10-01&to=2026-10-31&page=3',
    '?from=2026-11-01T00:00&to=2026-11-01T09:14&verdict=authentic',
    '?q=REQ_8FA9&verdict=synthetic&from=2026-10-30&to=2026-10-30&page=1',
    '?q=a%2Bb&verdict=synthetic',
    '?q=req+8fa9',
  ])('/history%s', (search) => {
    const target = signInPath('/history', search)
    expect(target.startsWith('/signin?next=')).toBe(true)
    const next = new URL(target, 'http://localhost:8001').searchParams.get('next')
    expect(next).toBe('/history' + search)
    expect(safeNext(next)).toBe('/history' + search)
  })

  it('a details path survives', () => {
    const next = new URL(signInPath(`/history/${ID}`, ''), 'http://localhost:8001').searchParams.get('next')
    expect(safeNext(next)).toBe(`/history/${ID}`)
  })
})
