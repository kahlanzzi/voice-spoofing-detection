import { afterEach, describe, expect, it, vi } from 'vitest'
import { apiFetch } from '../api/client'
import { ApiError } from '../api/errors'
import { SIGN_IN_COPY, signInOutcome, type SignInError } from './signInOutcome'

// The whole path: a raw answer through apiFetch and ApiError, then the S1 variant (docs/api_design.md §4.8
// "Errors for S1", §6).

function body(code: string, requestId: string | null = null): string {
  return JSON.stringify({ error: { code, message: 'Server text, never shown.', request_id: requestId } })
}

async function outcomeOf(response: Response | Error): Promise<SignInError> {
  const fetchMock = vi.fn<typeof fetch>()
  if (response instanceof Error) fetchMock.mockRejectedValue(response)
  else fetchMock.mockResolvedValue(response)
  vi.stubGlobal('fetch', fetchMock)
  try {
    await apiFetch('/api/session', { method: 'POST', json: { username: 'analyst01', password: 'x' } })
  } catch (error) {
    return signInOutcome(error)
  }
  throw new Error('expected a failure')
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('the four S1 errors', () => {
  it.each([
    [401, 'invalid_credentials', 'invalid'],
    // Pre-lookup and the same for every username (api/routers/session.py lines 87-96).
    [400, 'invalid_request', 'invalid'],
    [403, 'access_denied', 'denied'],
    [429, 'account_locked', 'locked'],
    [503, 'sign_in_unavailable', 'unavailable'],
  ])('API %i %s → %s', async (status, code, variant) => {
    expect(await outcomeOf(new Response(body(code), { status }))).toBe(variant)
  })
})

describe('everything else is unavailable, never a wrong password', () => {
  // nginx's own answers, verbatim from deploy/nginx.conf lines 113, 127 and 134.
  it.each([
    [429, '{"error":{"code":"too_many_requests","message":"Too many sign-in attempts from this address; try again later.","request_id":null}}\n'],
    [503, '{"error":{"code":"analysis_unavailable","message":"The analysis service is unavailable.","request_id":null}}\n'],
    [404, '{"error":{"code":"not_found","message":"The requested resource was not found.","request_id":null}}\n'],
  ])('nginx %i', async (status, raw) => {
    expect(await outcomeOf(new Response(raw, { status }))).toBe('unavailable')
  })

  it('a network error', async () => {
    expect(await outcomeOf(new TypeError('Failed to fetch'))).toBe('unavailable')
  })

  it.each([
    [401, 'session_expired'],
    [401, 'access_denied'],
    [403, 'origin_not_allowed'],
    [403, 'invalid_credentials'],
    [429, 'invalid_credentials'],
    [400, 'empty_file'],
    [422, 'invalid_request'],
    [422, 'corrupt_audio'],
    [405, 'invalid_request'],
    [500, 'internal_error'],
    [504, 'analysis_timeout'],
    [418, 'invalid_credentials'],
  ])('%i %s', async (status, code) => {
    expect(await outcomeOf(new Response(body(code), { status }))).toBe('unavailable')
  })

  it.each([
    ['HTML', '<html>502</html>'],
    ['empty', ''],
    ['unknown code', body('wrong_password')],
    ['not the §6 shape', '{"detail":"Invalid credentials"}'],
  ])('401 with a malformed body (%s)', async (_label, raw) => {
    expect(await outcomeOf(new Response(raw, { status: 401 }))).toBe('unavailable')
  })

  it('a 2xx that is not JSON', async () => {
    expect(await outcomeOf(new Response('<!doctype html>', { status: 200 }))).toBe('unavailable')
  })

  it.each([
    ['a plain Error', new Error('boom')],
    ['a string', 'boom'],
    ['undefined', undefined],
  ])('anything thrown that is not an ApiError (%s)', (_label, thrown) => {
    expect(signInOutcome(thrown)).toBe('unavailable')
  })

  it('ApiError.network() directly', () => {
    expect(signInOutcome(ApiError.network())).toBe('unavailable')
  })
})

it('the copy is the prototype text (lines 834-837), one per variant', () => {
  expect(SIGN_IN_COPY).toEqual({
    invalid: 'The analyst ID or password is not correct. Check both and try again.',
    denied: 'This account does not have access to the analyst console. Ask your administrator for access.',
    locked: 'This account is temporarily locked. Try again later, or ask your administrator.',
    unavailable: 'Sign-in is temporarily unavailable. Try again in a few minutes.',
  })
})
