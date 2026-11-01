import { describe, expect, expectTypeOf, it } from 'vitest'
import { ApiError, isHttpError, isNetworkError, parseErrorBody } from './errors'
import type { ErrorCode, RequestId } from './types'

const ID = 'req_8fa98c02e4b14d0c9a1f3b7d6e5c2a10'
const SERVER_TEXT = 'The recording could not be decoded.'

function response(status: number, body: string, type = 'application/json'): Response {
  return new Response(body, { status, headers: { 'Content-Type': type } })
}

describe('parseErrorBody', () => {
  it('reads the §6 shape from the API', () => {
    const body = { error: { code: 'corrupt_audio', message: SERVER_TEXT, request_id: ID } }
    expect(parseErrorBody(body)).toEqual({ code: 'corrupt_audio', requestId: ID })
  })

  it('keeps a null request_id', () => {
    const body = { error: { code: 'session_expired', message: 'No valid session.', request_id: null } }
    expect(parseErrorBody(body)).toEqual({ code: 'session_expired', requestId: null })
  })

  it.each([
    ['not an object', 'oops'],
    ['null', null],
    ['an array', []],
    ['no error key', { detail: 'Not Found' }],
    ['error is a string', { error: 'not_found' }],
    ['unknown code', { error: { code: 'teapot', message: 'x', request_id: null } }],
    ['code missing', { error: { message: 'x', request_id: null } }],
    ['request_id missing', { error: { code: 'not_found', message: 'x' } }],
    ['request_id a number', { error: { code: 'not_found', message: 'x', request_id: 7 } }],
    ['request_id not the req_ form', { error: { code: 'not_found', message: 'x', request_id: 'req_ABC' } }],
  ])('%s → internal_error', (_, body) => {
    expect(parseErrorBody(body)).toEqual({ code: 'internal_error', requestId: null })
  })
})

describe('ApiError.fromResponse', () => {
  // nginx's own answers, verbatim from deploy/nginx.conf lines 113, 120, 127 and 134.
  it.each([
    [404, 'not_found', '{"error":{"code":"not_found","message":"The requested resource was not found.","request_id":null}}\n'],
    [413, 'file_too_large', '{"error":{"code":"file_too_large","message":"The file is larger than the size limit.","request_id":null}}\n'],
    [429, 'too_many_requests', '{"error":{"code":"too_many_requests","message":"Too many sign-in attempts from this address; try again later.","request_id":null}}\n'],
    [503, 'analysis_unavailable', '{"error":{"code":"analysis_unavailable","message":"The analysis service is unavailable.","request_id":null}}\n'],
  ])('nginx %i → %s', async (status, code, body) => {
    const error = await ApiError.fromResponse(response(status, body))
    expect(error).toMatchObject({ kind: 'http', status, code, requestId: null })
  })

  it('an upload error keeps its request ID', async () => {
    const body = JSON.stringify({ error: { code: 'corrupt_audio', message: SERVER_TEXT, request_id: ID } })
    const error = await ApiError.fromResponse(response(422, body))
    expect(error).toMatchObject({ kind: 'http', status: 422, code: 'corrupt_audio', requestId: ID })
  })

  it.each([
    ['an HTML page', '<html><body>Bad Gateway</body></html>', 'text/html'],
    ['an empty body', '', 'application/json'],
    ['truncated JSON', '{"error":{"code":', 'application/json'],
  ])('%s → internal_error', async (_, body, type) => {
    const error = await ApiError.fromResponse(response(502, body, type))
    expect(error).toMatchObject({ kind: 'http', status: 502, code: 'internal_error', requestId: null })
  })

  it('never keeps the server message', async () => {
    const body = JSON.stringify({ error: { code: 'corrupt_audio', message: SERVER_TEXT, request_id: ID } })
    const error = await ApiError.fromResponse(response(422, body))
    expect(error.message).toBe('API error 422 corrupt_audio')
    expect(JSON.stringify(error)).not.toContain(SERVER_TEXT)
    expect(Object.values(error)).not.toContain(SERVER_TEXT)
    expect(String(error)).not.toContain(SERVER_TEXT)
  })
})

describe('ApiError.network', () => {
  it('has no status, code or request ID', () => {
    const error = ApiError.network()
    expect(error).toBeInstanceOf(ApiError)
    expect(error).toMatchObject({ kind: 'network', status: undefined, code: null, requestId: undefined })
    expect(error.message).toBe('Network error')
  })
})

describe('isHttpError and isNetworkError', () => {
  const http = ApiError.http(422, { error: { code: 'corrupt_audio', message: SERVER_TEXT, request_id: ID } })
  const network = ApiError.network()

  it.each([
    ['an http ApiError', http, true, false],
    ['a network ApiError', network, false, true],
    ['a plain Error', new Error('x'), false, false],
    ['a TypeError', new TypeError('Failed to fetch'), false, false],
    ['an ApiError-shaped object', { kind: 'http', code: 'not_found', requestId: null, status: 404 }, false, false],
    ['null', null, false, false],
    ['undefined', undefined, false, false],
  ])('%s → http %s, network %s', (_, value, httpExpected, networkExpected) => {
    expect(isHttpError(value)).toBe(httpExpected)
    expect(isNetworkError(value)).toBe(networkExpected)
  })

  it('isHttpError narrows code, requestId and status', () => {
    const error: unknown = http
    if (!isHttpError(error)) throw new Error('expected an http error')
    expectTypeOf(error.kind).toEqualTypeOf<'http'>()
    expectTypeOf(error.code).toEqualTypeOf<ErrorCode>()
    expectTypeOf(error.requestId).toEqualTypeOf<RequestId | null>()
    expectTypeOf(error.requestId).toExtend<string | null>()
    expectTypeOf(error.status).toEqualTypeOf<number>()
    expect(error.code).toBe('corrupt_audio')
  })

  it('isNetworkError narrows code to null', () => {
    const error: unknown = network
    if (!isNetworkError(error)) throw new Error('expected a network error')
    expectTypeOf(error.kind).toEqualTypeOf<'network'>()
    expectTypeOf(error.code).toEqualTypeOf<null>()
    expectTypeOf(error.requestId).toEqualTypeOf<undefined>()
    expectTypeOf(error.status).toEqualTypeOf<undefined>()
    expect(error.code).toBeNull()
  })

  it('checking kind alone does not narrow code', () => {
    expectTypeOf<ApiError['code']>().toEqualTypeOf<ErrorCode | null>()
  })
})
