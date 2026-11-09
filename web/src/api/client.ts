// One fetch wrapper for every API call (docs/api_design.md §2, §4.8, §6). Same origin through nginx:
// only relative /api/ paths, the session cookie sent by the browser (credentials 'same-origin'), and no
// Origin, Cookie or auth header set here; the browser adds Origin to POST, PUT and DELETE itself, which
// the CSRF check needs. JSON bodies get Content-Type application/json; a FormData upload is passed as is
// so the browser sets the multipart boundary.
//
// Expired session: a session_expired answer calls the handler session state registers, so the app can
// send the analyst to sign-in. Not for the /api/session calls, which handle their own 401, and not for
// POST /api/analyses, whose expired session is an Analyze error state with "Sign in again" (HANDOFF S5).
import { ApiError } from './errors'

export type ApiMethod = 'GET' | 'POST' | 'PUT' | 'DELETE'

export interface ApiRequest {
  method?: ApiMethod
  json?: unknown
  body?: FormData
}

let sessionExpiredHandler: (() => void) | null = null

export function setSessionExpiredHandler(handler: (() => void) | null): void {
  sessionExpiredHandler = handler
}

// '.' and '..', also percent-encoded: the URL parser treats %2e as a dot.
const DOT_SEGMENT = /^(?:\.|%2e){1,2}$/i

function hasForbiddenChar(path: string): boolean {
  return [...path].some((char) => {
    const code = char.codePointAt(0)!
    return code < 0x20 || code === 0x7f || /[\s#\\]/.test(char)
  })
}

export function checkApiPath(path: string): void {
  const pathname = path.split('?', 1)[0]!
  const valid =
    path.startsWith('/api/') &&
    !path.includes('//') &&
    !hasForbiddenChar(path) &&
    !pathname.split('/').some((segment) => DOT_SEGMENT.test(segment))
  if (!valid) throw new Error(`apiFetch: not an /api/ path: ${JSON.stringify(path)}`)
}

function reportsExpiry(method: ApiMethod, path: string): boolean {
  const pathname = path.split('?', 1)[0]
  if (pathname === '/api/session') return false
  return !(method === 'POST' && pathname === '/api/analyses')
}

export async function apiFetch<T>(path: string, request: ApiRequest = {}): Promise<T> {
  checkApiPath(path)
  if (request.json !== undefined && request.body !== undefined) {
    throw new Error('apiFetch: give json or body, not both')
  }
  const method = request.method ?? 'GET'
  const headers: Record<string, string> = { Accept: 'application/json' }
  let body: BodyInit | undefined = request.body
  if (request.json !== undefined) {
    headers['Content-Type'] = 'application/json'
    body = JSON.stringify(request.json)
  }

  let response: Response
  try {
    response = await fetch(path, { method, headers, body, credentials: 'same-origin' })
  } catch (error) {
    if (error instanceof TypeError) throw ApiError.network()
    throw error
  }

  if (!response.ok) {
    const error = await ApiError.fromResponse(response)
    if (error.code === 'session_expired' && reportsExpiry(method, path)) sessionExpiredHandler?.()
    throw error
  }
  if (response.status === 204) return undefined as T
  try {
    return (await response.json()) as T
  } catch (error) {
    // The connection dropped while the body was read, or a 2xx that is not JSON (e.g. index.html).
    if (error instanceof TypeError) throw ApiError.network()
    throw ApiError.http(response.status, undefined)
  }
}
