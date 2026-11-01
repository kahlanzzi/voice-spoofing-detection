// The one error type for API calls (docs/api_design.md §6). It keeps the status, the §6 code and the
// request ID, and never the server's `message`: screens show their own wording per code (HANDOFF §8),
// so no raw server text can reach them. A body that is not the §6 shape, from the API or from nginx,
// counts as internal_error. A network failure has no status, no code and no request ID (HANDOFF §12).
import { ERROR_CODES, type ErrorCode, type RequestId } from './types'

export type ApiErrorKind = 'http' | 'network'

const REQUEST_ID = /^req_[0-9a-f]{32}$/

interface ParsedError {
  code: ErrorCode
  requestId: RequestId | null
}

const MALFORMED: ParsedError = { code: 'internal_error', requestId: null }

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isErrorCode(value: unknown): value is ErrorCode {
  return ERROR_CODES.includes(value as ErrorCode)
}

function isRequestId(value: unknown): value is RequestId {
  return typeof value === 'string' && REQUEST_ID.test(value)
}

export function parseErrorBody(body: unknown): ParsedError {
  if (!isRecord(body) || !isRecord(body.error)) return MALFORMED
  const { code, request_id: requestId } = body.error
  if (!isErrorCode(code)) return MALFORMED
  if (requestId !== null && !isRequestId(requestId)) return MALFORMED
  return { code, requestId }
}

export class ApiError extends Error {
  readonly kind: ApiErrorKind
  readonly status: number | undefined
  // null only for kind 'network'.
  readonly code: ErrorCode | null
  // From the body: the upload's ID or null. undefined when there was no response.
  readonly requestId: RequestId | null | undefined

  private constructor(kind: ApiErrorKind, status?: number, parsed?: ParsedError) {
    super(kind === 'network' ? 'Network error' : `API error ${status} ${parsed?.code}`)
    this.name = 'ApiError'
    this.kind = kind
    this.status = status
    this.code = parsed?.code ?? null
    this.requestId = parsed?.requestId
  }

  static network(): ApiError {
    return new ApiError('network')
  }

  static http(status: number, body: unknown): ApiError {
    return new ApiError('http', status, parseErrorBody(body))
  }

  static async fromResponse(response: Response): Promise<ApiError> {
    let body: unknown
    try {
      body = await response.json()
    } catch {
      body = undefined
    }
    return ApiError.http(response.status, body)
  }
}
