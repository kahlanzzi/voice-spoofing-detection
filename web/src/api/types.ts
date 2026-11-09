// Request and response shapes of the analyst API (docs/api_design.md §2, §4, §5, §6). Field names are
// the API's (snake_case), checked field by field against api/schemas.py. AnalysisResponse is exactly
// the 8 §11 fields (ADR 0008 §3): a field is added or removed only through a contract change.

export type Verdict = 'synthetic' | 'uncertain' | 'authentic'

export type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH'

// `req_` + 32 lowercase hex characters, made by the server (§2).
export type RequestId = `req_${string}`

// UTC, ISO 8601 with Z, e.g. 2026-10-30T09:14:05Z (§2).
export type UtcTimestamp = string

// YYYY-MM-DD (§2).
export type IsoDate = string

export interface AnalysisResponse {
  prediction: Verdict
  confidence: number // 0 to 1; shown only in Technical details (HANDOFF §8)
  risk_level: RiskLevel
  model_version: string
  profile_version: string
  policy_version: string
  request_id: RequestId
  notice: string
}

export const ANALYSIS_RESPONSE_FIELDS = [
  'prediction',
  'confidence',
  'risk_level',
  'model_version',
  'profile_version',
  'policy_version',
  'request_id',
  'notice',
] as const satisfies readonly (keyof AnalysisResponse)[]

// §4.6
export interface Note {
  text: string
  updated_at: UtcTimestamp
  updated_by: string | null // null for notes saved before sign-in existed
}

export interface NoteRequest {
  text: string
}

export interface NoteResponse {
  note: Note | null
}

// §5
export interface StoredAnalysis {
  analysis: AnalysisResponse
  created_at: UtcTimestamp
  note: Note | null
}

export interface VerdictCounts {
  synthetic: number
  uncertain: number
  authentic: number
}

// §4.4
export interface HistoryQuery {
  verdict?: Verdict[]
  from?: IsoDate
  to?: IsoDate
  q?: string
  page?: number
  page_size?: number
}

export interface HistoryResponse {
  items: StoredAnalysis[]
  total: number
  page: number
  page_size: number
  counts: VerdictCounts
}

// §4.1
export type CheckState = 'available' | 'unavailable'

export interface StatusResponse {
  state: 'ready' | 'degraded' | 'unavailable'
  checked_at: UtcTimestamp
  checks: { analysis: CheckState; history_and_stats: CheckState }
  versions: { model_version: string; profile_version: string; policy_version: string }
  bundle_manifest_sha256: string | null
  deferral_on_fit: number
  mock: boolean
}

// §4.2
export interface LimitsResponse {
  formats: string[]
  max_size_bytes: number
  max_duration_s: number
  audio_deletion: string
}

// §4.7
export interface StatsQuery {
  from?: IsoDate
  to?: IsoDate
}

export interface StatsDay extends VerdictCounts {
  date: IsoDate
  total: number
}

export interface StatsResponse {
  from: IsoDate
  to: IsoDate
  time_zone: string
  generated_at: UtcTimestamp
  today: { date: IsoDate; total: number }
  total: number
  counts: VerdictCounts
  uncertain_share: number | null
  median_processing_ms: number | null
  awaiting_note: { total: number; recent: StoredAnalysis[] }
  days: StatsDay[]
  monitoring_marker: { deferral_on_fit: number; policy_version: string; mock: boolean }
}

// §4.8
export interface SignInRequest {
  username: string
  password: string
}

export interface SessionInfo {
  username: string
  signed_in_at: UtcTimestamp
  expires_at: UtcTimestamp
}

// §6: every code in the table, from the API and from nginx.
export const ERROR_CODES = [
  'unsupported_format',
  'file_too_large',
  'recording_too_long',
  'empty_file',
  'corrupt_audio',
  'analysis_unavailable',
  'analysis_timeout',
  'session_expired',
  'access_denied',
  'invalid_credentials',
  'account_locked',
  'too_many_requests',
  'sign_in_unavailable',
  'origin_not_allowed',
  'not_found',
  'internal_error',
  'invalid_request',
] as const

export type ErrorCode = (typeof ERROR_CODES)[number]

// The §6 body on the wire. `message` is for logs and support only; ApiError never keeps it.
export interface ErrorBody {
  error: { code: ErrorCode; message: string; request_id: RequestId | null }
}
