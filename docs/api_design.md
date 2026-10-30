# API design: backend ↔ analyst web app (Stage 17)

- **Status:** Proposed, for team review. Nothing here is decided until the PR is approved.
- **Owner:** Iheb Zemzemi (Part E). **Reviewers (D26):** Donia Mabrouk, Malak Ben Salem, Khalil Dridi.
- **Serves:** the Stage 17 API (`api/`, FastAPI) and the Stage 18 web app (`web/`, React, D38).
- **Sources:** [HANDOFF.md](design/HANDOFF.md) §1, §11, §12; the
  [prototype v4](design/analyst-console-prototype.html); Protocol v1.0 §11
  ([experimental_protocol.md](experimental_protocol.md)); [architecture.md](architecture.md) §4–5;
  [ADR 0008](decisions/0008-web-frontend-react.md).

This note changes nothing in the §11 response, the ModelBundle or the risk policy. It turns
HANDOFF §11 item 10 (endpoints beyond §11) into a concrete proposal. Items marked
**Open decision** are not decided here.

## 1. Principles

- The API is the only source of verdicts, upload limits and statistics. The web app computes none
  of them and holds no risk-policy logic (ADR 0008 §3, §4, §8; NFR-08).
- Every result is decision support, not proof of fraud. The `notice` field says so on every result.
- Errors never carry a verdict: no `prediction`, `confidence` or `risk_level` in any error (NFR-06).
- No audio, original file name or IP address is ever stored (FR-09, NFR-04).

## 2. Conventions

- **Base path:** every endpoint is under `/api` (for example `POST /api/analyses`). The web app
  reaches it on the same origin through nginx, which serves the app and proxies `/api` to the
  `api` container. No CORS is needed or enabled.
- **Format:** JSON, field names in `snake_case`. File upload is the only multipart request.
- **Times:** UTC, ISO 8601 with `Z` (for example `2026-10-30T09:14:05Z`). **Dates** in queries:
  `YYYY-MM-DD`, read as days in the statistics time zone (section 9).
- **Request IDs:** made by the server, `req_` + 32 lowercase hex characters from a random UUID
  (the prototype shows a shortened form, `req_8fa98c02`). The browser never makes one.
- **Verdict values:** `synthetic`, `uncertain`, `authentic` (the §11 `prediction`).
  **Risk values:** `LOW`, `MEDIUM`, `HIGH`.

## 3. Endpoints at a glance

Paths are relative to `/api`.

| Method | Path | Purpose | Prototype screen |
|---|---|---|---|
| GET | `/status` | Service and bundle status | S10 System status; rail status line |
| GET | `/limits` | Accepted formats, size and duration caps | S2 Upload, S3 File selected |
| POST | `/analyses` | Analyze one recording | S4 Processing, S5 Error, S6 Result |
| GET | `/analyses` | Stored results, filtered and paged | S7 History |
| GET | `/analyses/{request_id}` | One stored result with its note | S8 Analysis details |
| PUT | `/analyses/{request_id}/note` | Save or remove the analyst note | S6 and S8 note panel |
| GET | `/stats` | Operational figures for a period | S9 Overview |
| — | sign-in | Placeholder, see section 4.8 | S1 Sign in |

## 4. Endpoints

### 4.1 `GET /status`

Request: no parameters.

Response `200`:

```json
{
  "state": "ready",
  "checked_at": "2026-10-30T09:14:05Z",
  "checks": { "analysis": "available", "history_and_stats": "available" },
  "versions": { "model_version": "mock", "profile_version": "preprocessing-v1", "policy_version": "risk-v1" },
  "bundle_manifest_sha256": null,
  "deferral_on_fit": 0.0993,
  "mock": true
}
```

- `state`: `ready`, `degraded` or `unavailable`. `unavailable` when analysis is unavailable.
  **Open decision 7:** the rule for `degraded` (prototype proposal: analysis works, History or
  statistics do not).
- The connection check on S10 is done by the browser: no answer means "cannot reach the service".
- `versions`, `bundle_manifest_sha256`, `deferral_on_fit` and `mock` come from the analyzer
  (section 8). With the MockAnalyzer the manifest hash is `null`.
- The endpoint reports problems in its body; it answers `200` even when `state` is `unavailable`.

Status codes: `200`; `500 internal_error`.

### 4.2 `GET /limits`

Request: no parameters.

Response `200`:

```json
{
  "formats": ["wav", "flac", "mp3"],
  "max_size_bytes": 20971520,
  "max_duration_s": 60,
  "audio_deletion": "Deleted when processing finishes"
}
```

- All four values come from config. **Open decision 2:** the real limits; the values above are the
  prototype mock.
- Size is in bytes so the browser compares it with `File.size` directly (the prototype mock used
  `max_size_mb`).
- The browser runs the same checks before upload, using these values only (NFR-08). If this call
  fails, the app keeps analysis off (S2 "limits unavailable").

Status codes: `200`; `500 internal_error`.

### 4.3 `POST /analyses`

Request: `multipart/form-data` with one part, `file`. The server reads only the extension from the
part's file name, then discards the name. The part's content type is ignored.

Response `201`, header `Location: /api/analyses/{request_id}`, body = **exactly the 8 §11 fields**:

```json
{
  "prediction": "uncertain",
  "confidence": 0.62,
  "risk_level": "MEDIUM",
  "model_version": "mock",
  "profile_version": "preprocessing-v1",
  "policy_version": "risk-v1",
  "request_id": "req_8fa98c02e4b14d0c9a1f3b7d6e5c2a10",
  "notice": "Decision support only. This result estimates whether the recording is synthetic. It does not prove fraud and does not verify identity."
}
```

**Processing order.** The request ID is assigned first, so every upload error carries it.

| Step | Check | On failure |
|---|---|---|
| 1 | Extension (lower case) is in `formats` | `415 unsupported_format` |
| 2 | Size cap (see the first note below); stream the file to a temp file in chunks and stop as soon as it passes `max_size_bytes` | `413 file_too_large` |
| 3 | File is not empty (more than 0 bytes) | `422 empty_file` |
| 4 | Decodes with `soundfile`; a file that decodes to 0 frames counts as empty | `422 corrupt_audio`; `422 empty_file` for 0 frames |
| 5 | Duration (frames ÷ sample rate) ≤ `max_duration_s` | `422 recording_too_long` |
| 6 | `analyzer.analyze(path)` within `analysis_timeout_s` | `503 analysis_unavailable`, `504 analysis_timeout`, `500 internal_error` |
| 7 | Store the analysis record (section 7) | `500 internal_error` |

- **Size protection (step 2):** Starlette reads the whole multipart body before the handler runs,
  so the handler's streamed check cannot stop a large upload on its own. The real protection is
  nginx's body-size limit plus a `Content-Length` check before the body is parsed. The streamed
  check in the handler is the second line of defence.
- The temp file has a random name in the configured temp folder. It is deleted when the request
  ends, on success and on every error (NFR-03). **Open decision 4:** what happens if deletion fails.
- No record is stored when any step fails. A verdict is returned only after the record is stored,
  so every shown result can be found again in History.
- On `504` the API stops waiting and stores nothing, even if the analyzer finishes later.
- Stage 18: the prototype's timeout screen must not say a late result will appear in History;
  with this contract a server timeout stores nothing.
- **Open decision 3:** how recordings longer than 2 s are handled (the model works on 2 s clips).
  That rule lives inside the analyzer, not in this endpoint.

Status codes: `201`; `400 invalid_request` (no `file` part); `413`; `415`; `422`; `401`/`403`
(section 4.8); `503`; `504`; `500`.

### 4.4 `GET /analyses`

Query parameters:

| Name | Meaning | Default |
|---|---|---|
| `verdict` | Repeatable: `?verdict=synthetic&verdict=uncertain`. None given = all verdicts | all |
| `from`, `to` | Inclusive day range (`YYYY-MM-DD`) | no limit |
| `q` | Part of a request ID, case-insensitive | none |
| `page` | Page number, from 1 | 1 |
| `page_size` | Items per page, 1 to `page_size_max` | 10 |

Response `200`:

```json
{
  "items": [ { "analysis": { "...8 §11 fields...": "" }, "created_at": "2026-10-30T09:14:05Z", "note": null } ],
  "total": 37,
  "page": 1,
  "page_size": 10,
  "counts": { "synthetic": 9, "uncertain": 6, "authentic": 22 }
}
```

- `items`: stored results (section 5), newest first.
- `total`: number of results matching all filters.
- `counts`: results per verdict matching `from`, `to` and `q`, **ignoring** the verdict filter, so
  each verdict chip shows how many results it would add (as in the prototype).
- A page past the last one returns `200` with empty `items` and the correct `total`.
- `page < 1`, `page_size` out of range, an unknown `verdict`, a malformed date, or `from` after
  `to` → `400 invalid_request`.
- Items include `confidence` because they carry the whole §11 object. The History table does not
  show it (HANDOFF §10).

Status codes: `200`; `400`; `401`/`403`; `500`.

### 4.5 `GET /analyses/{request_id}`

Response `200`: one stored result (section 5). Used by S8, and by S6 if the page is reloaded.

Status codes: `200`; `404 not_found` (unknown ID: S8 "unknown request ID"); `401`/`403`; `500`.

### 4.6 `PUT /analyses/{request_id}/note`

Request body: `{ "text": "Called back on the known number; caller confirmed." }`

Response `200`: `{ "note": { "text": "…", "updated_at": "2026-10-30T09:20:11Z" } }`

- One note per analysis; a new `PUT` replaces it.
- Empty text (after trimming spaces) removes the note; the response is `{ "note": null }`.
- Text longer than `note_max_chars` → `400 invalid_request`.
- The note is stored in its own table and never changes the analysis record (FR-11).

Status codes: `200`; `400`; `404 not_found`; `401`/`403`; `500`.

### 4.7 `GET /stats`

Query parameters: `from`, `to` (inclusive days). Without them: the last 7 days, including today.

Response `200`:

```json
{
  "from": "2026-10-24",
  "to": "2026-10-30",
  "time_zone": "UTC",
  "generated_at": "2026-10-30T09:14:05Z",
  "today": { "date": "2026-10-30", "total": 12 },
  "total": 214,
  "counts": { "synthetic": 41, "uncertain": 23, "authentic": 150 },
  "uncertain_share": 0.1075,
  "median_processing_ms": 840,
  "awaiting_note": { "total": 18, "recent": [ { "analysis": { "...": "" }, "created_at": "…", "note": null } ] },
  "days": [ { "date": "2026-10-24", "synthetic": 5, "uncertain": 3, "authentic": 20, "total": 28 } ],
  "monitoring_marker": { "deferral_on_fit": 0.0993, "policy_version": "risk-v1", "mock": true }
}
```

- `awaiting_note`: synthetic or uncertain results with no note; `recent` holds the 6 newest.
- `days` lists every day of the period, with zeros for empty days.
- `uncertain_share` = uncertain ÷ total; `null` when total is 0.
- A figure the service cannot compute is `null`; the app shows "Not available" for that figure
  only (S9 "one figure unavailable").
- `monitoring_marker` comes from the analyzer (section 8), never from a constant in code
  (ADR 0008 §8).
- **Open decision 6:** the time zone used to count days (config; UTC for now).
- No accuracy, EER or other quality metric, no per-analyst figure, nothing from DEEP-VOICE
  (ADR 0008 §8).
- Malformed date or `from` after `to` → `400 invalid_request`.

Status codes: `200`; `400`; `401`/`403`; `500`.

### 4.8 Sign-in (placeholder)

**Open decision 1:** the production authentication and access-control mechanism (SRS §7.4; threat
model §10.1). This note does not choose one. The decision must settle:

- the sign-in and sign-out endpoints and their request and response;
- which of the endpoints above need a signed-in session (possibly not `GET /status`);
- session lifetime, lockout ("temporarily locked" on S1) and password recovery;
- the error codes for S1's invalid-credentials, locked and sign-in-unavailable states;
- rate limits (threat model §10.1) are part of open decision 1.

Reserved now: `401 session_expired` (no valid session) and `403 access_denied` (signed in, not
allowed). Both use the error shape in section 6, with `request_id: null`.

## 5. Shapes

| Shape | Fields | Notes |
|---|---|---|
| `AnalysisResponse` | `prediction`, `confidence`, `risk_level`, `model_version`, `profile_version`, `policy_version`, `request_id`, `notice` | Exactly §11. Mirrored by the web app's TypeScript type (ADR 0008 §3). |
| `StoredAnalysis` | `analysis` (an `AnalysisResponse`), `created_at`, `note` (`{text, updated_at}` or `null`) | The §11 object stays whole; stored-only fields sit beside it. |
| `Error` | `error.code`, `error.message`, `error.request_id` | Section 6. |

`processing_ms` and the mock flag are stored but not returned per item; they feed `GET /stats`
and `GET /status`.

## 6. Errors

Every error uses one shape:

```json
{ "error": { "code": "corrupt_audio", "message": "The recording could not be decoded.", "request_id": "req_8fa98c02e4b14d0c9a1f3b7d6e5c2a10" } }
```

| Code | HTTP | When | Prototype state |
|---|---|---|---|
| `unsupported_format` | 415 | Extension not in `formats` | S3 check (browser), repeated by the API |
| `file_too_large` | 413 | Body passes `max_size_bytes` | S3 check, repeated by the API |
| `recording_too_long` | 422 | Duration passes `max_duration_s` | S3 check, repeated by the API |
| `empty_file` | 422 | 0 bytes, or decodes to 0 frames | S3 check, repeated by the API |
| `corrupt_audio` | 422 | `soundfile` cannot decode it | S5 Corrupt audio (input) |
| `analysis_unavailable` | 503 | Analyzer not loaded or refusing work | S5 Analysis service unavailable |
| `analysis_timeout` | 504 | Analyzer did not finish in `analysis_timeout_s` | S5 Request timeout |
| `session_expired` | 401 | No valid session (section 4.8) | S5 / S1 Expired session |
| `access_denied` | 403 | Session not allowed (section 4.8) | S1 Access denied |
| `not_found` | 404 | Unknown request ID or route | S8 Unknown request ID |
| `internal_error` | 500 | Anything unexpected, including a failed store | S5 Unexpected service error |
| `invalid_request` | 400 | Bad query or body (**proposal beyond the agreed list**) | none (the app prevents these) |

Rules:

- No error contains `prediction`, `confidence`, `risk_level` or any partial result.
- `request_id` is the upload's ID for errors from `POST /analyses` after the ID is assigned;
  otherwise `null`. A network failure in the browser has no request ID at all (HANDOFF §12).
- `message` is one plain sentence for logs and support. The web app shows its own wording per
  `code` (HANDOFF §8) and never displays raw server text.
- FastAPI's default error bodies (validation `422`, `HTTPException`, unhandled `500`) are replaced
  by this shape. No stack trace or internal path ever reaches the client.
- nginx must not answer with its own error pages for `/api`: its body-size limit is set above
  `max_size_bytes` and its proxy timeout above `analysis_timeout_s`, so the API's codes reach the app.

## 7. Storage

Two tables (SQLite on a volume first; architecture §7, open):

| Table | Columns |
|---|---|
| `analysis_record` | `request_id` (key), `created_at` (UTC), `prediction`, `confidence`, `risk_level`, `model_version`, `profile_version`, `policy_version`, `notice`, `processing_ms`, `is_mock` |
| `analysis_note` | `request_id` (key, refers to `analysis_record`), `text`, `updated_at` (UTC) |

- `processing_ms`: from upload received to result ready.
- `is_mock`: true for MockAnalyzer results, so they can be told apart from real ones.
- **Never stored:** audio, the original file name, the client IP address. The file name is not
  logged either.
- The operational log (FR-20, `service_log` in architecture §4) is not part of this contract.

## 8. Analyzer seam

One interface is the only place the API calls model code:

```text
analyze(path) -> AnalysisOutcome   # prediction, confidence, risk_level,
                                   # model_version, profile_version, policy_version
info() -> AnalyzerInfo             # ready, versions, deferral_on_fit,
                                   # bundle_manifest_sha256, mock
```

- **Now:** `MockAnalyzer`, with `model_version: "mock"` and `mock: true`. It returns fixed,
  plausible outcomes that follow `risk-v1` behaviour (as the prototype's mock data does) and reads
  its `deferral_on_fit` from config, labelled mock.
- **Later:** the team's frozen ModelBundle (Stage 12), loaded and verified by `src/bundle/`, which
  supplies `deferral_on_fit` from its risk policy. The API code does not change.
- The API adds `request_id` and `notice` to the outcome to make the §11 response.
- Analyzer not ready → `503 analysis_unavailable`; over `analysis_timeout_s` → `504`; any other
  exception → `500 internal_error`.

## 9. Configuration

All values come from config (`configs/api.yaml`, written in Stage 17), never from code.

| Key | Meaning | Value now | Status |
|---|---|---|---|
| `formats` | Accepted extensions | `wav`, `flac`, `mp3` | Mock; **open decision 2** |
| `max_size_bytes` | Upload size cap | 20,971,520 (20 MB) | Mock; **open decision 2** |
| `max_duration_s` | Recording duration cap | 60 | Mock; **open decision 2** |
| `audio_deletion` | Deletion statement shown on S2 | "Deleted when processing finishes" | Depends on **open decision 4** |
| `analysis_timeout_s` | How long the API waits for the analyzer | 30 | Proposal |
| `page_size_max` | Largest `page_size` | 50 | Proposal |
| `note_max_chars` | Longest note | 2,000 | Proposal |
| `stats_time_zone` | Time zone for counting days | `UTC` | **Open decision 6** |
| `notice_text` | The §11 `notice` text | prototype wording | Proposal |
| `mock_deferral_on_fit` | Monitoring marker for the MockAnalyzer only | 0.0993 | Mock; replaced by the bundle's value |
| `temp_dir` | Folder for temp uploads | — | Deployment setting |

`analysis_timeout_s` is not a latency target: **open decision 5** (latency target) stays open.

## 10. Open decisions and questions

From HANDOFF §11, as they touch the API:

1. Authentication and access control (section 4.8).
2. Real upload limits (section 9).
3. Recordings longer than 2 s (inside the analyzer).
4. Fallback if temp-file deletion fails.
5. Latency target.
6. Time zone for statistics.
7. Definition of "Degraded" (section 4.1).
9. Cancel analysis: no API change. Cancel stops the browser waiting; a request already accepted
   may still finish and be stored, and then shows in History.

For the reviewers of this note:

- **`invalid_request` (400)** is added beyond the agreed error list, for bad queries and bodies.
- **Decision record:** once approved, does this contract need a D-ID in `docs/decisions/README.md`
  (HANDOFF §11 item 10)? This note does not add one.
