import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// A fresh module per test (vi.resetModules), as in limits.test.ts: the store has no reset function. The session and
// draft stores come from the same fresh registry, so the subscriptions are fresh too.

type Analysis = typeof import('./analysis')
type Draft = typeof import('./fileDraft')
type Session = typeof import('../session/session')

let analysis: Analysis
let draft: Draft
let session: Session
let fetchMock: ReturnType<typeof vi.fn<typeof fetch>>

const ID = `req_${'a'.repeat(32)}`
const FILE = new File(['audio bytes'], 'call-with-a-customer.wav', { type: 'audio/wav' })

const BODY = {
  prediction: 'uncertain',
  confidence: 0.62,
  risk_level: 'MEDIUM',
  model_version: 'mock',
  profile_version: 'preprocessing-v1',
  policy_version: 'risk-v1',
  request_id: ID,
  notice: 'Decision support only.',
}

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

function apiError(status: number, code: string, requestId: string | null = ID): Response {
  return json(status, { error: { code, message: 'server text', request_id: requestId } })
}

function deferred(): { promise: Promise<Response>; resolve: (response: Response) => void } {
  let resolve: (response: Response) => void = () => undefined
  const promise = new Promise<Response>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

// The form the last POST carried.
function sentForm(): FormData {
  return fetchMock.mock.calls.at(-1)![1]!.body as FormData
}

const toEqualResult = () => ({ status: 'result', analysis: BODY })

beforeEach(async () => {
  vi.resetModules()
  fetchMock = vi.fn<typeof fetch>()
  vi.stubGlobal('fetch', fetchMock)
  analysis = await import('./analysis')
  draft = await import('./fileDraft')
  session = await import('../session/session')
  session.markSignedIn('analyst01')
  draft.setFileDraft(FILE)
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('the request', () => {
  beforeEach(() => {
    fetchMock.mockImplementation(async () => json(201, BODY))
  })

  it('posts once to /api/analyses with POST and same-origin credentials', async () => {
    await analysis.startAnalysis(FILE)
    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [path, init] = fetchMock.mock.calls[0]!
    expect(path).toBe('/api/analyses')
    expect(init?.method).toBe('POST')
    expect(init?.credentials).toBe('same-origin')
  })

  it('the form has exactly one entry, named file', async () => {
    await analysis.startAnalysis(FILE)
    expect([...sentForm().keys()]).toEqual(['file'])
  })

  it.each([
    ['call.wav', 'upload.wav'],
    ['call.flac', 'upload.flac'],
    ['call.mp3', 'upload.mp3'],
    ['CALL.WAV', 'upload.wav'],
    ['Call.Mp3', 'upload.mp3'],
    ['rec..wav', 'upload.wav'],
  ])('%s is sent as %s', async (name, sent) => {
    await analysis.startAnalysis(new File(['audio'], name))
    expect((sentForm().get('file') as File).name).toBe(sent)
  })

  it('the real name is nowhere in the form', async () => {
    await analysis.startAnalysis(FILE)
    const values = [...sentForm().entries()].flat().map((value) => (value instanceof File ? value.name : value))
    expect(values.join(' ')).not.toContain('customer')
  })

  it("the bytes are the file's", async () => {
    await analysis.startAnalysis(FILE)
    expect(await (sentForm().get('file') as File).text()).toBe('audio bytes')
  })

  it('sets no Origin, Cookie or Content-Type header', async () => {
    await analysis.startAnalysis(FILE)
    const names = Object.keys(fetchMock.mock.calls[0]![1]!.headers as Record<string, string>).map((name) => name.toLowerCase())
    expect(names).toEqual(['accept'])
  })
})

describe('one POST per press', () => {
  it('two calls in one tick send one POST', async () => {
    fetchMock.mockImplementation(async () => json(201, BODY))
    await Promise.all([analysis.startAnalysis(FILE), analysis.startAnalysis(FILE)])
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('a call while submitting sends nothing', async () => {
    const answer = deferred()
    fetchMock.mockReturnValueOnce(answer.promise)
    const first = analysis.startAnalysis(FILE)
    await analysis.startAnalysis(FILE)
    expect(fetchMock).toHaveBeenCalledTimes(1)
    answer.resolve(json(201, BODY))
    await first
  })

  it('a call from a result sends nothing', async () => {
    fetchMock.mockResolvedValueOnce(json(201, BODY))
    await analysis.startAnalysis(FILE)
    await analysis.startAnalysis(FILE)
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(analysis.getSnapshot()).toEqual(toEqualResult())
  })

  it('Retry sends one more, two Retries one', async () => {
    fetchMock.mockResolvedValueOnce(apiError(500, 'internal_error'))
    await analysis.startAnalysis(FILE)
    fetchMock.mockResolvedValueOnce(json(201, BODY))
    await Promise.all([analysis.startAnalysis(FILE), analysis.startAnalysis(FILE)])
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(analysis.getSnapshot()).toEqual(toEqualResult())
  })
})

describe('the session and the draft', () => {
  async function held(): Promise<{ finish: (response: Response) => Promise<void> }> {
    const answer = deferred()
    fetchMock.mockReturnValueOnce(answer.promise)
    const running = analysis.startAnalysis(FILE)
    expect(analysis.getSnapshot()).toEqual({ status: 'submitting' })
    return {
      finish: async (response) => {
        answer.resolve(response)
        await running
      },
    }
  }

  it('sign-out during the POST drops the answer', async () => {
    const post = await held()
    session.markSignedOut()
    expect(analysis.getSnapshot()).toEqual({ status: 'idle' })
    await post.finish(json(201, BODY))
    expect(analysis.getSnapshot()).toEqual({ status: 'idle' })
  })

  it('another analyst after expiry drops the answer', async () => {
    const post = await held()
    session.expire()
    session.markSignedIn('analyst02')
    await post.finish(json(201, BODY))
    expect(analysis.getSnapshot()).toEqual({ status: 'idle' })
  })

  it('the same analyst after expiry keeps it', async () => {
    const post = await held()
    session.expire()
    session.markSignedIn('analyst01')
    await post.finish(json(201, BODY))
    expect(analysis.getSnapshot()).toEqual(toEqualResult())
  })

  it('a 401 after the session ended gives idle, not S5', async () => {
    const post = await held()
    session.expire()
    await post.finish(apiError(401, 'session_expired', null))
    expect(analysis.getSnapshot()).toEqual({ status: 'idle' })
  })

  it('a session error resets when the session ends', async () => {
    fetchMock.mockResolvedValueOnce(apiError(401, 'session_expired', null))
    await analysis.startAnalysis(FILE)
    expect(analysis.getSnapshot()).toEqual({ status: 'error', kind: 'session', requestId: null })
    session.expire()
    expect(analysis.getSnapshot()).toEqual({ status: 'idle' })
    expect(draft.getSnapshot()).toBe(FILE)
  })

  it('sign-out at a result clears both', async () => {
    fetchMock.mockResolvedValueOnce(json(201, BODY))
    await analysis.startAnalysis(FILE)
    session.markSignedOut()
    expect(draft.getSnapshot()).toBeNull()
    expect(analysis.getSnapshot()).toEqual({ status: 'idle' })
  })

  it('another analyst clears both', async () => {
    fetchMock.mockResolvedValueOnce(json(201, BODY))
    await analysis.startAnalysis(FILE)
    session.expire()
    session.markSignedIn('analyst02')
    expect(draft.getSnapshot()).toBeNull()
    expect(analysis.getSnapshot()).toEqual({ status: 'idle' })
  })

  it('expiry keeps both', async () => {
    fetchMock.mockResolvedValueOnce(json(201, BODY))
    await analysis.startAnalysis(FILE)
    session.expire()
    session.markSignedIn('analyst01')
    expect(draft.getSnapshot()).toBe(FILE)
    expect(analysis.getSnapshot()).toEqual(toEqualResult())
  })
})

describe('parseAnalysis', () => {
  it('accepts the §11 body and keeps its 8 fields', () => {
    expect(analysis.parseAnalysis({ ...BODY, extra: 'x' })).toEqual(BODY)
  })

  it.each(Object.keys(BODY))('refuses a body without %s', (field) => {
    const body: Record<string, unknown> = { ...BODY }
    delete body[field]
    expect(analysis.parseAnalysis(body)).toBeNull()
  })

  it.each([
    ['a prediction outside the three', { prediction: 'fake' }],
    ['a risk outside the three', { risk_level: 'low' }],
    ['a confidence under 0', { confidence: -0.1 }],
    ['a confidence over 1', { confidence: 1.1 }],
    ['a confidence that is text', { confidence: '0.9' }],
    ['a malformed request ID', { request_id: 'req_1' }],
    ['an empty notice', { notice: '' }],
  ])('refuses %s', (_name, change) => {
    expect(analysis.parseAnalysis({ ...BODY, ...change })).toBeNull()
  })
})

describe('every answer but a good 2xx is S5', () => {
  const error = (kind: string, requestId: string | null) => ({ status: 'error', kind, requestId })

  it.each([
    ['400 invalid_request', () => apiError(400, 'invalid_request'), 'server', ID],
    ['401 session_expired', () => apiError(401, 'session_expired', null), 'session', null],
    ['403 origin_not_allowed', () => apiError(403, 'origin_not_allowed', null), 'server', null],
    ['403 access_denied', () => apiError(403, 'access_denied', null), 'server', null],
    ['404 not_found', () => apiError(404, 'not_found', null), 'server', null],
    ['405 invalid_request', () => apiError(405, 'invalid_request', null), 'server', null],
    ['413 from the API', () => apiError(413, 'file_too_large'), 'size', ID],
    ['413 from nginx', () => apiError(413, 'file_too_large', null), 'size', null],
    ['415 unsupported_format', () => apiError(415, 'unsupported_format'), 'format', ID],
    ['422 corrupt_audio', () => apiError(422, 'corrupt_audio'), 'corrupt', ID],
    ['422 empty_file', () => apiError(422, 'empty_file'), 'empty', ID],
    ['422 recording_too_long', () => apiError(422, 'recording_too_long'), 'duration', ID],
    ['429 too_many_requests', () => apiError(429, 'too_many_requests', null), 'server', null],
    ['500 with an ID', () => apiError(500, 'internal_error'), 'server', ID],
    ['500 with no ID', () => apiError(500, 'internal_error', null), 'server', null],
    ['503 from the API', () => apiError(503, 'analysis_unavailable'), 'service', ID],
    ['503 from nginx', () => apiError(503, 'analysis_unavailable', null), 'service', null],
    ['503 sign_in_unavailable', () => apiError(503, 'sign_in_unavailable', null), 'server', null],
    ['504 analysis_timeout', () => apiError(504, 'analysis_timeout'), 'timeout', ID],
    ['an unknown code', () => apiError(500, 'something_new'), 'server', null],
    ['a 500 that is not JSON', () => new Response('<html>', { status: 500 }), 'server', null],
    ['a 201 that is not JSON', () => new Response('<html>', { status: 201 }), 'server', null],
    ['a 201 of the wrong shape', () => json(201, { ...BODY, confidence: 2 }), 'server', null],
    ['a 200 of {}', () => json(200, {}), 'server', null],
  ])('%s', async (_name, response, kind, requestId) => {
    fetchMock.mockResolvedValueOnce(response())
    await analysis.startAnalysis(FILE)
    expect(analysis.getSnapshot()).toEqual(error(kind, requestId))
  })

  it('no answer', async () => {
    fetchMock.mockRejectedValueOnce(new TypeError('network'))
    await analysis.startAnalysis(FILE)
    expect(analysis.getSnapshot()).toEqual(error('network', null))
  })

  it('the connection drops while the body is read', async () => {
    const response = json(201, BODY)
    vi.spyOn(response, 'json').mockRejectedValueOnce(new TypeError('dropped'))
    fetchMock.mockResolvedValueOnce(response)
    await analysis.startAnalysis(FILE)
    expect(analysis.getSnapshot()).toEqual(error('network', null))
  })
})
