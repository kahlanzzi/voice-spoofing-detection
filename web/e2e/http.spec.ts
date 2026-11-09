// HTTP checks through nginx with Playwright's request context: no browser, no cookie, no Origin unless given.
// docs/api_design.md §4.8 CSRF (Origin on the routes that change data, checked before the session) and §6;
// deploy/nginx.conf (the JSON 404) and deploy/nginx_security_headers.conf (A13). Two sign-in POSTs: POST
// /api/session without Origin and with a foreign one (nginx counts every POST to that location; no "null" case here,
// to stay inside the burst: the other three routes cover it).
import { expect, test, type APIResponse } from '@playwright/test'
import { securityHeaders } from './support/nginx-headers.ts'
import { BASE_URL } from './support/stack.ts'

const FOREIGN = 'http://evil.example'
const REQUEST_ID = `req_${'0123456789abcdef'.repeat(2)}`

const ORIGINS = { none: undefined, foreign: FOREIGN, null: 'null' } as const

// The routes that change data (api/session_guard.py needs_origin), but sign-in, which is limited by nginx.
const SESSION_ROUTES = [
  { method: 'DELETE', path: '/api/session' },
  { method: 'POST', path: '/api/analyses' },
  { method: 'PUT', path: `/api/analyses/${REQUEST_ID}/note` },
] as const

const DOCUMENTS = ['/signin', '/analyze', '/history', `/history/${REQUEST_ID}`, '/overview', '/status']

async function expectError(response: APIResponse, status: number, code: string) {
  expect(response.status()).toBe(status)
  expect(response.headers()['content-type']).toMatch(/^application\/json/)
  expect(await response.json()).toEqual({ error: expect.objectContaining({ code, request_id: null }) })
}

const headersFor = (origin: string | undefined): Record<string, string> => (origin === undefined ? {} : { origin })

for (const [label, origin] of Object.entries({ none: ORIGINS.none, foreign: ORIGINS.foreign })) {
  test(`POST /api/session, Origin ${label} → 403 origin_not_allowed`, async ({ request }) => {
    await expectError(await request.post('/api/session', { headers: headersFor(origin) }), 403, 'origin_not_allowed')
  })
}

for (const { method, path } of SESSION_ROUTES) {
  for (const [label, origin] of Object.entries(ORIGINS)) {
    test(`${method} ${path}, Origin ${label} → 403 origin_not_allowed`, async ({ request }) => {
      await expectError(await request.fetch(path, { method, headers: headersFor(origin) }), 403, 'origin_not_allowed')
    })
  }

  // The control: the allowed Origin passes the Origin check and stops at the session check, so the 403s above come
  // from the Origin check.
  test(`${method} ${path}, allowed Origin, no session → 401 session_expired`, async ({ request }) => {
    await expectError(await request.fetch(path, { method, headers: { origin: BASE_URL } }), 401, 'session_expired')
  })
}

test('the security headers on every app route equal deploy/nginx_security_headers.conf', async ({ request }) => {
  const expected = securityHeaders()
  for (const path of DOCUMENTS) {
    const response = await request.get(path)
    expect(response.status(), path).toBe(200)
    expect(response.headers()['content-type'], path).toMatch(/^text\/html/)
    expect(response.headers(), path).toMatchObject(expected)
  }
})

for (const path of ['/nope', '/index.html', '/docs']) {
  test(`${path} → 404 application/json not_found, with the security headers`, async ({ request }) => {
    const response = await request.get(path)
    await expectError(response, 404, 'not_found')
    expect(response.headers()).toMatchObject(securityHeaders())
  })
}
