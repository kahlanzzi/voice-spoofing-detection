import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { createMemoryRouter } from 'react-router'
import { RouterProvider } from 'react-router/dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { routes } from '../routing/router'
import type { SessionState } from '../session/session'
import * as session from '../session/session'

// S1 through the real router and guard, against a stand-in session store: markSignedIn signs in (so the guard
// navigates), checkSession is a spy each test drives. fetch is stubbed; nothing reaches a network.

const store = vi.hoisted(() => {
  let state: SessionState = { status: 'signedOut', reason: 'none' }
  const listeners = new Set<() => void>()
  return {
    get: (): SessionState => state,
    set(next: SessionState): void {
      state = next
      for (const listener of [...listeners]) listener()
    },
    subscribe(listener: () => void): () => void {
      listeners.add(listener)
      return () => {
        listeners.delete(listener)
      }
    },
  }
})

vi.mock('../session/session', () => ({
  getSnapshot: () => store.get(),
  subscribe: (listener: () => void) => store.subscribe(listener),
  checkSession: vi.fn(() => Promise.resolve()),
  expire: vi.fn(),
  markSignedIn: vi.fn((username: string) => store.set({ status: 'signedIn', username })),
  markSignedOut: vi.fn(),
  startSession: vi.fn(),
}))

const SIGNED_OUT: SessionState = { status: 'signedOut', reason: 'none' }
const EXPIRED: SessionState = { status: 'signedOut', reason: 'expired' }
const UNAVAILABLE: SessionState = { status: 'unavailable' }

const PASSWORD = '  correct horse battery staple  '
const TITLE = 'Sign in to the analyst console'
const COPY = {
  invalid: 'The analyst ID or password is not correct. Check both and try again.',
  denied: 'This account does not have access to the analyst console. Ask your administrator for access.',
  locked: 'This account is temporarily locked. Try again later, or ask your administrator.',
  unavailable: 'Sign-in is temporarily unavailable. Try again in a few minutes.',
}
const NGINX_429 =
  '{"error":{"code":"too_many_requests","message":"Too many sign-in attempts from this address; try again later.","request_id":null}}\n'
const NGINX_503 = '{"error":{"code":"analysis_unavailable","message":"The analysis service is unavailable.","request_id":null}}\n'

let fetchMock: ReturnType<typeof vi.fn<typeof fetch>>

function apiError(status: number, code: string): Response {
  return new Response(JSON.stringify({ error: { code, message: 'Server text.', request_id: null } }), { status })
}

function signedIn(username = 'analyst01'): Response {
  return new Response(
    JSON.stringify({ username, signed_in_at: '2026-11-03T09:00:00Z', expires_at: '2026-11-03T21:00:00Z' }),
    { status: 200 },
  )
}

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

function renderAt(entry = '/signin', state: SessionState = SIGNED_OUT) {
  store.set(state)
  const router = createMemoryRouter(routes, { initialEntries: [entry] })
  render(<RouterProvider router={router} />)
  return { router, user: userEvent.setup() }
}

const idField = () => screen.getByLabelText('Analyst ID') as HTMLInputElement
const passwordField = () => screen.getByLabelText('Password') as HTMLInputElement
const submitButton = () => screen.getByRole('button', { name: /^(Sign in|Signing in…)$/ }) as HTMLButtonElement
const alertText = () => screen.queryByRole('alert')?.textContent ?? null

async function fillAndSubmit(user: ReturnType<typeof userEvent.setup>, id = 'analyst01', password = PASSWORD) {
  await user.type(idField(), id)
  await user.type(passwordField(), password)
  await user.click(submitButton())
}

beforeEach(() => {
  fetchMock = vi.fn<typeof fetch>()
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  vi.clearAllMocks()
})

describe('default', () => {
  it('the form: labels, names, autocomplete, no alert, no Retry, the recovery line', () => {
    renderAt()
    expect(screen.getByRole('heading', { level: 1, name: TITLE }).getAttribute('tabindex')).toBe('-1')
    expect(idField()).toMatchObject({ name: 'username', type: 'text', autocomplete: 'username' })
    expect(passwordField()).toMatchObject({ name: 'password', type: 'password', autocomplete: 'current-password' })
    expect(document.querySelector('[autocomplete="off"]')).toBeNull()
    expect(document.querySelector('form')?.getAttribute('method')).toBe('post')
    expect(alertText()).toBeNull()
    expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()
    expect(screen.getByText('Forgot your password? Ask your administrator.')).toBeTruthy()
    expect(screen.queryByText(/Open product decision|Enter demo/)).toBeNull()
    expect(idField().getAttribute('aria-invalid')).toBe('false')
  })

  it('the keyboard order: Analyst ID, password, Show, Sign in', async () => {
    const { user } = renderAt()
    await user.tab()
    expect(document.activeElement).toBe(idField())
    await user.tab()
    expect(document.activeElement).toBe(passwordField())
    await user.tab()
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Show password' }))
    await user.tab()
    expect(document.activeElement).toBe(submitButton())
  })

  it('Show / Hide switches the password field and its pressed state', async () => {
    const { user } = renderAt()
    await user.click(screen.getByRole('button', { name: 'Show password' }))
    expect(passwordField().type).toBe('text')
    const hide = screen.getByRole('button', { name: 'Hide password' })
    expect(hide.getAttribute('aria-pressed')).toBe('true')
    expect(hide.getAttribute('aria-controls')).toBe(passwordField().id)
  })
})

describe('field errors (nothing is sent)', () => {
  it('both empty: both errors, linked, first field focused', async () => {
    const { user } = renderAt()
    await user.click(submitButton())
    expect(fetchMock).not.toHaveBeenCalled()
    for (const [field, text] of [
      [idField(), 'Enter your analyst ID.'],
      [passwordField(), 'Enter your password.'],
    ] as const) {
      expect(field.getAttribute('aria-invalid')).toBe('true')
      expect(document.getElementById(field.getAttribute('aria-describedby')!)?.textContent).toBe(text)
    }
    expect(document.activeElement).toBe(idField())
  })

  it('spaces only in Analyst ID: its error only, and the password is kept', async () => {
    const { user } = renderAt()
    await fillAndSubmit(user, '   ')
    expect(fetchMock).not.toHaveBeenCalled()
    expect(screen.getByText('Enter your analyst ID.')).toBeTruthy()
    expect(screen.queryByText('Enter your password.')).toBeNull()
    expect(passwordField().value).toBe(PASSWORD)
    expect(document.activeElement).toBe(idField())
  })

  it('password empty: focus on the password', async () => {
    const { user } = renderAt()
    await user.type(idField(), 'analyst01')
    await user.keyboard('{Enter}')
    expect(fetchMock).not.toHaveBeenCalled()
    expect(document.activeElement).toBe(passwordField())
  })
})

describe('submit', () => {
  it('loading: "Signing in…", disabled, aria-busy, announced', async () => {
    const pending = deferred<Response>()
    fetchMock.mockReturnValue(pending.promise)
    const { user } = renderAt()
    await fillAndSubmit(user)
    expect(submitButton().textContent).toBe('Signing in…')
    expect(submitButton().disabled).toBe(true)
    expect(submitButton().getAttribute('aria-busy')).toBe('true')
    expect(screen.getAllByRole('status').map((node) => node.textContent)).toContain('Signing in…')
    await act(async () => pending.resolve(apiError(401, 'invalid_credentials')))
    expect(submitButton().textContent).toBe('Sign in')
    expect(submitButton().getAttribute('aria-busy')).toBeNull()
  })

  it('the body is exactly username (trimmed) and password (untouched); markSignedIn gets the server username', async () => {
    fetchMock.mockResolvedValue(signedIn('analyst01'))
    const { user } = renderAt()
    await fillAndSubmit(user, ' Analyst01 ')
    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [path, init] = fetchMock.mock.calls[0]!
    expect(path).toBe('/api/session')
    expect(init?.method).toBe('POST')
    expect(JSON.parse(init?.body as string)).toStrictEqual({ username: 'Analyst01', password: PASSWORD })
    expect(session.markSignedIn).toHaveBeenCalledWith('analyst01')
  })

  it('a 200 without a username is not a sign-in', async () => {
    fetchMock.mockResolvedValue(new Response('{}', { status: 200 }))
    const { user } = renderAt()
    await fillAndSubmit(user)
    expect(session.markSignedIn).not.toHaveBeenCalled()
    expect(alertText()).toBe(COPY.unavailable)
  })
})

describe('failed answers', () => {
  it.for([
    ['401 invalid_credentials', () => apiError(401, 'invalid_credentials'), 'invalid'],
    ['400 invalid_request', () => apiError(400, 'invalid_request'), 'invalid'],
    ['403 access_denied', () => apiError(403, 'access_denied'), 'denied'],
    ['429 account_locked', () => apiError(429, 'account_locked'), 'locked'],
    ['503 sign_in_unavailable', () => apiError(503, 'sign_in_unavailable'), 'unavailable'],
    ['nginx 429', () => new Response(NGINX_429, { status: 429 }), 'unavailable'],
    ['nginx 503', () => new Response(NGINX_503, { status: 503 }), 'unavailable'],
    ['500 internal_error', () => apiError(500, 'internal_error'), 'unavailable'],
    ['403 origin_not_allowed', () => apiError(403, 'origin_not_allowed'), 'unavailable'],
  ] as const)('%s: alert, password cleared, Analyst ID kept, focus', async ([, answer, variant]) => {
    fetchMock.mockResolvedValue(answer())
    const { user } = renderAt()
    await user.click(screen.getByRole('button', { name: 'Show password' }))
    await fillAndSubmit(user)
    expect(alertText()).toBe(COPY[variant])
    expect(passwordField().value).toBe('')
    expect(passwordField().type).toBe('password')
    expect(idField().value).toBe('analyst01')
    expect(session.markSignedIn).not.toHaveBeenCalled()
    const focused = variant === 'invalid' ? idField() : screen.getByRole('heading', { level: 1 })
    await waitFor(() => expect(document.activeElement).toBe(focused))
    const invalid = String(variant === 'invalid')
    expect(idField().getAttribute('aria-invalid')).toBe(invalid)
    expect(passwordField().getAttribute('aria-invalid')).toBe(invalid)
    if (variant === 'invalid') expect(idField().getAttribute('aria-describedby')).toBe(screen.getByRole('alert').id)
  })

  it('a network error → unavailable', async () => {
    fetchMock.mockRejectedValue(new TypeError('Failed to fetch'))
    const { user } = renderAt()
    await fillAndSubmit(user)
    expect(alertText()).toBe(COPY.unavailable)
  })

  it('the same failure twice is announced twice (a new alert node)', async () => {
    fetchMock.mockImplementation(() => Promise.resolve(apiError(401, 'invalid_credentials')))
    const { user } = renderAt()
    await fillAndSubmit(user)
    const first = screen.getByRole('alert')
    await user.type(passwordField(), 'again')
    await user.click(submitButton())
    expect(screen.getByRole('alert')).not.toBe(first)
  })
})

describe('expired session', () => {
  it('shows the notice, which goes after a failed submit', async () => {
    fetchMock.mockResolvedValue(apiError(401, 'invalid_credentials'))
    const { user } = renderAt('/signin', EXPIRED)
    const notice = 'Your session expired. Sign in again to continue where you left off.'
    expect(screen.getByText(notice).closest('[role="status"]')).toBeTruthy()
    await fillAndSubmit(user)
    expect(screen.queryByText(notice)).toBeNull()
    expect(alertText()).toBe(COPY.invalid)
  })
})

describe('sign-in service unavailable (the check on load failed)', () => {
  it('the banner and Retry; the form stays usable', () => {
    renderAt('/signin', UNAVAILABLE)
    expect(alertText()).toBe(COPY.unavailable)
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
    expect(idField().disabled).toBe(false)
  })

  it('Retry calls checkSession once, shows "Checking…", blocks the form; signed out → the form, ID kept and focused', async () => {
    const pending = deferred<void>()
    vi.mocked(session.checkSession).mockReturnValue(pending.promise)
    const { user } = renderAt('/signin', UNAVAILABLE)
    await user.type(idField(), 'analyst01')
    await user.click(screen.getByRole('button', { name: 'Retry' }))
    const retry = screen.getByRole('button', { name: 'Checking…' }) as HTMLButtonElement
    expect(retry.disabled).toBe(true)
    expect(retry.getAttribute('aria-busy')).toBe('true')
    expect(submitButton().disabled).toBe(true)
    fireEvent.submit(document.querySelector('form')!)
    expect(fetchMock).not.toHaveBeenCalled()
    await act(async () => {
      store.set(SIGNED_OUT)
      pending.resolve()
    })
    expect(session.checkSession).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole('button', { name: /Retry|Checking/ })).toBeNull()
    expect(alertText()).toBeNull()
    expect(idField().value).toBe('analyst01')
    await waitFor(() => expect(document.activeElement).toBe(idField()))
  })

  it('still unavailable: Retry again, focused, the alert announced again', async () => {
    const { user } = renderAt('/signin', UNAVAILABLE)
    const first = screen.getByRole('alert')
    await user.click(screen.getByRole('button', { name: 'Retry' }))
    const retry = await screen.findByRole('button', { name: 'Retry' })
    await waitFor(() => expect(document.activeElement).toBe(retry))
    expect(screen.getByRole('alert')).not.toBe(first)
  })

  it('a failed submit replaces the banner and hides Retry', async () => {
    fetchMock.mockResolvedValue(apiError(401, 'invalid_credentials'))
    const { user } = renderAt('/signin', UNAVAILABLE)
    await fillAndSubmit(user)
    expect(alertText()).toBe(COPY.invalid)
    expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()
  })
})

describe('one request at a time', () => {
  it('double click sends one request', async () => {
    fetchMock.mockReturnValue(deferred<Response>().promise)
    const { user } = renderAt()
    await user.type(idField(), 'analyst01')
    await user.type(passwordField(), PASSWORD)
    await user.dblClick(submitButton())
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('two submit events in a row send one request', async () => {
    fetchMock.mockReturnValue(deferred<Response>().promise)
    const { user } = renderAt()
    await user.type(idField(), 'analyst01')
    await user.type(passwordField(), PASSWORD)
    const form = document.querySelector('form')!
    act(() => {
      fireEvent.submit(form)
      fireEvent.submit(form)
    })
    await user.keyboard('{Enter}')
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })
})

describe('landing after sign-in (the guard, not S1, reads next)', () => {
  it.each([
    ['a valid next', '/signin?next=%2Fhistory%3Fverdict%3Dsynthetic', '/history', '?verdict=synthetic', 'History'],
    ['no next', '/signin', '/analyze', '', 'Analyze a recording'],
    ['an absolute next', '/signin?next=https%3A%2F%2Fevil.test%2Fanalyze', '/analyze', '', 'Analyze a recording'],
    ['next to sign-in', '/signin?next=%2Fsignin', '/analyze', '', 'Analyze a recording'],
  ])('%s', async (_label, entry, pathname, search, page) => {
    fetchMock.mockResolvedValue(signedIn())
    const { router, user } = renderAt(entry)
    await fillAndSubmit(user)
    expect(await screen.findByRole('heading', { level: 1, name: page })).toBeTruthy()
    expect(router.state.location.pathname).toBe(pathname)
    expect(router.state.location.search).toBe(search)
  })
})

describe('the password stays in the form', () => {
  it('never reaches the console, storage, cookies, history, the URL or the page text', async () => {
    const spies = [
      ...(['log', 'info', 'warn', 'error', 'debug'] as const).map((name) => vi.spyOn(console, name)),
      vi.spyOn(Storage.prototype, 'setItem'),
      vi.spyOn(window.history, 'pushState'),
      vi.spyOn(window.history, 'replaceState'),
    ]
    const cookie = vi.spyOn(Document.prototype, 'cookie', 'set')
    fetchMock.mockResolvedValueOnce(apiError(401, 'invalid_credentials')).mockResolvedValueOnce(signedIn())
    const { router, user } = renderAt('/signin?next=%2Fhistory')
    await fillAndSubmit(user)
    expect(document.body.textContent).not.toContain(PASSWORD.trim())
    await user.type(passwordField(), PASSWORD)
    await user.click(submitButton())
    await screen.findByRole('heading', { level: 1, name: 'History' })
    for (const spy of spies) expect(JSON.stringify(spy.mock.calls)).not.toContain(PASSWORD.trim())
    expect(cookie).not.toHaveBeenCalled()
    expect(JSON.stringify(router.state.location)).not.toContain(PASSWORD.trim())
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })
})
