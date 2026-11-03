import { act, cleanup, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { StatusResult, StatusState } from '../status/status'
import { formatCheckedAt } from '../status/checkedAt'
import Status from './Status'

// S10 rendered on a stand-in status store whose state the test sets (the RailStatus.test pattern). checkStatus is a
// spy; fetch is a spy that must never be called. Check now itself is in Status.check.test.tsx, on the real store.

const store = vi.hoisted(() => {
  let state: StatusState = { checking: true, result: null }
  const listeners = new Set<() => void>()
  return {
    get: (): StatusState => state,
    set(next: StatusState): void {
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

const checkStatus = vi.hoisted(() => vi.fn<() => Promise<void>>())

vi.mock('../status/status', () => ({
  getSnapshot: () => store.get(),
  subscribe: (listener: () => void) => store.subscribe(listener),
  checkStatus,
}))

const DETAILS = {
  versions: { model_version: 'mock', profile_version: 'preprocessing-v1', policy_version: 'risk-v1' },
  bundleManifestSha256: null,
  deferralOnFit: 0.0993,
}

const READY: StatusResult = {
  kind: 'ready',
  notice: false,
  failing: [],
  mock: true,
  checkedAt: '2026-11-03T09:14:05Z',
  timeSource: 'api',
  details: DETAILS,
}
const DEGRADED: StatusResult = { ...READY, notice: true, failing: ['history_and_stats'] }
const SERVICE_DOWN: StatusResult = { ...READY, kind: 'unavailable', source: 'service', failing: ['analysis'], mock: false }
const NO_ANSWER: StatusResult = { kind: 'unavailable', source: 'noAnswer', checkedAt: '2026-11-03T09:20:00.000Z', timeSource: 'browser' }

const fetchMock = vi.fn<typeof fetch>()

beforeEach(() => {
  checkStatus.mockReset()
  checkStatus.mockResolvedValue(undefined)
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
  store.set({ checking: true, result: null })
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  expect(fetchMock).not.toHaveBeenCalled()
})

function show(state: StatusState) {
  store.set(state)
  return render(<Status />)
}

const summary = () => screen.getByRole('region', { name: /^(Checking…|Ready for analysis|Unavailable)$/ })
const checks = () => screen.getByRole('region', { name: 'What is checked' })
const items = () =>
  within(checks())
    .getAllByRole('listitem')
    .map((item) => item.textContent)
const button = () => screen.getByRole('button', { name: 'Check now' })
const announcer = () => document.getElementById('status-announcer')!

describe('the head', () => {
  it('one h1 "System status", focusable for page changes, and the sub line', () => {
    show({ checking: false, result: READY })
    const h1 = screen.getByRole('heading', { level: 1 })
    expect(h1.textContent).toBe('System status')
    expect(h1.getAttribute('tabindex')).toBe('-1')
    expect(screen.getByText('Whether recordings can be analyzed right now.', { exact: false })).toBeTruthy()
  })

  it.each([
    ['ready, mock', READY, 1],
    ['unavailable from the service, not mock', SERVICE_DOWN, 0],
    ['no answer (no body)', NO_ANSWER, 0],
    ['checking, no result', null, 0],
  ] as const)('"Mock status values" tag: %s → %i', (_, result, count) => {
    show({ checking: result === null, result })
    expect(screen.queryAllByText('Mock status values')).toHaveLength(count)
  })
})

describe('each S10 state', () => {
  it('checking, the first check: Checking…, every item waiting, no time, Check now busy', () => {
    show({ checking: true, result: null })
    expect(summary().getAttribute('aria-busy')).toBe('true')
    expect(within(summary()).getByText('Contacting the service.')).toBeTruthy()
    expect(items()).toEqual([
      'Connection to the serviceWaiting for the service to answer.Checking',
      'Analysis availabilityWaiting for the service to answer.Checking',
      'History and statistics availabilityWaiting for the service to answer.Checking',
    ])
    expect(screen.queryByText(/^Last (checked|attempt)/)).toBeNull()
    expect(button().hasAttribute('disabled')).toBe(true)
    expect(button().getAttribute('aria-busy')).toBe('true')
    expect(screen.queryByText('Technical details')).toBeNull()
  })

  it('ready: summary, three items, Last checked, details', () => {
    show({ checking: false, result: READY })
    expect(summary().className).toContain('sum-ready')
    expect(summary().hasAttribute('aria-busy')).toBe(false)
    expect(within(summary()).getByRole('heading', { level: 2 }).textContent).toBe('Ready for analysis')
    expect(within(summary()).getByText('Recordings can be analyzed. History and statistics are available.')).toBeTruthy()
    expect(items()).toEqual([
      'Connection to the serviceThis console can reach the service.Connected',
      'Analysis availabilityRecordings can be analyzed.Available',
      'History and statistics availabilityStored results and Overview figures can be loaded.Available',
    ])
    expect(screen.getByText(`Last checked ${formatCheckedAt(READY.checkedAt)}`)).toBeTruthy()
    expect(button().hasAttribute('disabled')).toBe(false)
    expect(button().hasAttribute('aria-busy')).toBe(false)
  })

  it('ready with a notice: Ready for analysis, the notice names the failing check, no role, the decision line', () => {
    show({ checking: false, result: DEGRADED })
    expect(summary().className).toContain('sum-degraded')
    expect(within(summary()).getByRole('heading', { level: 2 }).textContent).toBe('Ready for analysis')
    const notice = within(summary()).getByText('History and statistics is not fully available.').closest('.banner-warn')!
    expect(notice).toBeTruthy()
    expect(notice.hasAttribute('role')).toBe(false)
    expect(notice.querySelector('svg')?.getAttribute('aria-hidden')).toBe('true')
    expect(within(summary()).getByText('Open product decision:')).toBeTruthy()
    expect(items()[2]).toBe(
      'History and statistics availabilityStored results and Overview figures cannot be loaded right now. Analysis is not affected.Unavailable',
    )
    expect(screen.getByText(/^Last checked /)).toBeTruthy()
  })

  it('unavailable from the service: names the failing check, Connected, Last checked', () => {
    show({ checking: false, result: SERVICE_DOWN })
    expect(summary().className).toContain('sum-down')
    expect(within(summary()).getByRole('heading', { level: 2 }).textContent).toBe('Unavailable')
    expect(
      within(summary()).getByText('The service responds, but it cannot analyze recordings right now. Requests fail with no verdict.'),
    ).toBeTruthy()
    expect(items().map((text) => text?.replace(/^.*\.(?=[A-Z][a-z]+$)/, ''))).toEqual(['Connected', 'Unavailable', 'Available'])
    expect(screen.getByText(/^Last checked /)).toBeTruthy()
    expect(screen.queryByText(/^Last attempt/)).toBeNull()
  })

  it('no answer: Unreachable, Last attempt (never Last checked), no Technical details', () => {
    show({ checking: false, result: NO_ANSWER })
    expect(within(summary()).getByRole('heading', { level: 2 }).textContent).toBe('Unavailable')
    expect(items()).toEqual([
      'Connection to the serviceThis console cannot reach the analysis service.Unreachable',
      'Analysis availabilityCannot be checked while the service is unreachable.Unavailable',
      'History and statistics availabilityStored results cannot be loaded while the service is unreachable.Unavailable',
    ])
    expect(screen.getByText(`Last attempt ${formatCheckedAt(NO_ANSWER.checkedAt)}`)).toBeTruthy()
    expect(screen.queryByText(/^Last checked/)).toBeNull()
    expect(screen.queryByText('Technical details')).toBeNull()
  })

  it('a re-check shows Checking… and keeps the previous time label', () => {
    show({ checking: true, result: NO_ANSWER })
    expect(within(summary()).getByRole('heading', { level: 2 }).textContent).toBe('Checking…')
    expect(within(summary()).getByText('Contacting the service. The previous status stays valid until the check finishes.')).toBeTruthy()
    expect(screen.getByText(/^Last attempt /)).toBeTruthy()
    expect(button().textContent).toBe('Check now')
  })

  it('states never by colour alone: each pill has text, every icon is hidden', () => {
    for (const result of [READY, DEGRADED, SERVICE_DOWN, NO_ANSWER]) {
      const { container, unmount } = show({ checking: false, result })
      for (const pill of container.querySelectorAll('.state-pill')) expect(pill.textContent?.trim()).not.toBe('')
      for (const svg of container.querySelectorAll('svg')) expect(svg.getAttribute('aria-hidden')).toBe('true')
      for (const spin of container.querySelectorAll('.spin')) expect(spin.getAttribute('aria-hidden')).toBe('true')
      unmount()
    }
  })

  it('no summary or list has a live role; only the S10 region is live', () => {
    const { container } = show({ checking: false, result: DEGRADED })
    expect(container.querySelectorAll('[role="status"], [role="alert"]')).toHaveLength(0)
    expect([...container.querySelectorAll('[aria-live]')].map((node) => node.id)).toEqual(['status-announcer'])
    expect(announcer().getAttribute('aria-live')).toBe('polite')
  })
})

describe('Technical details', () => {
  it('a closed native disclosure with the body values', () => {
    show({ checking: false, result: READY })
    const details = screen.getByText('Technical details').closest('details')!
    expect(details.open).toBe(false)
    const values = [...details.querySelectorAll('dt')].map((dt) => [dt.textContent, dt.nextElementSibling?.textContent])
    expect(values).toEqual([
      ['Model', 'mock'],
      ['Preprocessing profile', 'preprocessing-v1'],
      ['Risk policy', 'risk-v1'],
      ['Validation “Cannot decide” share', 'deferral_on_fit = 0.0993'],
      ['Manifest SHA-256', 'Not available until the bundle is frozen.'],
    ])
  })

  it('shows the manifest hash when the API gives one', () => {
    const sha = 'a'.repeat(64)
    show({ checking: false, result: { ...READY, details: { ...DETAILS, bundleManifestSha256: sha } } })
    expect(screen.getByText(sha)).toBeTruthy()
  })
})

describe('no check and no announcement without Check now', () => {
  it('mounting starts no check and sends no request', () => {
    show({ checking: false, result: READY })
    expect(checkStatus).not.toHaveBeenCalled()
    expect(announcer().childNodes).toHaveLength(0)
  })

  it('a store change from elsewhere (the shell’s check) re-renders without announcing', () => {
    show({ checking: true, result: null })
    act(() => store.set({ checking: false, result: READY }))
    act(() => store.set({ checking: false, result: { ...NO_ANSWER } }))
    expect(screen.getByText(/^Last attempt /)).toBeTruthy()
    expect(announcer().childNodes).toHaveLength(0)
  })
})
