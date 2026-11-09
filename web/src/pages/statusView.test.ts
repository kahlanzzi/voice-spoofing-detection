import { describe, expect, it } from 'vitest'
import type { StatusDetails, StatusResult } from '../status/status'
import { noticeText, statusView, timeLabel } from './statusView'

// S10's copy for each status store state (Stage 18 PR 7b; prototype lines 1180-1201, team decision 2026-11-01).

const DETAILS: StatusDetails = {
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
const SERVICE_DOWN: StatusResult = { ...READY, kind: 'unavailable', source: 'service', failing: ['analysis'] }
const NO_ANSWER: StatusResult = { kind: 'unavailable', source: 'noAnswer', checkedAt: '2026-11-03T09:20:00.000Z', timeSource: 'browser' }

const pills = (result: StatusResult) => statusView({ checking: false, result }).items.map(({ label }) => label)

describe('checking', () => {
  it('the first check: "Checking…", no previous status to keep, every item waiting, nothing to announce', () => {
    const view = statusView({ checking: true, result: null })
    expect(view.summary).toEqual({ kind: 'loading', title: 'Checking…', text: 'Contacting the service.' })
    expect(view.items).toEqual(
      ['Connection to the service', 'Analysis availability', 'History and statistics availability'].map((title) => ({
        title,
        pill: 'wait',
        label: 'Checking',
        text: 'Waiting for the service to answer.',
      })),
    )
    expect(view.notice).toBeNull()
    expect(view.announcement).toBeNull()
  })

  it('a re-check keeps the previous status valid (prototype line 1190)', () => {
    for (const result of [READY, DEGRADED, SERVICE_DOWN, NO_ANSWER]) {
      expect(statusView({ checking: true, result }).summary).toEqual({
        kind: 'loading',
        title: 'Checking…',
        text: 'Contacting the service. The previous status stays valid until the check finishes.',
      })
    }
  })

  it('no result and not checking (a 401 answer) still reads as Checking', () => {
    expect(statusView({ checking: false, result: null }).summary.kind).toBe('loading')
  })
})

describe('ready', () => {
  it('summary, three available items and the announcement', () => {
    const view = statusView({ checking: false, result: READY })
    expect(view.summary).toEqual({ kind: 'ready', title: 'Ready for analysis', text: 'Recordings can be analyzed. History and statistics are available.' })
    expect(view.notice).toBeNull()
    expect(view.items).toEqual([
      { title: 'Connection to the service', pill: 'ok', label: 'Connected', text: 'This console can reach the service.' },
      { title: 'Analysis availability', pill: 'ok', label: 'Available', text: 'Recordings can be analyzed.' },
      { title: 'History and statistics availability', pill: 'ok', label: 'Available', text: 'Stored results and Overview figures can be loaded.' },
    ])
    expect(view.announcement).toBe('Status checked: Ready for analysis.')
  })
})

describe('ready with a notice (degraded)', () => {
  it('"Ready for analysis", the warn summary and a notice naming the failing part', () => {
    const view = statusView({ checking: false, result: DEGRADED })
    expect(view.summary).toEqual({ kind: 'degraded', title: 'Ready for analysis', text: 'Recordings can be analyzed.' })
    expect(view.notice).toBe('History and statistics is not fully available.')
    expect(pills(DEGRADED)).toEqual(['Connected', 'Available', 'Unavailable'])
    expect(view.items[2]).toMatchObject({ pill: 'err', text: 'Stored results and Overview figures cannot be loaded right now. Analysis is not affected.' })
    expect(view.announcement).toBe('Status checked: Ready for analysis, with a notice.')
  })

  it('the notice names each failing part, or gives the fallback', () => {
    expect(noticeText(['analysis'])).toBe('Analysis is not fully available.')
    expect(noticeText(['analysis', 'history_and_stats'])).toBe('Analysis and History and statistics are not fully available.')
    expect(noticeText([])).toBe('Some parts of the service are not fully available.')
    expect(statusView({ checking: false, result: { ...DEGRADED, failing: [] } }).notice).toBe('Some parts of the service are not fully available.')
  })
})

describe('unavailable from the service', () => {
  it.each([
    [['analysis'], 'The service responds, but it cannot analyze recordings right now. Requests fail with no verdict.', ['Connected', 'Unavailable', 'Available']],
    [['analysis', 'history_and_stats'], 'The service responds, but it cannot analyze recordings right now. Requests fail with no verdict.', ['Connected', 'Unavailable', 'Unavailable']],
    [['history_and_stats'], 'The service responds, but History and statistics is unavailable.', ['Connected', 'Available', 'Unavailable']],
    [[], 'The service responds, but it is not available right now.', ['Connected', 'Available', 'Available']],
  ] as const)('failing %j → the text names it', (failing, text, labels) => {
    const result: StatusResult = { ...SERVICE_DOWN, failing }
    const view = statusView({ checking: false, result })
    expect(view.summary).toEqual({ kind: 'down', title: 'Unavailable', text })
    expect(view.notice).toBeNull()
    expect(pills(result)).toEqual(labels)
    expect(view.announcement).toBe('Status checked: Unavailable.')
  })
})

describe('unavailable, no answer', () => {
  it('Unreachable, both parts unavailable, the prototype line 691 text', () => {
    const view = statusView({ checking: false, result: NO_ANSWER })
    expect(view.summary).toEqual({
      kind: 'down',
      title: 'Unavailable',
      text: 'The console cannot reach the analysis service. No recording can be analyzed until the connection is back. Stored results cannot be loaded either.',
    })
    expect(view.items).toEqual([
      { title: 'Connection to the service', pill: 'err', label: 'Unreachable', text: 'This console cannot reach the analysis service.' },
      { title: 'Analysis availability', pill: 'err', label: 'Unavailable', text: 'Cannot be checked while the service is unreachable.' },
      { title: 'History and statistics availability', pill: 'err', label: 'Unavailable', text: 'Stored results cannot be loaded while the service is unreachable.' },
    ])
    expect(view.announcement).toBe('Status checked: Unavailable.')
  })
})

describe('timeLabel', () => {
  it('"Last checked" for the API time, unavailable from the service included; "Last attempt" for the browser clock', () => {
    expect(timeLabel(READY)).toBe('Last checked')
    expect(timeLabel(DEGRADED)).toBe('Last checked')
    expect(timeLabel(SERVICE_DOWN)).toBe('Last checked')
    expect(timeLabel(NO_ANSWER)).toBe('Last attempt')
    expect(timeLabel(null)).toBeNull()
  })
})
