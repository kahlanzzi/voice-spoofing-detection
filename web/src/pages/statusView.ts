// What S10 System status shows for the status store's state (Stage 18 PR 7b; HANDOFF §1 S10, prototype lines
// 1180-1201). The copy is the prototype's, except where the team decisions change it: degraded is "Ready for
// analysis" with a notice naming what fails (2026-11-01), and the browser derives nothing, so each item follows the
// body's checks. The texts marked "proposal" are open for review (project log, Open items): they name the failing
// checks, give the degraded fallback, and say "Unreachable" for a check that got no answer.
import type { StatusPart, StatusState } from '../status/status'

export type SummaryKind = 'loading' | 'ready' | 'degraded' | 'down'
export type PillKind = 'ok' | 'err' | 'wait'

export interface StatusItem {
  readonly title: string
  readonly pill: PillKind
  readonly label: string
  readonly text: string
}

export interface StatusView {
  readonly summary: { readonly kind: SummaryKind; readonly title: string; readonly text: string }
  readonly notice: string | null
  readonly items: readonly StatusItem[]
  readonly announcement: string | null // after Check now: "Status checked: <label>."
}

const PART_NAMES: Record<StatusPart, string> = { analysis: 'Analysis', history_and_stats: 'History and statistics' }

const CONNECTION = 'Connection to the service'
const ANALYSIS = 'Analysis availability'
const HISTORY = 'History and statistics availability'

const WAITING = 'Waiting for the service to answer.'

// Prototype lines 1197-1198; the unavailable History text is a proposal (the checks are available or unavailable,
// so the prototype's "Partly unavailable" has no source).
const PART_ITEMS: Record<StatusPart, { title: string; available: string; unavailable: string }> = {
  analysis: {
    title: ANALYSIS,
    available: 'Recordings can be analyzed.',
    unavailable: 'The service cannot analyze recordings right now. Requests fail with no verdict.',
  },
  history_and_stats: {
    title: HISTORY,
    available: 'Stored results and Overview figures can be loaded.',
    unavailable: 'Stored results and Overview figures cannot be loaded right now. Analysis is not affected.',
  },
}

function names(parts: readonly StatusPart[]): string {
  return parts.map((part) => PART_NAMES[part]).join(' and ')
}

// The degraded notice (proposal; the fallback when the API names no failing check).
export function noticeText(failing: readonly StatusPart[]): string {
  if (failing.length === 0) return 'Some parts of the service are not fully available.'
  return `${names(failing)} ${failing.length > 1 ? 'are' : 'is'} not fully available.`
}

// Unavailable from the service (proposal): prototype line 692 when analysis fails.
function serviceDownText(failing: readonly StatusPart[]): string {
  if (failing.includes('analysis')) return 'The service responds, but it cannot analyze recordings right now. Requests fail with no verdict.'
  if (failing.length > 0) return `The service responds, but ${names(failing)} ${failing.length > 1 ? 'are' : 'is'} unavailable.`
  return 'The service responds, but it is not available right now.'
}

function partItems(failing: readonly StatusPart[]): StatusItem[] {
  return (['analysis', 'history_and_stats'] as const).map((part) => {
    const item = PART_ITEMS[part]
    const down = failing.includes(part)
    return { title: item.title, pill: down ? 'err' : 'ok', label: down ? 'Unavailable' : 'Available', text: down ? item.unavailable : item.available }
  })
}

const CONNECTED: StatusItem = { title: CONNECTION, pill: 'ok', label: 'Connected', text: 'This console can reach the service.' }

export function statusView({ checking, result }: StatusState): StatusView {
  if (checking || result === null) {
    // Prototype line 1190; its second sentence only when there is a previous status to keep.
    const text = result === null ? 'Contacting the service.' : 'Contacting the service. The previous status stays valid until the check finishes.'
    return {
      summary: { kind: 'loading', title: 'Checking…', text },
      notice: null,
      items: [CONNECTION, ANALYSIS, HISTORY].map((title) => ({ title, pill: 'wait', label: 'Checking', text: WAITING })),
      announcement: null,
    }
  }
  if (result.kind === 'ready') {
    return {
      summary: result.notice
        ? { kind: 'degraded', title: 'Ready for analysis', text: 'Recordings can be analyzed.' }
        : { kind: 'ready', title: 'Ready for analysis', text: 'Recordings can be analyzed. History and statistics are available.' },
      notice: result.notice ? noticeText(result.failing) : null,
      items: [CONNECTED, ...partItems(result.failing)],
      announcement: `Status checked: Ready for analysis${result.notice ? ', with a notice' : ''}.`,
    }
  }
  if (result.source === 'service') {
    return {
      summary: { kind: 'down', title: 'Unavailable', text: serviceDownText(result.failing) },
      notice: null,
      items: [CONNECTED, ...partItems(result.failing)],
      announcement: 'Status checked: Unavailable.',
    }
  }
  // No answer: prototype lines 691 and 1196-1198; "Unreachable" for nginx's 503 is a proposal, and its text drops
  // the prototype's "Check your network connection.", which a 503 from nginx would contradict.
  return {
    summary: {
      kind: 'down',
      title: 'Unavailable',
      text: 'The console cannot reach the analysis service. No recording can be analyzed until the connection is back. Stored results cannot be loaded either.',
    },
    notice: null,
    items: [
      { title: CONNECTION, pill: 'err', label: 'Unreachable', text: 'This console cannot reach the analysis service.' },
      { title: ANALYSIS, pill: 'err', label: 'Unavailable', text: 'Cannot be checked while the service is unreachable.' },
      { title: HISTORY, pill: 'err', label: 'Unavailable', text: 'Stored results cannot be loaded while the service is unreachable.' },
    ],
    announcement: 'Status checked: Unavailable.',
  }
}

// "Last checked" only for the API's own time (a valid 200); "Last attempt" for the browser's clock (no answer).
export function timeLabel(result: StatusState['result']): 'Last checked' | 'Last attempt' | null {
  if (result === null) return null
  return result.timeSource === 'api' ? 'Last checked' : 'Last attempt'
}
