// S10 System status (Stage 18 PR 7b; HANDOFF §1, prototype lines 1180-1201). It reads the status store, the one
// behind the rail line (PR 7a). Entering the page does not check (prototype line 1180): the shell checked when it
// mounted, and only Check now checks again. No timer, no polling. The shell (PR 6a) owns main; the h1 takes focus
// on a page change and PageChange announces it (HANDOFF line 70).
//
// Check now keeps its label while busy, with the spinner, disabled and aria-busy (HANDOFF line 137); the store
// shares one request, so a double click sends one. When the check finishes, "Status checked: <label>." goes to
// this page's own polite region as a new node (PageChange's technique, no timer) and focus returns to the button,
// unless the analyst has moved it elsewhere. Only a Check now press arms that, so a mount, a re-render or the
// shell's own check never announce. The effect waits for React's commit: the button is enabled again by then.
//
// The summary has no role (HANDOFF line 143): the region above is what speaks. Not built here: the "Audio and
// stored data" panel (prototype lines 1202-1205, from /limits in PR 8) and the stale-status banner (Open item).
import { useEffect, useRef } from 'react'
import Icon from '../components/Icon'
import { formatCheckedAt } from '../status/checkedAt'
import { checkStatus } from '../status/status'
import { useStatus } from '../status/useStatus'
import { statusView, timeLabel, type PillKind, type SummaryKind } from './statusView'
import './Status.css'

const SUMMARY_ICON: Record<Exclude<SummaryKind, 'loading'>, 'check' | 'alert' | 'x'> = { ready: 'check', degraded: 'alert', down: 'x' }

function PillIcon({ kind }: { kind: PillKind }) {
  if (kind === 'wait') return <span className="spin" aria-hidden="true" />
  return <Icon name={kind === 'ok' ? 'check' : 'alert'} size={14} />
}

export default function Status() {
  const state = useStatus()
  const { checking, result } = state
  const view = statusView(state)
  const label = timeLabel(result)
  const mock = result !== null && 'mock' in result && result.mock
  const details = result !== null && 'details' in result ? result.details : null
  const loading = view.summary.kind === 'loading'

  const button = useRef<HTMLButtonElement>(null)
  const live = useRef<HTMLDivElement>(null)
  const armed = useRef(false)

  useEffect(() => {
    if (!armed.current || view.announcement === null) return
    armed.current = false
    const message = document.createElement('p')
    message.textContent = view.announcement
    live.current?.replaceChildren(message)
    const active = document.activeElement
    if (active === null || active === document.body || active === button.current) button.current?.focus()
  }, [view.announcement, state])

  function checkNow() {
    armed.current = true
    void checkStatus()
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1 tabIndex={-1}>System status</h1>
          <p className="page-sub">
            Whether recordings can be analyzed right now.
            {mock && (
              <>
                {' '}
                <span className="tag tag-demo">Mock status values</span>
              </>
            )}
          </p>
        </div>
        <div className="updated">
          {label !== null && result !== null && <span className="checked-at">{`${label} ${formatCheckedAt(result.checkedAt)}`}</span>}
          <button
            ref={button}
            type="button"
            className="btn btn-quiet btn-sm"
            disabled={checking}
            aria-busy={checking || undefined}
            onClick={checkNow}
          >
            {checking ? <span className="spin" aria-hidden="true" /> : <Icon name="refresh" size={14} />}
            Check now
          </button>
        </div>
      </div>

      <section className={`panel sum sum-${view.summary.kind}`} aria-labelledby="sum-h" aria-busy={loading || undefined}>
        <div className="sum-icon">
          {loading ? <span className="spin" aria-hidden="true" /> : <Icon name={SUMMARY_ICON[view.summary.kind as Exclude<SummaryKind, 'loading'>]} size={22} />}
        </div>
        <div className="sum-body">
          <p className="kicker">Current status</p>
          <h2 id="sum-h">{view.summary.title}</h2>
          <p className="sum-text">{view.summary.text}</p>
          {view.notice !== null && (
            <>
              <div className="banner-warn">
                <Icon name="alert" size={16} />
                <span>{view.notice}</span>
              </div>
              <p className="decision">
                <Icon name="decision" size={16} />
                <span>
                  <strong>Open product decision:</strong> the rule for “Degraded” is not defined yet. The service
                  reports it; this console shows it as Ready for analysis with a notice.
                </span>
              </p>
            </>
          )}
        </div>
      </section>

      <section className="panel" aria-labelledby="checks-h">
        <h2 id="checks-h" className="checks-title">What is checked</h2>
        <ul className="check-list">
          {view.items.map((item) => (
            <li className="check-item" key={item.title}>
              <div>
                <h3>{item.title}</h3>
                <p className="check-text">{item.text}</p>
              </div>
              <span className={`state-pill sp-${item.pill}`}>
                <PillIcon kind={item.pill} />
                {item.label}
              </span>
            </li>
          ))}
        </ul>
        {details !== null && (
          <details className="more">
            <summary>Technical details</summary>
            <dl className="kv">
              <dt>Model</dt>
              <dd className="mono">{details.versions.model_version}</dd>
              <dt>Preprocessing profile</dt>
              <dd className="mono">{details.versions.profile_version}</dd>
              <dt>Risk policy</dt>
              <dd className="mono">{details.versions.policy_version}</dd>
              <dt>Validation “Cannot decide” share</dt>
              <dd className="mono">{`deferral_on_fit = ${details.deferralOnFit.toFixed(4)}`}</dd>
              <dt>Manifest SHA-256</dt>
              <dd className="mono">{details.bundleManifestSha256 ?? 'Not available until the bundle is frozen.'}</dd>
            </dl>
          </details>
        )}
      </section>

      <div className="sr" aria-live="polite" aria-atomic="true" id="status-announcer" ref={live} />
    </>
  )
}
