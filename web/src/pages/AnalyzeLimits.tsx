// The limits block of S2 and of a kept file while limits are not ready (Stage 18 PR A; HANDOFF line 162, prototype
// lines 871-878). Idle counts as loading (prototype line 873). Failed is one alert, not also an announcement (Q22), and
// nothing is guessed (HANDOFF line 207). The Mock tag is static: /limits has no mock field, and the values stay mock
// until open decision 2 (Q6a).
import type { RefObject } from 'react'
import Banner from '../components/Banner'
import Button from '../components/Button'
import Spinner from '../components/Spinner'
import { formatList, limitInMb } from '../upload/checks'
import { retryLimits, type LimitsState } from '../upload/limits'

export default function AnalyzeLimits({ state, retryFocus }: { state: LimitsState; retryFocus: RefObject<HTMLElement | null> }) {
  // Retry unmounts the button that holds focus (HANDOFF §7), so focus moves first.
  function retry() {
    retryFocus.current?.focus()
    void retryLimits()
  }

  if (state.status === 'failed') {
    return (
      <div className="limits-failed">
        <Banner tone="err" icon="alert" role="alert">
          <strong>Upload limits could not be loaded.</strong> Analysis stays off until they load, so the console never
          guesses them.{' '}
          <Button variant="link" onClick={retry}>
            Retry
          </Button>
        </Banner>
      </div>
    )
  }
  if (state.status !== 'ready') {
    return (
      <div role="status" className="limits-wait small muted">
        <Spinner />
        Loading upload limits from the service…
      </div>
    )
  }
  const { limits } = state
  return (
    <>
      <dl className="limits" aria-label="Upload requirements">
        <div>
          <dt>Accepted formats</dt>
          <dd>{formatList(limits.formats)}</dd>
        </div>
        <div>
          <dt>Maximum size</dt>
          <dd>{`${limitInMb(limits.max_size_bytes)} MB`}</dd>
        </div>
        <div>
          <dt>Maximum duration</dt>
          <dd>{`${limits.max_duration_s} s`}</dd>
        </div>
        <div>
          <dt>Audio after analysis</dt>
          <dd>{limits.audio_deletion}</dd>
        </div>
      </dl>
      <p className="limits-src">
        <span>
          Supplied by the service (<span className="mono">GET /limits</span>)
        </span>
        <span className="tag tag-demo">Mock values · real limits not decided</span>
      </p>
    </>
  )
}
