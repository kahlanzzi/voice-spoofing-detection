// S3 and S4 (Stage 18 PR A and B; HANDOFF §1 and lines 177-178, prototype lines 918-941): the selected file, its checks
// and the wait. Each check is a list item with a hidden "Passed:" / "Failed:" prefix; a check that is neither has none.
// The failure text is the one alert (Q22). The file name is shown here only.
//
// Analyze recording (prototype lines 911, 938): enabled when no check failed and none is waiting, so a duration the
// browser could not read does not block it (the service will check, line 909), and a status of Unavailable only warns
// ("You can still try", line 940). Limits not ready never reach this card (line 915).
//
// S4 has two stages, not the prototype's three: fetch cannot see where the upload ends, and there are no timers (Entry
// 146 Q27). No Cancel (agreed 2026-11-01), and no "you will be notified": there is no toast yet (Entry 146).
//
// onSettled fires once the duration read has finished, so Analyze never announces a result that is still coming
// (departure from prototype line 1281, which announces before the read).
import { useEffect, type Ref } from 'react'
import Button from '../components/Button'
import Icon, { type IconName } from '../components/Icon'
import type { LimitsResponse } from '../api/types'
import Stages from '../components/Stages'
import { useStatus } from '../status/useStatus'
import { formatDuration, formatSize, runChecks, type DurationInput, type FileCheck } from '../upload/checks'
import { useDuration } from '../upload/useDuration'
import { PRIVACY_TEXT } from './AnalyzeUpload'

interface AnalyzeFileProps {
  file: File
  limits: LimitsResponse
  headingRef: Ref<HTMLHeadingElement>
  submitting: boolean
  onAnalyze: () => void
  onReplace: () => void
  onRemove: () => void
  onSettled: (allPassed: boolean) => void
}

const STAGES = ['Validating', 'Uploading and analyzing'] as const
const CURRENT_STAGE = 1

function durationLine(duration: DurationInput): string {
  if (duration === 'reading') return 'reading duration…'
  return `duration ${typeof duration === 'number' ? formatDuration(duration) : 'unknown'}`
}

function CheckRow({ check }: { check: FileCheck }) {
  const look: { className: string; icon: IconName } =
    check.ok === true
      ? { className: 'ck-ok', icon: 'check' }
      : check.ok === false
        ? { className: 'ck-bad', icon: 'x' }
        : { className: 'ck-wait', icon: 'info' }
  return (
    <li className={look.className}>
      <Icon name={look.icon} size={16} />
      <span>
        {check.ok === true && <span className="sr">Passed: </span>}
        {check.ok === false && <span className="sr">Failed: </span>}
        {check.text}
      </span>
    </li>
  )
}

export default function AnalyzeFile({ file, limits, headingRef, submitting, onAnalyze, onReplace, onRemove, onSettled }: AnalyzeFileProps) {
  const duration = useDuration(file, limits)
  const status = useStatus()
  const { checks, valid } = runChecks(file, limits, duration)
  const settled = !checks.some((check) => check.wait)
  const allPassed = checks.every((check) => check.ok === true)

  useEffect(() => {
    if (settled) onSettled(allPassed)
  })

  return (
    <section className="panel panel-pad file-card" aria-labelledby="file-h" aria-busy={submitting || undefined}>
      <div className="file">
        <div className="file-info">
          <h2 id="file-h" className="file-title muted" tabIndex={-1} ref={headingRef}>
            Selected recording <span className="tag">shown on this screen only, not stored</span>
          </h2>
          <div className="file-name">{file.name}</div>
          <div className="small muted">{`${formatSize(file.size)} · ${durationLine(duration)}`}</div>
        </div>
        {!submitting && (
          <div className="actions">
            <Button variant="quiet" size="sm" onClick={onReplace}>
              Choose another file
            </Button>
            <Button variant="destructive" size="sm" onClick={onRemove}>
              Remove file
            </Button>
          </div>
        )}
      </div>
      <ul className="checks" aria-label="Checks before upload">
        {checks.map((check) => (
          <CheckRow key={check.id} check={check} />
        ))}
      </ul>
      {checks.some((check) => check.ok === false) && (
        <p className="small input-error" role="alert">
          <strong>Input error:</strong> this file cannot be analyzed. Choose another recording that passes every check.
          Nothing was sent and no verdict was made.
        </p>
      )}
      {submitting && (
        <div className="progress">
          <Stages names={STAGES} current={CURRENT_STAGE} />
          <p className="info-line" id="progress-text">
            <Icon name="info" size={16} />
            <span>{`${STAGES[CURRENT_STAGE]}… The request is sent once. You can safely open other pages; the result will be waiting here.`}</span>
          </p>
        </div>
      )}
      <div className="actions">
        <Button variant="primary" busy={submitting} disabled={!valid} onClick={onAnalyze}>
          {submitting ? 'Analyzing…' : 'Analyze recording'}
        </Button>
        {status.result?.kind === 'unavailable' && (
          <span className="small analysis-warn">
            <Icon name="alert" size={16} />
            Analysis is currently unavailable. You can still try; the request may fail.
          </span>
        )}
      </div>
      <p className="small muted privacy">{PRIVACY_TEXT}</p>
    </section>
  )
}
