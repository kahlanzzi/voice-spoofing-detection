// S3 (Stage 18 PR A; HANDOFF §1 and line 177, prototype lines 927-943): the selected file and its checks. Each check
// is a list item with a hidden "Passed:" / "Failed:" prefix; a check that is neither has none. The failure text is the
// one alert (Q22). Analyze recording stays disabled in this PR: PR B enables it. The file name is shown here only.
//
// onSettled fires once the duration read has finished, so Analyze never announces a result that is still coming
// (departure from prototype line 1281, which announces before the read).
import { useEffect, type Ref } from 'react'
import Button from '../components/Button'
import Icon, { type IconName } from '../components/Icon'
import type { LimitsResponse } from '../api/types'
import { formatDuration, formatSize, runChecks, type DurationInput, type FileCheck } from '../upload/checks'
import { useDuration } from '../upload/useDuration'
import { PRIVACY_TEXT } from './AnalyzeUpload'

interface AnalyzeFileProps {
  file: File
  limits: LimitsResponse
  headingRef: Ref<HTMLHeadingElement>
  onReplace: () => void
  onRemove: () => void
  onSettled: (allPassed: boolean) => void
}

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

export default function AnalyzeFile({ file, limits, headingRef, onReplace, onRemove, onSettled }: AnalyzeFileProps) {
  const duration = useDuration(file, limits)
  const { checks } = runChecks(file, limits, duration)
  const settled = !checks.some((check) => check.wait)
  const allPassed = checks.every((check) => check.ok === true)

  useEffect(() => {
    if (settled) onSettled(allPassed)
  })

  return (
    <section className="panel panel-pad file-card" aria-labelledby="file-h">
      <div className="file">
        <div className="file-info">
          <h2 id="file-h" className="file-title muted" tabIndex={-1} ref={headingRef}>
            Selected recording <span className="tag">shown on this screen only, not stored</span>
          </h2>
          <div className="file-name">{file.name}</div>
          <div className="small muted">{`${formatSize(file.size)} · ${durationLine(duration)}`}</div>
        </div>
        <div className="actions">
          <Button variant="quiet" size="sm" onClick={onReplace}>
            Choose another file
          </Button>
          <Button variant="destructive" size="sm" onClick={onRemove}>
            Remove file
          </Button>
        </div>
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
      <div className="actions">
        <Button variant="primary" disabled>
          Analyze recording
        </Button>
      </div>
      <p className="small muted privacy">{PRIVACY_TEXT}</p>
    </section>
  )
}
