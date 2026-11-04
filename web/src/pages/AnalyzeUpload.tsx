// S2 (Stage 18 PR A; HANDOFF §1, prototype lines 880-887). Prototype's upload area without the drop target, which is
// optional (HANDOFF line 177) and out of the MVP, so its heading is a copy proposal, not "Drop the recorded segment
// here". No Demo scenarios panel: prototype tooling (Q23). The file input lives in Analyze, so S2 and S3 share it.
import { useRef, type Ref } from 'react'
import Button from '../components/Button'
import Icon from '../components/Icon'
import type { LimitsState } from '../upload/limits'
import AnalyzeLimits from './AnalyzeLimits'

export const PRIVACY_TEXT =
  'The audio is used only for this analysis. The file name is shown on this screen only; History and stored records never keep it.'

interface AnalyzeUploadProps {
  state: LimitsState
  chooseRef: Ref<HTMLButtonElement>
  onChoose: () => void
}

export default function AnalyzeUpload({ state, chooseRef, onChoose }: AnalyzeUploadProps) {
  const heading = useRef<HTMLHeadingElement>(null)
  return (
    <section className="panel upload" aria-labelledby="upload-h">
      <div className="upload-icon">
        <Icon name="upload" size={26} />
      </div>
      <h2 id="upload-h" tabIndex={-1} ref={heading}>
        Choose the recorded segment
      </h2>
      <p className="muted small">Browse for it on your computer.</p>
      <Button
        ref={chooseRef}
        variant="primary"
        icon={<Icon name="upload" size={16} />}
        disabled={state.status !== 'ready'}
        onClick={onChoose}
      >
        Choose recording
      </Button>
      <AnalyzeLimits state={state} retryFocus={heading} />
      <p className="small muted privacy">{PRIVACY_TEXT}</p>
    </section>
  )
}
