// VerdictPill (Stage 18 PR C; HANDOFF line 140, prototype lines 313-315, 686): icon, short verdict and risk word, never
// colour alone. Both values come from the record as they are: the risk is risk_level, not the verdict's usual risk
// (HANDOFF line 270).
import type { RiskLevel, Verdict } from '../api/types'
import Icon, { type IconName } from './Icon'
import { VERDICT_SHORT } from './resultText'
import './VerdictPill.css'

const LOOK: Record<Verdict, { className: string; icon: IconName }> = {
  synthetic: { className: 'p-syn', icon: 'alert' },
  uncertain: { className: 'p-unc', icon: 'help' },
  authentic: { className: 'p-auth', icon: 'shieldcheck' },
}

export default function VerdictPill({ prediction, risk }: { prediction: Verdict; risk: RiskLevel }) {
  const look = LOOK[prediction]
  return (
    <span className={`pill ${look.className}`}>
      <Icon name={look.icon} size={13} />
      {`${VERDICT_SHORT[prediction]} · ${risk}`}
    </span>
  )
}
