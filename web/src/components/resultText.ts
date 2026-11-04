// The result's words (Stage 18 PR B; prototype lines 591-608, HANDOFF §8): one place, so the card, its announcement and
// PR C's History cannot disagree. A lookup on the response's own values, no policy (HANDOFF line 270).
import type { AnalysisResponse, RiskLevel, Verdict } from '../api/types'

export const VERDICT_TEXT: Record<Verdict, string> = {
  authentic: 'Likely authentic voice',
  uncertain: 'Cannot decide',
  synthetic: 'Likely synthetic voice',
}

export const RISK_TEXT: Record<RiskLevel, string> = { LOW: 'Low risk', MEDIUM: 'Medium risk', HIGH: 'High risk' }

export const REQUIRED_ACTION: Record<Verdict, { heading: string; items: readonly string[] }> = {
  synthetic: {
    heading: 'Required action: do not act on the request yet',
    items: [
      'Verify the request through a separate channel, such as a callback on a number already on file.',
      'Escalate to a second approver before the request is executed.',
      'Record what you did in the analyst note.',
    ],
  },
  uncertain: {
    heading: 'Manual verification required',
    items: [
      'Do not treat this recording as authentic.',
      'Verify the request through a separate channel before acting.',
      'Record what you did in the analyst note.',
    ],
  },
  authentic: {
    heading: 'Required action: continue your normal verification procedure',
    items: [
      'Complete every check your procedure requires for this request type. Do not skip a step because of this result.',
      'Escalate if anything else about the request seems unusual.',
    ],
  },
}

export const EXPLANATION: Record<Verdict, string> = {
  synthetic:
    'The recording shows characteristics the model associates with synthetic speech, and the frozen risk policy places it above its uncertainty band. This does not prove fraud and does not identify the person responsible.',
  uncertain:
    'The model output is close enough to the policy boundary that the frozen risk policy does not choose a side. This is not a weak “authentic” result: treat the request as unverified.',
  authentic:
    'The frozen risk policy places this recording below its uncertainty band. This does not verify the caller’s identity and does not guarantee that the request is legitimate.',
}

// Prototype line 1344, with the risk word from the response's risk_level.
export function resultAnnouncement(analysis: Pick<AnalysisResponse, 'prediction' | 'risk_level'>): string {
  const { prediction, risk_level: risk } = analysis
  return `Result ready. Verdict: ${VERDICT_TEXT[prediction]}. ${RISK_TEXT[risk]}. ${REQUIRED_ACTION[prediction].heading}.`
}
