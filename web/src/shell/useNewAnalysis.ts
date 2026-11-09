// New analysis (prototype lines 761-763, PR 8b): clear the File draft, which forgets the analysis (upload/analysis.ts),
// and go to /analyze. During a POST the draft stays: navigation does not abort or resend (HANDOFF line 276), so S4
// goes on and only the navigation happens. Already on /analyze, it replaces the entry, so Back is not a second /analyze,
// and PageChange repeats the page-change rules. The rail, the top bar and the result head share this.
import { useLocation, useNavigate } from 'react-router'
import { getSnapshot as getAnalysis } from '../upload/analysis'
import { clearFileDraft } from '../upload/fileDraft'
import { RESTART_PAGE } from './pageNames'

export function useNewAnalysis(): () => void {
  const { pathname } = useLocation()
  const navigate = useNavigate()
  return () => {
    // PR 11: D1 comes first, and the blocker runs inside navigate, so this clear moves behind it.
    if (getAnalysis().status !== 'submitting') clearFileDraft()
    void navigate('/analyze', pathname === '/analyze' ? { replace: true, state: RESTART_PAGE } : undefined)
  }
}
