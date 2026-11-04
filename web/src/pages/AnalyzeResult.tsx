// S6 (Stage 18 PR B; HANDOFF §1, prototype lines 859-863): the result page. No static notice here: the card carries the
// service's own. The head's line has no time, since the response has none: "New result" pairs with the prototype's
// "Stored result" (line 1012; copy proposal, Entry 146). No note panel until PR 11, so one column.
import type { Ref } from 'react'
import type { AnalysisResponse } from '../api/types'
import Button from '../components/Button'
import Icon from '../components/Icon'
import ResultCard from '../components/ResultCard'
import { useNewAnalysis } from '../shell/useNewAnalysis'

interface AnalyzeResultProps {
  analysis: AnalysisResponse
  fileName: string
  headingRef: Ref<HTMLHeadingElement>
}

export default function AnalyzeResult({ analysis, fileName, headingRef }: AnalyzeResultProps) {
  const startNewAnalysis = useNewAnalysis()
  return (
    <>
      <div className="page-head">
        <div>
          <h1 tabIndex={-1}>Analysis result</h1>
          <p className="page-sub">New result</p>
        </div>
        <div className="actions">
          <Button variant="primary" icon={<Icon name="plus" size={16} />} onClick={startNewAnalysis}>
            New analysis
          </Button>
        </div>
      </div>
      <ResultCard analysis={analysis} fileName={fileName} headingRef={headingRef} />
    </>
  )
}
