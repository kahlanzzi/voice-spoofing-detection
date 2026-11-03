// Spinner (Stage 18 PR 8a; HANDOFF line 149, prototype line 206). Decorative: the busy state is told by the label,
// disabled and aria-busy on its button, or by the text beside it. base.css stops it under reduced motion.
import './Spinner.css'

export default function Spinner() {
  return <span className="spin" aria-hidden="true" />
}
