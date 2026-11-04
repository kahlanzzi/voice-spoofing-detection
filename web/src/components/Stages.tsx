// Stages (Stage 18 PR B; HANDOFF line 149, prototype lines 921-923): words, never percentages. The caller says which
// stage is running; before it the stages are done, after it they wait. Icon + text, never colour alone.
import Icon from './Icon'
import Spinner from './Spinner'
import './Stages.css'

export default function Stages({ names, current }: { names: readonly string[]; current: number }) {
  return (
    <ol className="stages" aria-label="Progress">
      {names.map((name, index) => {
        const done = index < current
        const now = index === current
        return (
          <li key={name} className={done ? 'stage done' : now ? 'stage now' : 'stage'} aria-current={now ? 'step' : undefined}>
            {done && <Icon name="check" size={14} />}
            {now && <Spinner />}
            {name}
            {done && <span className="sr"> (done)</span>}
          </li>
        )
      })}
    </ol>
  )
}
