// An in-app path that is not an app route (nginx answers a direct load with its JSON 404). Shows no data, so
// it needs no session. Stub (Stage 18 PR 3c).
import { Link } from 'react-router'

export default function NotFound() {
  return (
    <main>
      <h1>Page not found</h1>
      <p>
        <Link to="/analyze">Go to Analyze</Link>
      </p>
    </main>
  )
}
