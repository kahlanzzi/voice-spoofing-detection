// Route guard (Stage 18 PR 3c). Every app route but /signin needs a session (docs/api_design.md §4.8,
// default-deny). A component, not a loader: expire() can fire mid-page, with no navigation, and the guard must
// follow the store. Its only effect is <Navigate>: it changes no session state and clears nothing, so the file
// and note drafts, kept in module stores (A23), survive the guarded page unmounting.
import type { JSX } from 'react'
import { Navigate, Outlet, useLocation } from 'react-router'
import Checking from '../pages/Checking'
import NotFound from '../pages/NotFound'
import SignIn from '../pages/SignIn'
import { useSession } from '../session/useSession'
import { safeNext, signInPath } from './next'
import { appRoute } from './routes'

// react-router also matches /analyze/ and /history/<anything>; nginx answers those 404, so the app does too.
export function AppRoutes(): JSX.Element {
  const { pathname } = useLocation()
  return appRoute(pathname) === null ? <NotFound /> : <Outlet />
}

// checking shows neither S1 nor the page, so a signed-in analyst never sees the sign-in screen flash.
export function RequireSession(): JSX.Element {
  const session = useSession()
  const { pathname, search } = useLocation()
  if (session.status === 'checking') return <Checking />
  if (session.status === 'signedIn') return <Outlet />
  // After the analyst's own Sign out, S1 has no next: the next sign-in starts on the default page (PR 6b). Expiry
  // keeps it, so the analyst returns to the same place (HANDOFF §9).
  if (session.status === 'signedOut' && session.reason === 'signOut') return <Navigate to="/signin" replace />
  return <Navigate to={signInPath(pathname, search)} replace />
}

export function SignInGate(): JSX.Element {
  const session = useSession()
  const { search } = useLocation()
  switch (session.status) {
    case 'checking':
      return <Checking />
    case 'signedIn':
      return <Navigate to={safeNext(new URLSearchParams(search).get('next'))} replace />
    case 'unavailable':
      return <SignIn variant="unavailable" />
    case 'signedOut':
      return <SignIn variant={session.reason === 'expired' ? 'expired' : 'default'} />
  }
}
