// The route tree for the data router (A15: data router, so useBlocker is there for D1). main.tsx creates the
// browser router once; tests use createMemoryRouter. Paths are APP_ROUTES (routes.test.ts checks it).
import { Navigate, type RouteObject } from 'react-router'
import AnalysisDetails from '../pages/AnalysisDetails'
import Analyze from '../pages/Analyze'
import History from '../pages/History'
import NotFound from '../pages/NotFound'
import Overview from '../pages/Overview'
import Status from '../pages/Status'
import { AppRoutes, RequireSession, SignInGate } from './guard'
import { DEFAULT_ROUTE } from './routes'

export const routes: RouteObject[] = [
  // nginx answers a direct load of / with 302 /analyze; this is the same for an in-app link.
  { path: '/', element: <Navigate to={DEFAULT_ROUTE} replace /> },
  {
    Component: AppRoutes,
    children: [
      { path: '/signin', caseSensitive: true, Component: SignInGate },
      {
        Component: RequireSession,
        children: [
          { path: '/analyze', caseSensitive: true, Component: Analyze },
          { path: '/history', caseSensitive: true, Component: History },
          { path: '/history/:requestId', caseSensitive: true, Component: AnalysisDetails },
          { path: '/overview', caseSensitive: true, Component: Overview },
          { path: '/status', caseSensitive: true, Component: Status },
        ],
      },
    ],
  },
  { path: '*', Component: NotFound },
]
