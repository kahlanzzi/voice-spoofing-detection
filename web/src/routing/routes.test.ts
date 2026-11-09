import type { RouteObject } from 'react-router'
import { describe, expect, it } from 'vitest'
import nginxConf from '../../../deploy/nginx.conf?raw'
import { routes } from './router'
import { APP_ROUTES, appRoute, DEFAULT_ROUTE, REQUEST_ID } from './routes'

// The client route list against deploy/nginx.conf (only read): the app-route regex, the request-ID form and
// the / redirect. Like the api/schemas.py contract tests in src/api/types.test.ts.

const ID_FRAGMENT = REQUEST_ID.source.replace(/^\^|\$$/g, '')

function nginxAppRoutes(): string[] {
  const alternatives = /location ~ "\^\/\(([^)]+)\)\$"/.exec(nginxConf)?.[1]
  if (alternatives === undefined) throw new Error('deploy/nginx.conf: no app-route location')
  return alternatives.split('|').map((route) => `/${route}`)
}

function asNginx(route: string): string {
  return route.replace(':requestId', ID_FRAGMENT)
}

function routerPaths(tree: RouteObject[]): string[] {
  return tree.flatMap((route) => [...(route.path ? [route.path] : []), ...routerPaths(route.children ?? [])])
}

describe('the route list equals the nginx app routes', () => {
  it('same routes, nothing more or less', () => {
    expect(APP_ROUTES.map(asNginx).sort()).toEqual(nginxAppRoutes().sort())
  })

  it('the request-ID form is the same', () => {
    expect(ID_FRAGMENT).toBe('req_[0-9a-f]{32}')
    expect(nginxAppRoutes()).toContain(`/history/${ID_FRAGMENT}`)
  })

  it('nginx sends / to the default route', () => {
    expect(nginxConf).toMatch(new RegExp(`location = / \\{[^}]*return 302 ${DEFAULT_ROUTE};`))
  })

  it('the router has exactly these routes, plus / and the catch-all', () => {
    expect(routerPaths(routes).sort()).toEqual([...APP_ROUTES, '/', '*'].sort())
  })
})

describe('appRoute', () => {
  const id = 'req_' + 'a1'.repeat(16)

  it.each([...APP_ROUTES.filter((route) => !route.includes(':'))])('%s', (route) => {
    expect(appRoute(route)).toBe(route)
  })

  it('a full request ID', () => {
    expect(appRoute(`/history/${id}`)).toBe('/history/:requestId')
  })

  it.each([
    `/history/${id.slice(0, -1)}`,
    `/history/${id}0`,
    `/history/${id.toUpperCase()}`,
    `/history/req_${'A1'.repeat(16)}`,
    '/history/:requestId',
    '/history/',
    '/history/foo',
    '/analyze/',
    '/Analyze',
    '/',
    '',
    '/index.html',
  ])('%s → null', (pathname) => {
    expect(appRoute(pathname)).toBeNull()
  })
})
