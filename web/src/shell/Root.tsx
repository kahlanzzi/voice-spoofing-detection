// The root layout (Stage 18 PR 6a): every route, with the page-change rules rendered once beside it.
import { Outlet } from 'react-router'
import PageChange from './PageChange'

export default function Root() {
  return (
    <>
      <Outlet />
      <PageChange />
    </>
  )
}
