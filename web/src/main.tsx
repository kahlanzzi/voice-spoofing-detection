// IBM Plex, self-hosted (A14): only the weights the prototype loads (line 9).
import '@fontsource/ibm-plex-sans/400.css'
import '@fontsource/ibm-plex-sans/500.css'
import '@fontsource/ibm-plex-sans/600.css'
import '@fontsource/ibm-plex-sans/700.css'
import '@fontsource/ibm-plex-mono/400.css'
import '@fontsource/ibm-plex-mono/500.css'
import './styles/tokens.css'
import './styles/base.css'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter } from 'react-router'
import { RouterProvider } from 'react-router/dom'
import { routes } from './routing/router'
import { startSession } from './session/session'
import { applyTheme, readTheme } from './theme'

applyTheme(readTheme())
// Module level, not an effect: StrictMode cannot run it twice.
void startSession()
// Once, outside React: a data router is not held in React state (react-router docs, data mode installation).
const router = createBrowserRouter(routes)

const root = document.getElementById('root')
if (!root) throw new Error('index.html has no #root element')

createRoot(root).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
)
