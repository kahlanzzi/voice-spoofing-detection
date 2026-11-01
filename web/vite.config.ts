/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Dev loop (A12): the dev server forwards /api to nginx on 8001 and presents the Origin the API
// allows (configs/api.yaml allowed_origins), so api.yaml stays unchanged. Dev server only:
// `vite preview` and the production build get no proxy.
const API_ORIGIN = 'http://localhost:8001'

export default defineConfig({
  plugins: [react()],
  server: {
    // Under Vitest only, the contract tests read api/schemas.py and docs/api_design.md
    // (src/api/types.test.ts); `.` keeps Vite's default. Folders, because a file entry does not admit
    // a `?raw` import. The dev server serves web/ only, as before.
    ...(process.env.VITEST ? { fs: { allow: ['.', '../api', '../docs'] } } : {}),
    proxy: {
      '/api': {
        target: API_ORIGIN,
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => {
            if (proxyReq.getHeader('origin')) proxyReq.setHeader('origin', API_ORIGIN)
          })
        },
      },
    },
  },
  test: {
    environment: 'jsdom',
    // Process CSS in tests: without it Vitest returns '' for CSS, `?raw` included, and the token and
    // colour tests would read empty files.
    css: true,
  },
})
