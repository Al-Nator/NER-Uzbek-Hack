import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const target = env.NER_API_TARGET || 'http://127.0.0.1:8000'
  const proxy = Object.fromEntries(
    ['/api', '/healthz', '/livez', '/docs', '/openapi.json'].map((path) => [
      path,
      { target, changeOrigin: true },
    ]),
  )
  return { plugins: [react()], server: { port: 5173, strictPort: true, proxy }, preview: { proxy } }
})
