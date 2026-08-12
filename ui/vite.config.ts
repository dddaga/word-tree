import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    allowedHosts: ['dhirajs-mac-mini.tail2416e7.ts.net'],
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
