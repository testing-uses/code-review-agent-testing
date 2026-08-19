import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/trigger': 'http://localhost:5000',
      '/runs': 'http://localhost:5000',
      '/status': 'http://localhost:5000',
    },
  },
})
