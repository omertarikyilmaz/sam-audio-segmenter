import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react()],
    server: {
        host: '0.0.0.0',
        port: 5173,
        proxy: {
            '/api/v1/pipelines/sam-audio': {
                target: 'http://backend:8011',
                changeOrigin: true,
            }
        }
    }
})
