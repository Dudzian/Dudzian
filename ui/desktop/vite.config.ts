import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

const rendererRoot = path.resolve(__dirname, 'src/renderer');

export default defineConfig({
  root: rendererRoot,
  base: './',
  plugins: [react()],
  build: {
    outDir: path.resolve(__dirname, 'dist/renderer'),
    emptyOutDir: true
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
      '@perf': path.resolve(__dirname, 'perf')
    }
  },
  server: {
    port: 5173,
    strictPort: true
  }
});
