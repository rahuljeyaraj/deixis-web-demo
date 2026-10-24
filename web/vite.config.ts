import { defineConfig } from 'vitest/config';

// The Python server (FastAPI, port 8000) answers /api; Vite serves the page in development.
export default defineConfig({
  optimizeDeps: { exclude: ['@mujoco/mujoco'] },
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8000', '/healthz': 'http://127.0.0.1:8000' },
  },
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 2000,
    // Three pages: the demo itself, a look at the end demo's shelf, and a look at the
    // workbench, each with nothing driving it.
    rollupOptions: {
      input: { main: 'index.html', shelf: 'shelf.html', workbench: 'workbench.html' },
    },
  },
  // PROBES=1 runs the measuring scripts in tests/probes instead of the tests.
  test: {
    include: process.env.PROBES ? ['tests/probes/*.test.ts'] : ['tests/*.test.ts'],
    testTimeout: 60000,
  },
});
