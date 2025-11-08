import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { DashboardCacheProvider } from './contexts/DashboardCacheContext';
import { PerformanceProfilerProvider } from './contexts/PerformanceProfilerContext';

const rootElement = document.getElementById('root');

if (!rootElement) {
  throw new Error('Root element not found');
}

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <PerformanceProfilerProvider>
      <DashboardCacheProvider>
        <App />
      </DashboardCacheProvider>
    </PerformanceProfilerProvider>
  </React.StrictMode>
);
