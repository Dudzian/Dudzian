import React from 'react';
import { usePerformanceLog } from '../hooks/usePerformanceLog';

const PerformanceConsole: React.FC = () => {
  const { logs, loading, error, refresh, clear } = usePerformanceLog();

  return (
    <section className="app-section performance-console">
      <header className="performance-console__header">
        <div>
          <h2>Logi wydajności</h2>
          <p>Monitoruj interakcje użytkownika oraz pomiary czasu kroków.</p>
        </div>
        <div className="performance-console__actions">
          <button type="button" className="button-primary" onClick={() => void refresh()} disabled={loading}>
            {loading ? 'Odświeżanie…' : 'Odśwież logi'}
          </button>
          <button type="button" onClick={() => void clear()} className="button-secondary">
            Wyczyść
          </button>
        </div>
      </header>
      {error && (
        <p role="alert" className="performance-console__error">
          {error}
        </p>
      )}
      <ul className="performance-console__list" data-testid="performance-log" aria-live="polite">
        {logs.length === 0 && !loading && <li>Brak zapisanych logów.</li>}
        {logs.map((entry) => (
          <li key={entry}>
            <code>{entry}</code>
          </li>
        ))}
      </ul>
    </section>
  );
};

export default PerformanceConsole;
