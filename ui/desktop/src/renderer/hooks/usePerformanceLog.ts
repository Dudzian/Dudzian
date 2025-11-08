import { useCallback, useEffect, useState } from 'react';

type PerformanceLogState = {
  logs: string[];
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  clear: () => Promise<void>;
};

const notAvailableMessage = 'Interfejs desktopAPI nie jest dostępny.';

export function usePerformanceLog(auto = true): PerformanceLogState {
  const [logs, setLogs] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (typeof window === 'undefined' || !window.desktopAPI?.getPerformanceLog) {
      setError(notAvailableMessage);
      setLogs([]);
      return;
    }

    setLoading(true);
    try {
      const nextLogs = await window.desktopAPI.getPerformanceLog();
      setLogs(nextLogs);
      setError(null);
    } catch (refreshError) {
      console.error('Nie udało się pobrać logów wydajności', refreshError);
      setError('Nie udało się pobrać logów wydajności.');
    } finally {
      setLoading(false);
    }
  }, []);

  const clear = useCallback(async () => {
    if (typeof window === 'undefined' || !window.desktopAPI?.clearPerformanceLog) {
      setLogs([]);
      return;
    }

    try {
      await window.desktopAPI.clearPerformanceLog();
    } catch (clearError) {
      console.error('Nie udało się wyczyścić logów wydajności', clearError);
    }

    await refresh();
  }, [refresh]);

  useEffect(() => {
    if (!auto) {
      return;
    }

    refresh().catch((refreshError) => {
      console.error('Automatyczne odświeżenie logów nie powiodło się', refreshError);
    });
  }, [auto, refresh]);

  return { logs, loading, error, refresh, clear };
}
