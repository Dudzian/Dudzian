import { useCallback, useEffect, useState } from 'react';
import { DashboardSnapshot } from '@perf/cache';
import { measureAsync } from '@perf/profiler';
import { useDashboardCache } from '../contexts/DashboardCacheContext';

type DashboardData = {
  metrics: {
    roi30d: number;
    maxDrawdown: number;
    sharpe: number;
  };
};

const FALLBACK_DATA: DashboardData = {
  metrics: {
    roi30d: 12.4,
    maxDrawdown: -8.9,
    sharpe: 1.4
  }
};

const CACHE_TTL = 1000 * 60 * 5;

async function fetchDashboardSnapshot(key: string): Promise<DashboardSnapshot> {
  return measureAsync(`dashboard-${key}`, async () => {
    await new Promise((resolve) => setTimeout(resolve, 120));
    return {
      timestamp: Date.now(),
      metrics: {
        roi30d: 12.4 + Math.random(),
        maxDrawdown: -8.5 - Math.random(),
        sharpe: 1.2 + Math.random() * 0.4
      }
    };
  });
}

export function useDashboardData(cacheKey: string): DashboardData {
  const [snapshot, setSnapshot] = useState<DashboardData>(FALLBACK_DATA);
  const { getSnapshot, setSnapshot: setCacheSnapshot } = useDashboardCache();

  const hydrate = useCallback(async () => {
    const cached = getSnapshot(cacheKey);
    if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
      setSnapshot({ metrics: cached.metrics });
      return;
    }

    const fresh = await fetchDashboardSnapshot(cacheKey);
    setCacheSnapshot(cacheKey, fresh);
    setSnapshot({ metrics: fresh.metrics });
  }, [cacheKey, getSnapshot, setCacheSnapshot]);

  useEffect(() => {
    hydrate().catch((error) => console.error('Nie udało się załadować danych dashboardu', error));
  }, [hydrate]);

  return snapshot;
}
