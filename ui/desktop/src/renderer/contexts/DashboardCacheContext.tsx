import React, { createContext, useCallback, useContext, useMemo, useRef } from 'react';
import { DashboardSnapshot, getDashboardCache, setDashboardCache } from '@perf/cache';

type DashboardCacheContextValue = {
  getSnapshot: (key: string) => DashboardSnapshot | null;
  setSnapshot: (key: string, snapshot: DashboardSnapshot) => void;
};

const DashboardCacheContext = createContext<DashboardCacheContextValue | undefined>(undefined);

export const DashboardCacheProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const localCache = useRef(new Map<string, DashboardSnapshot>());

  const getSnapshot = useCallback((key: string) => {
    return localCache.current.get(key) ?? getDashboardCache(key);
  }, []);

  const setSnapshot = useCallback((key: string, snapshot: DashboardSnapshot) => {
    localCache.current.set(key, snapshot);
    setDashboardCache(key, snapshot);
  }, []);

  const value = useMemo(() => ({ getSnapshot, setSnapshot }), [getSnapshot, setSnapshot]);

  return <DashboardCacheContext.Provider value={value}>{children}</DashboardCacheContext.Provider>;
};

export const useDashboardCache = (): DashboardCacheContextValue => {
  const context = useContext(DashboardCacheContext);
  if (!context) {
    throw new Error('useDashboardCache must be used inside DashboardCacheProvider');
  }
  return context;
};
