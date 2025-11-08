import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef } from 'react';
import { markInteraction, recordStepDuration, registerPerformancePublisher } from '@perf/profiler';

type ProfilerHandle = {
  recordStepDuration: (step: string) => void;
  markInteraction: (label: string) => void;
};

type PerformanceProfilerContextValue = (namespace: string) => ProfilerHandle;

const PerformanceProfilerContext = createContext<PerformanceProfilerContextValue | undefined>(undefined);

export const PerformanceProfilerProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const interactionBuffer = useRef<string[]>([]);

  useEffect(() => {
    if (typeof window === 'undefined' || !window.desktopAPI?.appendPerformanceLog) {
      return;
    }

    const unregister = registerPerformancePublisher((entry) => {
      return window.desktopAPI.appendPerformanceLog(entry);
    });

    return unregister;
  }, []);

  const createHandle = useCallback<PerformanceProfilerContextValue>((namespace) => {
    return {
      recordStepDuration: (step: string) => {
        recordStepDuration(`${namespace}:${step}`);
      },
      markInteraction: (label: string) => {
        const entry = `${namespace}:${label}`;
        interactionBuffer.current.push(entry);
        markInteraction(entry);
      }
    };
  }, []);

  const value = useMemo(() => createHandle, [createHandle]);

  return <PerformanceProfilerContext.Provider value={value}>{children}</PerformanceProfilerContext.Provider>;
};

export function usePerformanceProfiler(namespace: string): ProfilerHandle {
  const context = useContext(PerformanceProfilerContext);
  if (!context) {
    throw new Error('usePerformanceProfiler must be used inside PerformanceProfilerProvider');
  }
  return context(namespace);
}
