const performanceLog: string[] = [];
const stepTimers = new Map<string, number>();

type PerformancePublisher = (entry: string) => void | Promise<void>;

const publishers = new Set<PerformancePublisher>();

function isPromiseLike(value: unknown): value is Promise<unknown> {
  return Boolean(value) && typeof (value as Promise<unknown>).then === 'function';
}

function notifyPublishers(entry: string): void {
  for (const publisher of publishers) {
    try {
      const result = publisher(entry);
      if (isPromiseLike(result)) {
        void result.catch((error) => {
          console.error('Nie udało się opublikować logu wydajności', error);
        });
      }
    } catch (error) {
      console.error('Nie udało się opublikować logu wydajności', error);
    }
  }
}

export function appendPerformanceEntry(entry: string): void {
  performanceLog.push(entry);
  notifyPublishers(entry);
}

export function registerPerformancePublisher(publisher: PerformancePublisher): () => void {
  publishers.add(publisher);
  return () => {
    publishers.delete(publisher);
  };
}

export function recordStepDuration(step: string): void {
  const key = `step:${step}`;
  const now = performance.now();
  const start = stepTimers.get(key);
  if (typeof start === 'number') {
    const duration = now - start;
    performance.measure(key, { start, duration });
    stepTimers.delete(key);
    appendPerformanceEntry(`${key}:${duration.toFixed(2)}`);
  } else {
    stepTimers.set(key, now);
  }
}

export function markInteraction(label: string): void {
  performance.mark(`interaction:${label}`);
  appendPerformanceEntry(`interaction:${label}:${Date.now()}`);
}

export async function measureAsync<T>(label: string, fn: () => Promise<T>): Promise<T> {
  const start = performance.now();
  try {
    const result = await fn();
    appendPerformanceEntry(`async:${label}:${performance.now() - start}`);
    return result;
  } catch (error) {
    appendPerformanceEntry(`async-error:${label}`);
    throw error;
  }
}

export function getPerformanceLog(): string[] {
  return [...performanceLog];
}

export function clearPerformanceLog(): void {
  performanceLog.length = 0;
  stepTimers.clear();
}
