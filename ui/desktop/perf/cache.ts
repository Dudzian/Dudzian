export type DashboardSnapshot = {
  timestamp: number;
  metrics: {
    roi30d: number;
    maxDrawdown: number;
    sharpe: number;
  };
};

const dashboardCache = new Map<string, DashboardSnapshot>();

export function getDashboardCache(key: string): DashboardSnapshot | null {
  return dashboardCache.get(key) ?? null;
}

export function setDashboardCache(key: string, snapshot: DashboardSnapshot): void {
  dashboardCache.set(key, snapshot);
}

export function clearDashboardCache(): void {
  dashboardCache.clear();
}
