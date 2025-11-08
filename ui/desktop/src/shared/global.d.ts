export {}; // ensure module scope

declare global {
  interface Window {
    desktopAPI: {
      getPerformanceLog: () => Promise<string[]>;
      clearPerformanceLog: () => Promise<void>;
      appendPerformanceLog: (entry: string) => Promise<void>;
    };
  }
}
