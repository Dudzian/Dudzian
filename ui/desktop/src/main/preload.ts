import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('desktopAPI', {
  getPerformanceLog: () => ipcRenderer.invoke('perf:log'),
  clearPerformanceLog: () => ipcRenderer.invoke('perf:clear'),
  appendPerformanceLog: (entry: string) => ipcRenderer.invoke('perf:append', entry)
});
