import { app, BrowserWindow, ipcMain } from 'electron';
import path from 'path';
import { appendPerformanceEntry, clearPerformanceLog, getPerformanceLog } from '@perf/profiler';

const isDevelopment = process.env.NODE_ENV === 'development';

function createWindow(): BrowserWindow {
  const window = new BrowserWindow({
    width: 1280,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  if (isDevelopment) {
    window.loadURL('http://localhost:5173');
    window.webContents.openDevTools();
  } else {
    window.loadFile(path.join(__dirname, '../renderer/index.html'));
  }

  return window;
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

ipcMain.handle('perf:log', () => {
  return getPerformanceLog();
});

ipcMain.handle('perf:clear', () => {
  clearPerformanceLog();
});

ipcMain.handle('perf:append', (_event, entry: string) => {
  appendPerformanceEntry(entry);
});
