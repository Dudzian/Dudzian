const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const { pathToFileURL } = require('node:url');
const { _electron: electron } = require('playwright');
const { expect } = require('@playwright/test');

test('compiled desktop loads with the real Linux sandbox and preload IPC', { timeout: 90_000 }, async () => {
  assert.equal(process.platform, 'linux', 'this native qualification is Linux only');
  assert.notEqual(process.getuid(), 0, 'Electron must run as the normal CI user');
  assert.ok(process.env.DISPLAY, 'an actual X display or Xvfb is required');
  const projectRoot = path.resolve(__dirname, '..');
  const sandbox = '/usr/local/lib/dudzian-electron-native/chrome-sandbox';
  assert.equal(process.env.CHROME_DEVEL_SANDBOX, sandbox);
  const descriptor = fs.openSync(sandbox, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW);
  try {
    const metadata = fs.fstatSync(descriptor);
    assert.ok(metadata.isFile());
    assert.equal(metadata.uid, 0);
    assert.equal(metadata.mode & 0o7777, 0o4755);
    assert.equal(crypto.createHash('sha256').update(fs.readFileSync(descriptor)).digest('hex'),
      'd7d2a29b6bd265950a83d583a3901c5b7d6daba6ab7eca9439076bcf99343cc9');
  } finally {
    fs.closeSync(descriptor);
  }

  const application = await electron.launch({
    // Both are intentional: Playwright's defaults add no-sandbox on Linux and
    // inject a loader that appends additional Chromium switches.
    executablePath: require('electron'),
    chromiumSandbox: true,
    args: [projectRoot],
    cwd: projectRoot,
    env: { ...process.env, NODE_ENV: 'production' },
    timeout: 60_000
  });
  try {
    const page = await application.firstWindow();
    await expect(page).toHaveURL(pathToFileURL(path.join(projectRoot, 'dist/renderer/index.html')).href);
    await expect(page.getByRole('heading', { name: 'Konfiguracja Dudzian Desktop', exact: true })).toBeVisible();
    const runtime = await application.evaluate(({ app, BrowserWindow }) => {
      const window = BrowserWindow.getAllWindows()[0];
      const preferences = window.webContents.getLastWebPreferences();
      const forbidden = ['no-sandbox', 'disable-setuid-sandbox', 'disable-gpu-sandbox',
        'disable-web-security', 'allow-running-insecure-content', 'ignore-certificate-errors',
        'disable-ipc-flooding-protection', 'use-mock-keychain'];
      return {
        electronVersion: process.versions.electron,
        appPath: app.getAppPath(),
        rendererPid: window.webContents.getOSProcessId(),
        contextIsolation: preferences.contextIsolation,
        nodeIntegration: preferences.nodeIntegration,
        sandbox: preferences.sandbox,
        forbiddenSwitches: forbidden.filter(flag => app.commandLine.hasSwitch(flag))
      };
    });
    assert.equal(runtime.electronVersion, '43.8.10');
    assert.equal(runtime.appPath, projectRoot);
    assert.equal(runtime.contextIsolation, true);
    assert.equal(runtime.nodeIntegration, false);
    assert.equal(runtime.sandbox, true);
    assert.deepEqual(runtime.forbiddenSwitches, []);
    assert.ok(Number.isInteger(runtime.rendererPid) && runtime.rendererPid > 0);
    const status = fs.readFileSync(`/proc/${runtime.rendererPid}/status`, 'utf8');
    assert.match(status, /^NoNewPrivs:\s+1$/m);
    assert.match(status, /^Seccomp:\s+2$/m);
    const uid = status.match(/^Uid:\s+(\d+)/m);
    assert.ok(uid && Number(uid[1]) !== 0, 'renderer must not run as root');

    const bridge = await page.evaluate(async () => {
      const marker = 'native-smoke-ipc';
      await window.desktopAPI.appendPerformanceLog(marker);
      const appended = (await window.desktopAPI.getPerformanceLog()).includes(marker);
      await window.desktopAPI.clearPerformanceLog();
      return {
        appended,
        cleared: !(await window.desktopAPI.getPerformanceLog()).includes(marker),
        requireType: typeof window.require,
        processType: typeof window.process
      };
    });
    assert.deepEqual(bridge, { appended: true, cleared: true, requireType: 'undefined', processType: 'undefined' });
    console.log(JSON.stringify({ electronVersion: runtime.electronVersion, nativeLinux: 'PASS',
      rendererSeccomp: 2, rendererNoNewPrivs: 1, sandbox: runtime.sandbox, preloadIpc: 'PASS' }));
  } finally {
    await application.close();
  }
});
