const assert = require('node:assert/strict');
const fs = require('node:fs');
const Module = require('node:module');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

test('built entry resolves its real dependencies, preload and renderer', async () => {
  const projectRoot = path.resolve(__dirname, '..');
  const packageJson = JSON.parse(fs.readFileSync(path.join(projectRoot, 'package.json'), 'utf8'));
  const entry = path.resolve(projectRoot, packageJson.main);
  const entryCode = fs.readFileSync(entry, 'utf8');

  const windows = [];
  class BrowserWindow {
    constructor(options) {
      this.options = options;
      windows.push(this);
    }
    loadFile(file) {
      this.loadedFile = file;
    }
  }
  const electron = {
    app: { whenReady: () => Promise.resolve(), on() {} },
    BrowserWindow,
    ipcMain: { handle() {} }
  };
  const actualRequire = Module.createRequire(entry);
  const requireWithElectronStub = (name) => name === 'electron' ? electron : actualRequire(name);
  const compiled = vm.runInNewContext(
    Module.wrap(entryCode),
    { process: { env: { NODE_ENV: 'production' }, platform: process.platform } },
    { filename: entry }
  );
  const mainModule = { exports: {} };
  compiled(mainModule.exports, requireWithElectronStub, mainModule, entry, path.dirname(entry));
  await Promise.resolve();

  assert.equal(windows.length, 1);
  const window = windows[0];
  assert.ok(fs.statSync(window.options.webPreferences.preload).isFile());
  assert.equal(window.options.webPreferences.contextIsolation, true);
  assert.equal(window.options.webPreferences.nodeIntegration, false);
  assert.equal(window.loadedFile, path.join(projectRoot, 'dist/renderer/index.html'));
  assert.ok(fs.statSync(window.loadedFile).isFile());
});
