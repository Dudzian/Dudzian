import { expect, test as base } from '@playwright/test';

export const test = base.extend({
  page: async ({ page }, use) => {
    await page.addInitScript(() => {
      const logs: string[] = [];
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      // @ts-ignore - desktopAPI jest wstrzykiwane tylko w środowisku testowym
      window.desktopAPI = {
        getPerformanceLog: async () => [...logs],
        clearPerformanceLog: async () => {
          logs.length = 0;
        },
        appendPerformanceLog: async (entry: string) => {
          logs.push(entry);
        }
      };
    });

    await page.goto('/');
    await use(page);
  }
});

export { expect };
