import { expect, test } from './fixtures';

test.describe('Strategy activation flow', () => {
  test('activates selected strategy from marketplace', async ({ page }) => {
    await page.getByPlaceholder('Wyszukaj strategię').fill('Trend');
    await page.getByRole('button', { name: 'Aktywuj' }).first().click();

    const activeCard = page.locator('.marketplace-card--selected');
    await expect(activeCard).toHaveCount(1);
    await expect(activeCard.getByRole('button')).toHaveText('Aktywowana');
  });
});
