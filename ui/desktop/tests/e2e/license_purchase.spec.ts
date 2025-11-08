import { expect, test } from './fixtures';

test.describe('License purchase flow', () => {
  test('allows user to select plan and confirm purchase', async ({ page }) => {
    await page.getByRole('heading', { name: 'Licencje' }).scrollIntoViewIfNeeded();

    const selector = page.getByLabel('Wybór licencji');
    await selector.selectOption('institutional');

    await page.getByRole('button', { name: /Kup licencję/ }).click();
    await page.getByRole('button', { name: 'Odśwież logi' }).click();

    await expect(page.getByTestId('performance-log').getByText(/license-panel:purchase-institutional/)).toBeVisible();

    await page.getByRole('button', { name: 'Wyczyść' }).click();
    await expect(page.getByTestId('performance-log').getByText('Brak zapisanych logów.')).toBeVisible();
  });
});
