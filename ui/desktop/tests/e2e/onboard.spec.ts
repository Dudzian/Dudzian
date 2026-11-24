import { expect, test } from './fixtures';

const STEP_LABELS = ['Nazwa i profil ryzyka', 'Alokacja kapitału', 'Giełdy', 'Podsumowanie'];

test.describe('Onboarding wizard', () => {
  test('guides user through preset creation', async ({ page }) => {
    for (const label of STEP_LABELS) {
      await expect(page.locator('.stepper__step', { hasText: label })).toBeVisible();
    }

    await page.getByLabel('Nazwa presetu').fill('Test preset');
    await page.getByLabel('Agresywny').check();

    await page.getByRole('button', { name: 'Dalej' }).click();
    await expect(page.locator('.metrics-list__item').first()).toBeVisible();

    await page.getByLabel('Spot %').fill('50');
    await page.getByLabel('Futures %').fill('30');
    await page.getByLabel('Staking %').fill('20');

    await page.getByRole('button', { name: 'Dalej' }).click();
    await page.getByLabel('Filtruj giełdy').fill('kra');
    await page.getByRole('checkbox', { name: 'Kraken' }).check();

    await page.getByRole('button', { name: 'Dalej' }).click();
    await expect(page.getByRole('listitem').first()).toContainText('Test preset');
  });
});

test('odmawia przejścia dla błędnej nazwy i alokacji po migracji resolvera', async ({ page }) => {
  const next = page.getByRole('button', { name: 'Dalej' });
  await page.getByLabel('Nazwa presetu').fill('ab');
  await expect(next).toBeDisabled();
  await page.getByLabel('Nazwa presetu').fill('Poprawny preset');
  await expect(next).toBeEnabled();
  await next.click();

  await page.getByLabel('Spot %').fill('101');
  await expect(next).toBeDisabled();
  await page.getByLabel('Spot %').fill('40');
  await expect(next).toBeEnabled();
  await page.getByLabel('Futures %').fill('50');
  await expect(next).toBeDisabled();
});

test('odmawia przejścia bez wybranej giełdy', async ({ page }) => {
  const next = page.getByRole('button', { name: 'Dalej' });
  await expect(next).toBeEnabled();
  await next.click();
  await next.click();
  await page.getByRole('checkbox', { name: 'Binance' }).uncheck();
  await expect(next).toBeDisabled();
});
