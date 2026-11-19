import assert from 'node:assert/strict';
import test from 'node:test';
import { zodResolver } from '@hookform/resolvers/zod';
import { presetSchema, type PresetFormValues } from '../../src/shared/presetSchema';

const valid: PresetFormValues = {
  name: 'Test preset',
  riskLevel: 'balanced',
  allocation: { spot: 50, futures: 30, staking: 20 },
  exchanges: ['Kraken']
};

const resolve = (values: unknown) =>
  zodResolver(presetSchema)(values as PresetFormValues, undefined, {
    fields: {},
    shouldUseNativeValidation: false
  });

test('resolver returns the valid form values', async () => {
  assert.deepEqual(await resolve(valid), { errors: {}, values: valid });
});

for (const riskLevel of ['conservative', 'balanced', 'aggressive'] as const) {
  test(`accepts ${riskLevel} risk profile`, () => {
    assert.equal(presetSchema.safeParse({ ...valid, riskLevel }).success, true);
  });
}

for (const [label, values, field] of [
  ['short name', { ...valid, name: 'ab' }, 'name'],
  ['missing name', { ...valid, name: undefined }, 'name'],
  ['unknown risk', { ...valid, riskLevel: 'unbounded' }, 'riskLevel'],
  ['empty exchange set', { ...valid, exchanges: [] }, 'exchanges'],
  ['invalid exchange value', { ...valid, exchanges: [42] }, 'exchanges'],
  ['missing allocation', { ...valid, allocation: undefined }, 'allocation'],
  ['negative allocation', { ...valid, allocation: { spot: -1, futures: 81, staking: 20 } }, 'allocation'],
  ['over-limit allocation', { ...valid, allocation: { spot: 101, futures: 0, staking: 0 } }, 'allocation'],
  ['wrong total', { ...valid, allocation: { spot: 50, futures: 30, staking: 10 } }, 'allocation'],
  ['numeric string', { ...valid, allocation: { spot: '50', futures: 30, staking: 20 } }, 'allocation'],
  ['NaN', { ...valid, allocation: { spot: NaN, futures: 30, staking: 20 } }, 'allocation'],
  ['positive infinity', { ...valid, allocation: { spot: Infinity, futures: 30, staking: 20 } }, 'allocation'],
  ['negative infinity', { ...valid, allocation: { spot: -Infinity, futures: 30, staking: 20 } }, 'allocation']
] as const) {
  test(`resolver fails closed for ${label}`, async () => {
    assert.equal(presetSchema.safeParse(values).success, false);
    const result = await resolve(values);
    assert.deepEqual(result.values, {});
    assert.ok(Object.prototype.hasOwnProperty.call(result.errors, field));
  });
}

test('accepts allocation boundaries with a total of 100', async () => {
  const values = { ...valid, allocation: { spot: 100, futures: 0, staking: 0 } };
  assert.deepEqual(await resolve(values), { errors: {}, values });
});
