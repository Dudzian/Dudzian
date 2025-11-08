import React, { ChangeEvent, useCallback, useEffect, useMemo, useState } from 'react';
import { useForm, Controller } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { useDashboardData } from '../hooks/useDashboardData';
import { usePerformanceProfiler } from '../contexts/PerformanceProfilerContext';

const presetSchema = z.object({
  name: z.string().min(3, 'Nazwa powinna mieć co najmniej 3 znaki'),
  riskLevel: z.enum(['conservative', 'balanced', 'aggressive']),
  allocation: z.object({
    spot: z.number().min(0).max(100),
    futures: z.number().min(0).max(100),
    staking: z.number().min(0).max(100)
  }).refine((value) => value.spot + value.futures + value.staking === 100, 'Alokacja musi sumować się do 100%'),
  exchanges: z.array(z.string()).nonempty('Wybierz przynajmniej jedną giełdę')
});

type PresetFormValues = z.infer<typeof presetSchema>;

type PresetWizardProps = {
  onActivateStrategy: (strategyId: string | null) => void;
};

const exchanges = ['Binance', 'Kraken', 'OKX', 'Bitget'];
const exchangeRegions: Record<string, 'eu' | 'asia'> = {
  Binance: 'asia',
  Kraken: 'eu',
  OKX: 'asia',
  Bitget: 'asia'
};

const steps = [
  { id: 'naming', label: 'Nazwa i profil ryzyka' },
  { id: 'allocation', label: 'Alokacja kapitału' },
  { id: 'exchanges', label: 'Giełdy' },
  { id: 'summary', label: 'Podsumowanie' }
];

const defaultValues: PresetFormValues = {
  name: 'Nowy preset',
  riskLevel: 'balanced',
  allocation: { spot: 40, futures: 40, staking: 20 },
  exchanges: ['Binance']
};

const PresetWizard: React.FC<PresetWizardProps> = ({ onActivateStrategy }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [presetId, setPresetId] = useState<string | null>(null);
  const [exchangeFilter, setExchangeFilter] = useState('');
  const [exchangeRegion, setExchangeRegion] = useState<'all' | 'eu' | 'asia'>('all');
  const { register, control, handleSubmit, watch, formState, trigger } = useForm<PresetFormValues>({
    resolver: zodResolver(presetSchema),
    mode: 'onChange',
    defaultValues
  });
  const { recordStepDuration, markInteraction } = usePerformanceProfiler('preset-wizard');
  const dashboardData = useDashboardData('preset-wizard');

  useEffect(() => {
    void trigger();
  }, [trigger]);

  useEffect(() => {
    recordStepDuration(`step-${currentStep}`);
  }, [currentStep, recordStepDuration]);

  const nextStep = useCallback(() => {
    const stepId = steps[currentStep]?.id ?? `unknown-${currentStep}`;
    recordStepDuration(`step-${currentStep}`);
    markInteraction(`next-${stepId}`);
    setCurrentStep((step) => Math.min(step + 1, steps.length - 1));
  }, [currentStep, markInteraction, recordStepDuration]);

  const previousStep = useCallback(() => {
    const stepId = steps[currentStep]?.id ?? `unknown-${currentStep}`;
    markInteraction(`prev-${stepId}`);
    setCurrentStep((step) => Math.max(step - 1, 0));
  }, [currentStep, markInteraction]);

  const onSubmit = useCallback(
    (values: PresetFormValues) => {
      const generatedPresetId = `${values.name.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}`;
      setPresetId(generatedPresetId);
      onActivateStrategy(generatedPresetId);
      markInteraction('submit');
    },
    [markInteraction, onActivateStrategy]
  );

  const nameRegister = register('name', {
    onBlur: () => markInteraction('name-updated')
  });
  const riskLevelRegister = register('riskLevel', {
    onChange: (event: ChangeEvent<HTMLInputElement>) => {
      markInteraction(`risk-${event.target.value}`);
    }
  });
  const exchangesRegister = register('exchanges', {
    onChange: (event: ChangeEvent<HTMLInputElement>) => {
      const value = event.target.value;
      const action = event.target.checked ? 'select' : 'deselect';
      markInteraction(`exchange-${action}-${value}`);
    }
  });

  const exchangeSelection = watch('exchanges');
  const allocation = watch('allocation');
  const disableNext =
    currentStep !== steps.length - 1 && (!formState.isValid || formState.isValidating);

  const summaryItems = useMemo(
    () => [
      { label: 'Nazwa', value: watch('name') },
      { label: 'Profil ryzyka', value: watch('riskLevel') },
      { label: 'Alokacja spot', value: `${allocation.spot}%` },
      { label: 'Alokacja futures', value: `${allocation.futures}%` },
      { label: 'Alokacja staking', value: `${allocation.staking}%` },
      { label: 'Giełdy', value: exchangeSelection.join(', ') }
    ],
    [allocation, exchangeSelection, watch]
  );

  const filteredExchanges = useMemo(() => {
    return exchanges.filter((exchange) => {
      const matchesQuery = exchange.toLowerCase().includes(exchangeFilter.trim().toLowerCase());
      const matchesRegion = exchangeRegion === 'all' || exchangeRegions[exchange] === exchangeRegion;
      return matchesQuery && matchesRegion;
    });
  }, [exchangeFilter, exchangeRegion]);

  return (
    <form className="wizard" onSubmit={handleSubmit(onSubmit)}>
      <header>
        <h2>Konfiguracja presetu</h2>
        <p>Zaprezentowane metryki aktualizowane są w czasie rzeczywistym dzięki buforowaniu dashboardu.</p>
      </header>
      <div className="stepper">
        {steps.map((step, index) => (
          <div key={step.id} className={`stepper__step ${index === currentStep ? 'stepper__step--active' : ''}`}>
            {step.label}
          </div>
        ))}
      </div>

      {currentStep === 0 && (
        <section>
          <label>
            Nazwa presetu
            <input {...nameRegister} placeholder="Wprowadź nazwę" />
          </label>
          <fieldset>
            <legend>Profil ryzyka</legend>
            <label>
              <input type="radio" value="conservative" {...riskLevelRegister} /> Konserwatywny
            </label>
            <label>
              <input type="radio" value="balanced" {...riskLevelRegister} /> Zrównoważony
            </label>
            <label>
              <input type="radio" value="aggressive" {...riskLevelRegister} /> Agresywny
            </label>
          </fieldset>
        </section>
      )}

      {currentStep === 1 && (
        <section>
          <div className="metrics-list">
            <div className="metrics-list__item">
              <span>ROI 30d</span>
              <strong>{dashboardData.metrics.roi30d}%</strong>
            </div>
            <div className="metrics-list__item">
              <span>Max DD</span>
              <strong>{dashboardData.metrics.maxDrawdown}%</strong>
            </div>
            <div className="metrics-list__item">
              <span>Sharpe</span>
              <strong>{dashboardData.metrics.sharpe}</strong>
            </div>
          </div>
          <label>
            Spot %
            <Controller
              control={control}
              name="allocation.spot"
              render={({ field }) => (
                <input
                  type="number"
                  value={field.value}
                  onChange={(event) => field.onChange(Number(event.target.value))}
                />
              )}
            />
          </label>
          <label>
            Futures %
            <Controller
              control={control}
              name="allocation.futures"
              render={({ field }) => (
                <input
                  type="number"
                  value={field.value}
                  onChange={(event) => field.onChange(Number(event.target.value))}
                />
              )}
            />
          </label>
          <label>
            Staking %
            <Controller
              control={control}
              name="allocation.staking"
              render={({ field }) => (
                <input
                  type="number"
                  value={field.value}
                  onChange={(event) => field.onChange(Number(event.target.value))}
                />
              )}
            />
          </label>
        </section>
      )}

      {currentStep === 2 && (
        <section>
          <div className="filter-bar">
            <input
              aria-label="Filtruj giełdy"
              placeholder="Szukaj giełdy"
              value={exchangeFilter}
              onChange={(event) => {
                setExchangeFilter(event.target.value);
                markInteraction('filter-exchange-query');
              }}
            />
            <select
              aria-label="Filtr regionu"
              value={exchangeRegion}
              onChange={(event) => {
                const value = event.target.value as 'all' | 'eu' | 'asia';
                setExchangeRegion(value);
                markInteraction('filter-exchange-region');
              }}
            >
              <option value="all">Wszystkie regiony</option>
              <option value="eu">Europa</option>
              <option value="asia">Azja</option>
            </select>
          </div>
          <div className="marketplace-grid">
            {filteredExchanges.map((exchange) => {
              const selected = exchangeSelection.includes(exchange);
              return (
                <label key={exchange} className={`marketplace-card ${selected ? 'marketplace-card--selected' : ''}`}>
                  <span>{exchange}</span>
                  <span className="marketplace-card__tag">Pro</span>
                  <input type="checkbox" value={exchange} {...exchangesRegister} />
                </label>
              );
            })}
            {filteredExchanges.length === 0 && <p>Brak giełd dla zastosowanych filtrów.</p>}
          </div>
        </section>
      )}

      {currentStep === 3 && (
        <section>
          <ul>
            {summaryItems.map((item) => (
              <li key={item.label}>
                <strong>{item.label}:</strong> {item.value}
              </li>
            ))}
          </ul>
          {presetId && (
            <div className="notice">
              Preset zapisany z identyfikatorem <code>{presetId}</code>
            </div>
          )}
        </section>
      )}

      <footer style={{ marginTop: 24, display: 'flex', justifyContent: 'space-between' }}>
        <button type="button" className="button-primary" onClick={previousStep} disabled={currentStep === 0}>
          Wstecz
        </button>
        {currentStep === steps.length - 1 ? (
          <button type="submit" className="button-primary">
            Aktywuj preset
          </button>
        ) : (
          <button type="button" className="button-primary" onClick={nextStep} disabled={disableNext}>
            Dalej
          </button>
        )}
      </footer>
    </form>
  );
};

export default PresetWizard;
