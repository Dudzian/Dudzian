import React, { useEffect, useMemo, useState } from 'react';
import { usePerformanceProfiler } from '../contexts/PerformanceProfilerContext';

type LicenseTier = {
  id: string;
  name: string;
  price: string;
  description: string;
  features: string[];
};

const LICENSE_TIERS: LicenseTier[] = [
  {
    id: 'starter',
    name: 'Starter',
    price: '49 PLN / mies.',
    description: 'Idealna dla inwestorów zaczynających z automatyzacją.',
    features: ['2 aktywne strategie', 'Monitoring 24/7', 'Wsparcie mailowe']
  },
  {
    id: 'pro',
    name: 'Pro',
    price: '129 PLN / mies.',
    description: 'Zaawansowane narzędzia oraz zaawansowane alerty.',
    features: ['10 strategii', 'Analityka premium', 'Wsparcie na czacie']
  },
  {
    id: 'institutional',
    name: 'Institutional',
    price: '349 PLN / mies.',
    description: 'Pełna obsługa i wdrożenie korporacyjne.',
    features: ['Nieograniczone strategie', 'Konsultacje z zespołem', 'Integracja dedykowana']
  }
];

type LicensePurchasePanelProps = {
  onPurchase: (licenseId: string) => void;
};

const LicensePurchasePanel: React.FC<LicensePurchasePanelProps> = ({ onPurchase }) => {
  const [selected, setSelected] = useState('pro');
  const [confirmation, setConfirmation] = useState<string | null>(null);
  const { markInteraction } = usePerformanceProfiler('license-panel');
  const tier = useMemo(() => LICENSE_TIERS.find((item) => item.id === selected) ?? LICENSE_TIERS[0], [selected]);

  useEffect(() => {
    setConfirmation(null);
  }, [selected]);

  return (
    <section aria-label="Zakup licencji" className="app-section">
      <h2>Licencje</h2>
      <p>Wybierz plan odpowiadający Twojemu wolumenowi obrotu.</p>
      <div className="filter-bar">
        <select
          aria-label="Wybór licencji"
          value={selected}
          onChange={(event) => {
            setSelected(event.target.value);
            markInteraction(`select-${event.target.value}`);
          }}
        >
          {LICENSE_TIERS.map((license) => (
            <option key={license.id} value={license.id}>
              {license.name}
            </option>
          ))}
        </select>
        <button
          type="button"
          className="button-primary"
          onClick={() => {
            markInteraction(`purchase-${tier.id}`);
            onPurchase(tier.id);
            setConfirmation(`Zakupiono licencję ${tier.name}.`);
          }}
        >
          Kup licencję {tier.price}
        </button>
      </div>
      <div className="marketplace-card">
        <strong>{tier.name}</strong>
        <span>{tier.description}</span>
        <ul>
          {tier.features.map((feature) => (
            <li key={feature}>{feature}</li>
          ))}
        </ul>
        {confirmation && (
          <div className="notice" role="status">
            {confirmation}
          </div>
        )}
      </div>
    </section>
  );
};

export default LicensePurchasePanel;
