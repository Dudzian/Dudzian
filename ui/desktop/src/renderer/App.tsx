import React, { useMemo, useState } from 'react';
import PresetWizard from './components/PresetWizard';
import StrategyMarketplacePanel from './components/StrategyMarketplacePanel';
import LicensePurchasePanel from './components/LicensePurchasePanel';
import PerformanceConsole from './components/PerformanceConsole';
import './styles.css';

const App: React.FC = () => {
  const [selectedStrategyId, setSelectedStrategyId] = useState<string | null>(null);
  const title = useMemo(() => (selectedStrategyId ? 'Aktywacja strategii' : 'Konfiguracja Dudzian Desktop'), [selectedStrategyId]);

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>{title}</h1>
      </header>
      <main className="app-layout">
        <section className="app-column">
          <section className="app-section">
            <PresetWizard onActivateStrategy={setSelectedStrategyId} />
          </section>
          <LicensePurchasePanel onPurchase={(licenseId) => console.info('license-purchase', licenseId)} />
        </section>
        <aside className="app-aside">
          <section className="app-section">
            <StrategyMarketplacePanel selectedStrategyId={selectedStrategyId} onSelect={setSelectedStrategyId} />
          </section>
          <PerformanceConsole />
        </aside>
      </main>
    </div>
  );
};

export default App;
