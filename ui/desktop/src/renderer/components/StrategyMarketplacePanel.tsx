import React, { useEffect, useMemo, useState } from 'react';
import { StrategyListing } from '@/shared/strategies';
import { fetchStrategies } from '@/renderer/hooks/useMarketplaceData';
import { usePerformanceProfiler } from '../contexts/PerformanceProfilerContext';

type StrategyMarketplacePanelProps = {
  selectedStrategyId: string | null;
  onSelect: (strategyId: string) => void;
};

const StrategyMarketplacePanel: React.FC<StrategyMarketplacePanelProps> = ({ selectedStrategyId, onSelect }) => {
  const [strategies, setStrategies] = useState<StrategyListing[]>([]);
  const [query, setQuery] = useState('');
  const [segment, setSegment] = useState('all');
  const { markInteraction } = usePerformanceProfiler('marketplace-panel');

  useEffect(() => {
    fetchStrategies().then(setStrategies).catch((error) => console.error('Nie udało się pobrać strategii', error));
  }, []);

  const filtered = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return strategies.filter((strategy) => {
      const matchesQuery = !normalizedQuery || strategy.name.toLowerCase().includes(normalizedQuery);
      const matchesSegment = segment === 'all' || strategy.segment === segment;
      return matchesQuery && matchesSegment;
    });
  }, [query, segment, strategies]);

  const highlight = (strategy: StrategyListing) => (strategy.id === selectedStrategyId ? 'marketplace-card--selected' : '');

  return (
    <div>
      <header>
        <h2>Marketplace strategii</h2>
        <p>Przeglądaj zaufane automaty transakcyjne. Analiza wydajności wykonywana jest w tle.</p>
      </header>
      <div className="filter-bar">
        <input
          placeholder="Wyszukaj strategię"
          value={query}
          onChange={(event) => {
            setQuery(event.target.value);
            markInteraction('search-query');
          }}
        />
        <select
          value={segment}
          onChange={(event) => {
            setSegment(event.target.value);
            markInteraction('segment-switch');
          }}
        >
          <option value="all">Wszystkie</option>
          <option value="futures">Futures</option>
          <option value="spot">Spot</option>
          <option value="options">Opcje</option>
        </select>
      </div>
      <div className="marketplace-grid" data-testid="marketplace-grid">
        {filtered.map((strategy) => (
          <article key={strategy.id} className={`marketplace-card ${highlight(strategy)}`}>
            <strong>{strategy.name}</strong>
            <span>{strategy.description}</span>
            <div className="marketplace-card__tags">
              {strategy.tags.map((tag) => (
                <span key={tag} className="marketplace-card__tag">
                  {tag}
                </span>
              ))}
            </div>
            <button
              type="button"
              className="button-primary"
              onClick={() => {
                onSelect(strategy.id);
                markInteraction(`select-${strategy.id}`);
              }}
            >
              {selectedStrategyId === strategy.id ? 'Aktywowana' : 'Aktywuj'}
            </button>
          </article>
        ))}
        {filtered.length === 0 && <p>Brak strategii dla wybranych filtrów.</p>}
      </div>
    </div>
  );
};

export default StrategyMarketplacePanel;
