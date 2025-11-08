import { StrategyListing } from '@/shared/strategies';

const FAKE_STRATEGIES: StrategyListing[] = [
  {
    id: 'grid-eurusdt',
    name: 'Grid EUR/USDT',
    description: 'Stabilna siatka dla pary EUR/USDT z automatycznym zarządzaniem ryzykiem.',
    tags: ['grid', 'spot', 'low-risk'],
    segment: 'spot'
  },
  {
    id: 'perp-btc-trend',
    name: 'BTC Trend Rider',
    description: 'Strategia momentum dla kontraktów perpetual BTC.',
    tags: ['momentum', 'futures'],
    segment: 'futures'
  },
  {
    id: 'options-vol',
    name: 'Options Volatility Capture',
    description: 'Opcje na zmienność z dynamicznym hedgingiem.',
    tags: ['options', 'hedging'],
    segment: 'options'
  }
];

export async function fetchStrategies(): Promise<StrategyListing[]> {
  return new Promise((resolve) => {
    setTimeout(() => resolve(FAKE_STRATEGIES), 200);
  });
}
