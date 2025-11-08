export type StrategyListing = {
  id: string;
  name: string;
  description: string;
  tags: string[];
  segment: 'futures' | 'spot' | 'options';
};
