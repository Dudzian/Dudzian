import { z } from 'zod';

export const presetSchema = z.object({
  name: z.string().min(3, 'Nazwa powinna mieć co najmniej 3 znaki'),
  riskLevel: z.enum(['conservative', 'balanced', 'aggressive']),
  allocation: z.object({
    spot: z.number().min(0).max(100),
    futures: z.number().min(0).max(100),
    staking: z.number().min(0).max(100)
  }).refine((value) => value.spot + value.futures + value.staking === 100, 'Alokacja musi sumować się do 100%'),
  exchanges: z.array(z.string()).nonempty('Wybierz przynajmniej jedną giełdę')
});

export type PresetFormValues = z.infer<typeof presetSchema>;
