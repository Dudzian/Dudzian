# Dynamic ensemble weighting

Nowe zespoły modeli Decision Engine potrafią reagować na zmianę reżimu
rynkowego oraz pewność wynikającą z meta-labelingu. Każda definicja
`EnsembleDefinition` może określić:

* **`regime_weights`** – mapowanie nazw reżimów (`trend`, `daily`,
  `mean_reversion`) na wektory wag. Podczas agregacji wybierany jest
  odpowiedni profil wag zależnie od ostatniej oceny
  `MarketRegimeClassifier`.
* **`meta_weight_floor`** – minimalna waga stosowana przy skalowaniu
  komponentów na podstawie meta-labelingu.

Po treningu każdy artefakt modelu zapisuje metryki meta-etykiet w bloku
`metadata.meta_labeling`. `DecisionModelInference` ładuje prosty
klasyfikator logistyczny, którego wynik średniowany jest z klasyczną
kalibracją sygnału. Menedżer zespołów (`AIManager`) odczytuje uśrednione
trafności z artefaktów, co pozwala mu dociążać modele o wyższym
prawdopodobieństwie sukcesu w aktualnym reżimie.

Jeżeli dane treningowe nie pozwalają na zbudowanie stabilnego
meta-klasyfikatora, w artefakcie nadal pojawi się blok
`meta_labeling` z metrykami (m.in. `hit_rate`). Dzięki temu logika
ważenia nadal dysponuje informacją o jakości sygnału, a wagi mogą być
utrzymane powyżej progu `meta_weight_floor`.

Definicję w manifestach można rozszerzyć o:

```yaml
ensembles:
  - name: hybrid
    components: [light, heavy]
    aggregation: weighted
    weights: [0.6, 0.4]
    regime_weights:
      trend: [0.8, 0.2]
      mean_reversion: [0.4, 0.6]
    meta_weight_floor: 0.05
```

Jeżeli dla danego reżimu nie ma zdefiniowanych wag, menedżer użyje
globalnych wartości (`weights`). Gdy meta-labeling nie jest dostępny,
agregacja zachowuje dotychczasową logikę. `AIManager` buforuje także
trafności z artefaktów w repozytorium – jeśli inference nie został
jeszcze załadowany, odczyta `metadata.meta_labeling.subsets.*.hit_rate`
bezpośrednio z pliku i wykorzysta ją przy skalowaniu wag.
