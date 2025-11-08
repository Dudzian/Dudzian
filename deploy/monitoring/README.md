# Decision engine monitoring bundle

Ten katalog zawiera minimalny pakiet eksportów Prometheusa i Grafany dla nowego zestawu metryk decyzyjnych oraz guardraili ryzyka.

- `prometheus_decision_rules.yml` – reguły nagrywające tempo ocen kandydatów, powody odrzuceń i aktywacje trybu hedge.
- `grafana_decision_dashboard.json` – dashboard pokazujący najważniejsze wskaźniki Decision Engine wraz z topowymi powodami odrzuceń.

Pliki można podmontować w istniejącej instalacji (`prometheus.yml` / `provisioning/dashboards`) lub dystrybuować razem z release Stage6.
