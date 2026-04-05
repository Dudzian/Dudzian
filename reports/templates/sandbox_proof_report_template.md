# Sandbox Proof Report Template

## 1. Meta
- Run ID:
- Date (UTC):
- Operator:
- Repo commit:
- Environment:
  - [ ] local mock/integration
  - [ ] real testnet sandbox
- Notes:

## 2. Verdict taxonomy (must use one)
- PASS
- FAIL
- NOT EXECUTED
- SKIPPED
- BLOCKED
- ENVIRONMENTAL LIMITATION

## 3. Evidence index
| Artifact | Path | SHA256 / checksum | Comment |
|---|---|---|---|
| pytest logs |  |  |  |
| load test logs |  |  |  |
| metrics snapshot |  |  |  |
| signal quality report |  |  |  |

## 4. Adapter x area verdict matrix (required)
> Każdy wiersz musi zawierać realny test reference (np. dokładny pytest node ID) i artefakt.

| Adapter | Area | Verdict | Test / node reference | Artifact path | Rationale |
|---|---|---|---|---|---|
| Binance | Stability |  |  |  |  |
| Binance | Rate limits |  |  |  |  |
| Binance | Failover |  |  |  |  |
| Binance | Partial fills |  |  |  |  |
| Binance | Recovery |  |  |  |  |
| Kraken | Stability |  |  |  |  |
| Kraken | Rate limits |  |  |  |  |
| Kraken | Failover |  |  |  |  |
| Kraken | Partial fills |  |  |  |  |
| Kraken | Recovery |  |  |  |  |
| OKX | Stability |  |  |  |  |
| OKX | Rate limits |  |  |  |  |
| OKX | Failover |  |  |  |  |
| OKX | Partial fills |  |  |  |  |
| OKX | Recovery |  |  |  |  |

## 5. Scenario execution log (required)

### 5.1 Stability
- Command(s):
- Node IDs:
- Result:
- Evidence:
- Verdict:
- Rationale:

### 5.2 Rate limits
- Command(s):
- Node IDs:
- Result:
- Evidence:
- Verdict:
- Rationale:

### 5.3 Failover
- Command(s):
- Node IDs:
- Result:
- Evidence:
- Verdict:
- Rationale:

### 5.4 Partial fills
- Command(s):
- Node IDs:
- Result:
- Evidence:
- Verdict:
- Rationale:

### 5.5 Recovery
- Command(s):
- Node IDs:
- Result:
- Evidence:
- Verdict:
- Rationale:

## 6. Metrics snapshot
| Metric | Observed value / distribution | Threshold / expectation | Verdict | Evidence |
|---|---|---|---|---|
| exchange_rate_limit_monitor_events_total |  |  |  |  |
| exchange_retry_monitor_events_total |  |  |  |  |
| exchange_rate_limit_alerts_total |  |  |  |  |
| exchange_retry_alerts_total |  |  |  |  |
| exchange_signal_fill_ratio |  |  |  |  |
| exchange_signal_status_total{status="partial"} |  |  |  |  |
| exchange_watchdog_events_total |  |  |  |  |
| exchange_watchdog_status |  |  |  |  |
| exchange_watchdog_degradation_total |  |  |  |  |

## 7. Gaps / blocked evidence
| Area | Adapter | Gap type (NOT EXECUTED/SKIPPED/BLOCKED/ENVIRONMENTAL LIMITATION) | Reason | Action owner | ETA |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

## 8. Final operational decision
- [ ] READY for controlled pilot
- [ ] READY only for additional sandbox cycle
- [ ] NOT READY

Rationale:
