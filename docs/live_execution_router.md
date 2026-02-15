# LiveExecutionRouter: reconcile/retry, metryki i decision log

Dokument opisuje aktualną semantykę `LiveExecutionRouter` dla scenariuszy błędów `place_order` w trybie live, ze szczególnym uwzględnieniem reconcile po `client_order_id`.

## 1) Kiedy router wykonuje reconcile po błędzie

Router przechodzi do kroku reconcile **wyłącznie** gdy spełnione są wszystkie warunki:

1. `place_order` zakończył się błędem zaklasyfikowanym jako:
   - `network`, albo
   - `throttling`.
2. Żądanie ma poprawne `client_order_id`.

Następnie router wykonuje **support check**: jeśli adapter nadpisuje `fetch_order_by_client_id`, wykonywane jest właściwe zdalne wywołanie reconcile; jeśli adapter używa bazowej implementacji, reconcile jest traktowany jako `unsupported` i kończy się `reconcile_not_supported_failfast` (bez retry/fallback i bez wykonywania zdalnego fetch).

### Fail-fast przy braku `client_order_id`

Dla kategorii `network` i `throttling` bez `client_order_id` router działa w trybie **fail-fast**:
- nie wykonuje reconcile,
- nie wykonuje retry,
- nie wykonuje fallbacku na kolejną giełdę.

## 2) Zachowanie dla wyników reconcile

| Wynik reconcile | Zachowanie routera |
|---|---|
| `success` (`fetch_order_by_client_id` zwraca `OrderResult`) | Router kończy obsługę zlecenia sukcesem, zapisuje status `order_reconciled` i **nie wykonuje kolejnego `place_order`**. Wynikowy `OrderResult.raw_response` jest wzbogacany o `reconciled=true` oraz `reconcile_source="fetch_order_by_client_id"` (jeśli pola nie były ustawione). |
| `not_found` (`None`) | Router zapisuje `reconcile_not_found`, inkrementuje metrykę błędu reconcile i wraca do standardowego retry/fallback zgodnie z `max_retries_per_exchange` i polityką trasy. |
| wyjątek w reconcile | Router zapisuje `reconcile_failed` (z polem `error`), inkrementuje metrykę błędu reconcile i wraca do standardowego retry/fallback. |
| reconcile unsupported (adapter nie nadpisuje `fetch_order_by_client_id`) | Router zapisuje `reconcile_not_supported_failfast` i kończy ten błąd w trybie fail-fast (bez retry/fallback). |

## 3) Metryki

### `live_orders_reconciled_total`

Metryka rośnie, gdy reconcile zwróci poprawny `OrderResult` i router zakończy zlecenie jako odzyskane po błędzie `network/throttling`.

**Labelset:**
- `exchange`
- `symbol`
- `portfolio`
- `route`

### `live_orders_reconcile_failed_total`

Metryka rośnie, gdy reconcile **nie dał rozstrzygnięcia sukcesem**, tj.:
- `reconcile_not_found` (`None`),
- `reconcile_failed` (wyjątek w reconcile).

**Labelset:**
- `exchange`
- `symbol`
- `portfolio`
- `route`

### Relacja do istniejących metryk attempts/success/errors

| Metryka | Kiedy rośnie w ścieżce reconcile |
|---|---|
| `live_orders_attempts_total` | Dla pierwotnego błędu `place_order` rośnie z `result="error"`; przy skutecznym reconcile dodatkowo rośnie z `result="reconciled"`. |
| `live_orders_success_total` | Rośnie zarówno przy standardowym sukcesie `place_order`, jak i przy sukcesie reconcile (`order_reconciled`). |
| `live_orders_errors_total` | Rośnie przy błędzie `place_order` (np. `network`/`throttling`) niezależnie od tego, czy reconcile później się powiedzie. |

## 4) Decision log i attempts

W decision logu `payload.attempts` mogą pojawić się (co najmniej) następujące statusy:

| Status | Znaczenie | Kiedy występuje |
|---|---|---|
| `error` | Pierwotny błąd próby `place_order`. | Gdy `place_order` zwraca błąd (w kontekście reconcile: zwykle `network`/`throttling` przed próbą reconcile). |
| `reconcile_failed` | Reconcile zakończył się wyjątkiem. | Gdy `fetch_order_by_client_id` rzuci wyjątek. |
| `reconcile_not_found` | Reconcile nie znalazł zlecenia po `client_order_id`. | Gdy `fetch_order_by_client_id` zwróci `None`. |
| `order_reconciled` | Reconcile odnalazł istniejące zlecenie i router kończy sukcesem. | Gdy `fetch_order_by_client_id` zwróci `OrderResult`. |
| `reconcile_not_supported_failfast` | Adapter nie wspiera reconcile i router przerywa bez dalszych prób. | Gdy adapter nie nadpisuje `fetch_order_by_client_id` (bazowa implementacja). |
| `success` | Standardowy sukces `place_order` (bez reconcile albo po wcześniejszym `reconcile_not_found`/`reconcile_failed` i kolejnym retry). | Gdy `place_order` zwróci poprawny `OrderResult`. |

Każdy reconcile-attempt zawiera pola:
- `exchange`
- `attempt`
- `status`
- `decision_event`
- `error` (tylko dla `reconcile_failed`)

### Przykład JSONL (fragment `payload.attempts`)

```json
[
  {
    "exchange": "binance",
    "attempt": "1",
    "status": "error",
    "error": "ExchangeNetworkError('timeout')"
  },
  {
    "exchange": "binance",
    "attempt": "1",
    "status": "reconcile_not_found",
    "decision_event": "reconcile_not_found"
  },
  {
    "exchange": "binance",
    "attempt": "2",
    "status": "success",
    "latency_s": "0.412391"
  }
]
```

## Uwagi operacyjne

- Reconcile jest mechanizmem idempotencyjnym opartym o `client_order_id`, który ma ograniczać ryzyko podwójnej egzekucji po niejednoznacznych błędach transportowych.
- Brak wsparcia reconcile po stronie adaptera jest traktowany konserwatywnie (fail-fast), żeby nie wykonywać potencjalnie duplikujących prób bez możliwości jednoznacznego sprawdzenia stanu zlecenia.
