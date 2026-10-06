# Stage 9 — initial LPPI successor authority freeze

Normatywny subordinate contract znajduje się w
`stage9_lppi_authority_key_binding_contract.json`, a jego hash w
`stage9_lppi_authority_key_binding_freeze.json`. Guard porównuje exact fields,
profile, domains, statusy i creation outcomes z runtime oraz potwierdza
niezmienność parent contract. SHA256 subordinate contract:

```text
4f201676b786f2c2c7d816457f75d7aaaf40a3cb4f21ab7653b425b48638f1a8
```

Zakres obejmuje tylko initial successor (generation `1`): świeże client-side
package acceptance z exact pre-enrollment key, osobny persistent machine CNG key,
TPM creation custody, continuity, successor PoP i jeden trwały ACTIVE record.
Pre-enrollment fingerprint to SHA256(SEC1), authority fingerprint to
SHA256(canonical TPMT_PUBLIC). Pre-enrollment key pozostaje retained.

Stan `CREATION_RESERVED` powstaje przed native effect. Pole `creation_outcome`
rozróżnia `NOT_ATTEMPTED`, `ATTEMPT_STARTED`, `CREATE_COLLISION`, `CREATE_FAILED`,
`OWN_FINALIZE_PENDING` i `OWN_FINALIZED`. Sam marker próby nie ustanawia ownership.
CREATE zaczyna od `NCryptCreatePersistedKey`, bez wcześniejszego `open_key`.
`NTE_EXISTS` z Create lub Finalize utrwala terminal collision; inne błędy Create,
pusty handle albo failure wymaganych properties pozostają fail-closed.

`OWN_FINALIZE_PENDING` powstaje po successful own Create, niepustym własnym handle
i zastosowaniu properties; zapis z fsync poprzedza jedyny Finalize. Tylko own
outcomes pozwalają na RECONCILE bez mintu. RECOVER dodatkowo wymaga exact retained
SEC1 i Unique Name. Gdy po Finalize można zakwalifikować własny handle, jego exact
identity utrwala się przed free/reopen; lost response i process crash zachowują
own-stage-authorized reconciliation. Collision nigdy nie uprawnia do adopcji.
`CANDIDATE` i kolejne statusy wymagają outcome `OWN_FINALIZED`.
Starszy record bez `creation_outcome` jest odrzucany bez automatycznej migracji.
Retry zachowuje exact candidate, binding, signatures i creation timestamp.
Capabilities wymagają prywatnej runtime provenance i ponownej kwalifikacji;
publiczne artefakty ani status `ACTIVE` nie są samodzielną authority.

`stage_9=IN_PROGRESS`, `production_provisioning_ready=false`,
`windows_production_ready=NOT_READY`. Fizyczne Windows/TPM gates: `NOT_RUN`.
`LEGAL_PRODUCTION_ENROLLMENT=NOT_PERFORMED`. Membership, provisioning operations,
account genesis, freshness, Secret Resource oraz Stage 10 pozostają poza zakresem.
