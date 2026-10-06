# Stage 9 — initial LPPI successor authority freeze

Normatywny subordinate contract znajduje się w
`stage9_lppi_authority_key_binding_contract.json`, a jego hash w
`stage9_lppi_authority_key_binding_freeze.json`. Guard porównuje exact fields,
profile, domains i statusy z runtime oraz potwierdza niezmienność parent contract.

Zakres obejmuje tylko initial successor (generation `1`): świeże client-side
package acceptance z exact pre-enrollment key, osobny persistent machine CNG key,
TPM creation custody, continuity, successor PoP i jeden trwały ACTIVE record.
Pre-enrollment fingerprint to SHA256(SEC1), authority fingerprint to
SHA256(canonical TPMT_PUBLIC). Pre-enrollment key pozostaje retained.

Stan `CREATION_RESERVED` powstaje przed native effect. Marker próby tworzenia
utrwala się przed NCrypt: niejednoznaczny wynik nie uprawnia do drugiego mintu.
Retry zachowuje exact candidate, binding, signatures i creation timestamp.
Capabilities wymagają prywatnej runtime provenance i ponownej kwalifikacji;
publiczne artefakty ani status `ACTIVE` nie są samodzielną authority.

`stage_9=IN_PROGRESS`, `production_provisioning_ready=false`,
`windows_production_ready=NOT_READY`. Fizyczne Windows/TPM gates: `NOT_RUN`.
`LEGAL_PRODUCTION_ENROLLMENT=NOT_PERFORMED`. Membership, provisioning operations,
account genesis, freshness, Secret Resource oraz Stage 10 pozostają poza zakresem.
