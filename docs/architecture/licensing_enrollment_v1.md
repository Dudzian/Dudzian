# CryptoHunter Licensing / Enrollment V1

## Niezmienniki kontraktu

Aktywacja manualna offline pozostaje wspierana permanentnie. API online jest wyłącznie
warstwą automatyzacji/transportu, a nie następcą aktywacji offline. Oba transporty przyjmują
`CryptoHunterActivationRequestV1` i produkują dokładnie ten sam, kanoniczny
`PDSAEnrollmentPackageV1`; transport nie występuje w podpisywanym ładunku. Awaria serwera
nie unieważnia wcześniej zweryfikowanego lokalnego enrollmentu (**SERVER DOWN != LICENSE
INVALID**).

Request używa losowego kryptograficznie nonce (256 bitów), a `request_id` jest SHA-256
kanonicznego requestu bez samego `request_id`. Profil canonical JSON V1 ze Stage 9 sortuje
klucze, koduje UTF-8 bez odstępów, zabrania floatów oraz dużych nieprzenośnych integerów.
Parsery V1 odrzucają pola nieznane. Features są sortowaną listą bez duplikatów.
Zweryfikowane requesty, decyzje i enrollmenty przechowują kanoniczne bajty jako niemutowalne
źródło prawdy; właściwości dokumentów zwracają wyłącznie defensywne, głębokie projekcje.

Package wiąże podpisem Ed25519 2-z-3: pełny digest requestu, installation/device, nazwę i
digest public area K_PSA, EK/AK/evidence profile, dokładny ReleasePolicy digest/version,
entitlements oraz tożsamość zestawu PDSA. Moduł przyjmuje wyłącznie publiczną projekcję
K_PSA; nie tworzy i nie zapisuje prywatnego klucza TPM. TEST_ONLY signer odrzuca ścieżkę
production authority. Podpisy ani klucze production nie są tutaj dostępne.
`authority.signer_key_ids` wymienia dokładnie sygnatariuszy obecnych w `signature_block`, zaś
digest całego zaufanego zestawu 3 kluczy jest niezależnie rekonstruowany przez verifier.
`enrollment_id` jest zawsze SHA-256 z `request_id + ":" + license_id`, a
`issued_at_utc` musi być identyczne z `license.issued_at`.
`VerifiedEnrollmentV1` jest wygodnym niemutowalnym stanem, lecz nie jest kryptograficzną
capability persistence. Store przy każdym imporcie sam uruchamia skonfigurowany zaufany
verifier i dopiero po sukcesie zapisuje dokładnie zweryfikowane bajty. Profil PDSA jest
strukturalnie wymuszony jako dokładnie 2-z-3: wszystkie trzy unikalne klucze Ed25519 są
walidowane, także klucz nieużyty w konkretnym quorum.

## Startup, prywatność i ograniczenia

Store zapisuje jedynie podpisany publiczny package. Startup najpierw weryfikuje go lokalnie,
bez HTTP. Niedostępność online daje `ONLINE_SERVICE_UNAVAILABLE`: ważna instalacja działa
dalej, zaś nowa otrzymuje `OFFLINE_ACTIVATION_REQUIRED`. View-model udostępnia stany dla
„Activate Online”, „Generate Offline Activation Request” i „Import Offline License”:
`ACTIVATED`, `NOT_ACTIVATED`, `ONLINE_UNAVAILABLE`, `OFFLINE_ACTIVATION_REQUIRED`,
`LICENSE_INVALID`, `DEVICE_MISMATCH`, `EXPIRED`.

Request nie zbiera kont, nazw użytkownika, ścieżek, IP/MAC ani inwentarza. V1 może być
perpetual (`expires_at: null`) albo czasowy. Dla czasu expiry verifier wymaga wstrzykniętego
źródła zaufanej świeżości (docelowo istniejący FreshnessAuthority/TPM NV), zamiast ufać
`datetime.now()`.

Bez serwera i bez aktualizacji lokalnego stanu revocation natychmiastowe zdalne cofnięcie
jest niemożliwe. Przyszłe mechanizmy mogą obejmować podpisane listy cofnięć i policy updates,
expiry/renewal oraz freshness TPM; V1 nie udaje „instant revocation”. Formalny Stage 9 i
zamrożona ceremonia production pozostają niezmienione.
