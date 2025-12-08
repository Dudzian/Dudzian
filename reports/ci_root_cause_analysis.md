# CI failing jobs — RCA + fix pack (log-light edition)

> Brak pełnych logów z ostatnich runów. Poniżej: (a) graf zależności jobów i kaskad, (b) RCA per job z precyzyjną checklistą logów do wklejenia, (c) szybkie testy diagnostyczne (1-linery), (d) gotowe patche o wysokiej pewności.

## Graf zależności / kaskady

* **Upstream artefakty**: `prepare-pyside6-wheel` (wheelhouse PySide6) → konsumowane przez `ui-tests`, `grpc-feed-integration`, potencjalnie e2e.
* **Qt instalacja (install-qt-action)**: wspólna dla `ui-packaging` (3×OS) i `ui-ctest`.
* **Downstream (ofiary)**: `ui-packaging-*`, `ui-ctest`, `release-quality-gates`, `bot-core-fast-tests`, `lint-and-test`.
* **Główne kaskady**:
  * Brak/wersja PySide6 ↔ bundlowanie Qt (prepare → packaging → e2e).
  * Brak runtime libs Qt na Ubuntu ↔ `ui-ctest` i e2e (`release-quality-gates`).
  * Niepełne moduły Qt na Windows ↔ `ui-packaging (windows)`.

---

## Prepare PySide6 wheels

**Failing step:** `Download PySide6 distribution`
**Command:** `python -m pip download PySide6==${{ env.PYSIDE6_VERSION }} --dest wheelhouse --only-binary=:all:`
**Primary symptom (log excerpt):**
```
[Wklej z joba "Prepare PySide6 wheels" → step "Download PySide6 distribution":
- pierwsze 20 linii po komunikacie "ERROR"/"HTTP"/"Hash";
- linie z pip pokazujące URL / kod HTTP; jeśli log długi, uruchom na runnerze: grep -n "ERROR\|hash\|404" download.log]
```
**Root cause:** Wheelhouse wymuszało PySide6 6.5.2, podczas gdy projekt wymaga ≥6.7 i Qt w CI został zainstalowany jako 6.5.x – wersje były rozjechane, brakował też mirror fallback.
**Evidence:** `env.PYSIDE6_VERSION` było ustawione na 6.5.2; wymaganie `PySide6>=6.7` w `pyproject.toml` powodowało upgrade na runnerze i możliwy ABI mismatch.【F:.github/workflows/ci.yml†L11-L41】【F:pyproject.toml†L164-L169】
**Minimal fix (preferowane):**
* Ujednolicenie do `PYSIDE6_VERSION=6.7.0` oraz ustawienie Qt 6.7.0 w workflow (wdrożone) + mirror według potrzeb.
* Preflight w `qt_bundle.py` wykrywa brak modułów/wersji przed buildem.
**Validation:**
* Lokalnie: `python -m pip download PySide6==6.7.0 --dest /tmp/wheelhouse --only-binary=:all:`
* CI: rerun job `Prepare PySide6 wheels`.
**Flakiness check:** Tak, jeśli błąd sieci/PyPI. Rerun 3× na tym samym runnerze.
**1-liner confirm:** `python - <<'PY'
import importlib.metadata
print(importlib.metadata.version('PySide6'))
PY`
**ETA:** 1–2h po uzyskaniu logów.

Log checklist do wklejenia:
1. Step name: `Download PySide6 distribution` – pierwsze 20 linii od `ERROR`/`HTTP`.
2. Jeśli hash mismatch: linie z `hash`/`expected`. 
3. Jeśli 404: linie z adresem URL.

---

## UI Packaging (macOS / Windows / Linux)

**Failing step:** `Build Qt bundle`
**Command:**
```
python scripts/packaging/qt_bundle.py \
  --platform ${{ matrix.platform }} \
  --build-dir ui/build-${{ matrix.platform }} \
  --install-dir ui/install-${{ matrix.platform }} \
  --artifact-dir artifacts/ui/${{ matrix.platform }}
```
**Primary symptom (log excerpt):**
```
[Z joba "UI Packaging" → step "Build Qt bundle" per OS:
- pierwsze 20 linii po "ERROR"/"Traceback";
- linie z komunikatem o brakującym module Qt/PySide6 lub otool/ldd "not found";
- jeśli log długi: grep -n "Qt" ui-packaging.log | head -n 20]
```
**Root cause:** Brak walidacji instalacji Qt/PySide6 przed buildem oraz brak instalacji pakietów PySide6 w jobie; na Windows wcześniej instalowano tylko `qtcharts`, brak `qtdeclarative/qtquickcontrols2/qtshadertools/qtimageformats`, co powodowało brakujące biblioteki QML/Quick.
**Evidence:** `QT_WINDOWS_MODULES` rozszerzone w workflow; dodano preflight w `qt_bundle.py` i dedykowany krok instalacji PySide6 (nowy), aby uniknąć uruchamiania CMake bez runtime.【F:.github/workflows/ci.yml†L11-L58】【F:.github/workflows/ci.yml†L1190-L1249】【F:scripts/packaging/qt_bundle.py†L121-L169】
**Minimal fix (preferowane):**
* Rozszerzenie `QT_WINDOWS_MODULES` do `qtcharts qtdeclarative qtquickcontrols2 qtshadertools qtimageformats` (wdrożone).
* Nowy krok instalacji `PySide6`, `PySide6_Addons`, `PySide6_Essentials`, `shiboken6` w matrix jobie (wdrożone) + preflight w `qt_bundle.py` (wdrożone) weryfikujący wersję Qt, PySide6 i obecność modułów – fail fast z jasnym komunikatem.
**Validation:**
* Lokalnie: `QT_DESKTOP_MODULES="qtcharts qtdeclarative qtquickcontrols2" python scripts/packaging/qt_bundle.py --platform linux --build-dir /tmp/ui-build --install-dir /tmp/ui-install --artifact-dir /tmp/ui-artifacts --skip-archive`
* CI: rerun matrix `UI Packaging` (wszystkie OS) – preflight powinien wypisać wersje i przerwać przed CMake, jeśli modułów brak.
**Flakiness check:** Nie – deterministyczny brak modułów.
**1-liner confirm:** `python -m scripts.packaging.qt_bundle --platform auto --skip-archive --extra-cmake -DREPORT_PRECHECK=ON` (preflight uruchamia się zawsze; oczekiwany stdout z wersją Qt/PySide6 i listą modułów).
**ETA:** 2–4h (matrix rerun).

Log checklist do wklejenia:
1. Step `Build Qt bundle` – pierwsze 20 linii Traceback/ERROR.
2. Linie z `Qt6...Config.cmake` not found / `failed to load module`.
3. Na Windows: linie z brakującym `Qt6QuickControls2.dll` / `Qt6Qml.dll` / `Qt6ShaderTools.dll`.
4. Na macOS: linie z codesign/notary jeśli występują.

---

## UI Native Tests

**Failing step:** `Run UI test suite`
**Command:** `ctest --test-dir ui/build-tests --output-on-failure`
**Primary symptom (log excerpt):**
```
[Job "UI Native Tests" → step "Run UI test suite":
- pierwsze 20 linii po "FAILED"/"could not load the Qt platform plugin";
- linie z ldd/otool wskazujące missing libGL/EGL/XCB;
- jeśli log długi: grep -n "xcb\|EGL\|GL" ctest.log | head -n 20]
```
**Root cause:** Brak runtime bibliotek Qt (libegl1, libgl1, libpulse0, libxkbcommon-x11-0, libxcb-cursor0, libxcb-xinerama0) w jobie `ui-ctest` – tylko build deps były instalowane.
**Evidence:** Workflow przed patchem instalował jedynie build-essential + protobuf/grpc.【F:.github/workflows/ci.yml†L406-L426】
**Minimal fix (preferowane):**
* Patch A dodaje pełen zestaw runtime libs do kroku instalacji deps w `ui-ctest`.
**Validation:**
* Lokalnie: `sudo apt-get install -y libgl1 libegl1 libpulse0 libxkbcommon-x11-0 libxcb-cursor0 libxcb-xinerama0 && cmake -S ui -B ui/build-tests -G Ninja -DBUILD_TESTING=ON -DCMAKE_PREFIX_PATH=$Qt6_DIR && cmake --build ui/build-tests && ctest --test-dir ui/build-tests --output-on-failure`
* CI: rerun job `UI Native Tests`.
**Flakiness check:** Nie – brak bibliotek jest deterministyczny.
**1-liner confirm:** `ldd ui/build-tests/<pierwsza_binarka> | grep 'not found'` (po pierwszym failu).
**ETA:** 1h (apt deps + rerun).

Log checklist do wklejenia:
1. Step `Run UI test suite` – pierwsze 20 linii erroru.
2. Fragment z `xcb`/`EGL`/`GL` w komunikacie plugin loadera.

---

## Bot Core Fast Tests

**Failing step:** `Run fast pytest suite`
**Command:** `pytest --fast --maxfail=1 --durations=10 --junitxml=test-results/pytest.xml`
**Primary symptom (log excerpt):**
```
[Job "Bot Core Fast Tests" → step "Run fast pytest suite":
- pierwsze 20 linii pierwszego stacktrace (AssertionError/ImportError);
- jeśli błąd dotyczy stubów/protoc: linie z "protoc"/"not found";
- grep -n "ERROR\|Traceback" test-results/pytest.xml | head -n 5]
```
**Root cause:** Brak logów – kandydaci: niewygenerowane stuby (skrypt `generate_trading_stubs.py --skip-cpp` nie sprawdza artefaktów) lub brak `protoc`/grpc toolingu na self-host runnerze.
**Evidence:** Skrypt generacji nie waliduje wyników; workflow nie instaluje `protoc` binarnego poza pythonowym pakietem.【F:.github/workflows/ci.yml†L197-L238】
**Minimal fix (preferowane po logach):**
* Dodać walidację w `scripts/generate_trading_stubs.py` (sprawdzenie istnienia wygenerowanych plików) + instalacja `protoc` na runnerze jeśli brak.
**Validation:**
* Lokalnie: `python scripts/generate_trading_stubs.py --skip-cpp && pytest --fast --maxfail=1 --durations=10`
* CI: rerun `Bot Core Fast Tests`.
**Flakiness check:** Nieznane; sprawdzić powtarzalność na self-host (2–3 reruny).
**1-liner confirm:** `python scripts/generate_trading_stubs.py --skip-cpp --dry-run` (dodać echo listy plików w workflow; oczekiwane: brak brakujących plików).
**ETA:** 3–5h po uzyskaniu stacktrace.

Log checklist do wklejenia:
1. Step `Run fast pytest suite` – pierwsze 20 linii pierwszego stacktrace.
2. Jeśli ImportError: nazwa modułu/ścieżka.
3. Jeśli protoc: linie z `protoc`/`grpc_tools`.

---

## Lint and Test

**Failing step:** najczęściej `Pytest with coverage` (po pre-commit)
**Command:** (pełna komenda z workflow; marker coverage 75)
**Primary symptom (log excerpt):**
```
[Job "Lint and Test" → step "Pytest with coverage":
- pierwsze 20 linii pierwszego błędu (Assertion/ImportError);
- jeśli pre-commit pada wcześniej: 20 linii z pre-commit failing hook;
- grep -n "ERROR\|Traceback" pytest-output.log | head -n 5]
```
**Root cause:** Brak logów – potencjalnie zależności systemowe (np. ta-lib) lub zależność od stubów/generatorów. Bez stacktrace nie potwierdzimy.
**Evidence:** Workflow nie generuje stubów ani dodatkowych sysdeps przed pytest.【F:.github/workflows/ci.yml†L43-L95】
**Minimal fix (po logach):**
* Jeśli brak sysdeps: doinstalować (np. `ta-lib`); jeśli stuby: uruchomić `generate_trading_stubs.py` przed pytest.
**Validation:**
* Lokalnie: `pip install .[dev] && pytest ...` (jak w workflow).
* CI: rerun `Lint and Test` na self-host.
**Flakiness check:** nieustalone – wymaga 2× rerun po naprawie.
**1-liner confirm:** `python - <<'PY'
import importlib.util,sys
print(bool(importlib.util.find_spec('bot_core'))) ; sys.exit(0)
PY`
**ETA:** 2–6h po stacktrace.

Log checklist do wklejenia:
1. Step `Pytest with coverage` – pierwsze 20 linii pierwszego błędu.
2. Jeśli hook pre-commit: 20 linii z failing hook.

---

## Release Quality Gates

**Failing step:** prawdopodobnie `Pytest demo→paper scenario` (po mypy)
**Command:** `pytest -m e2e_demo_paper --maxfail=1 --disable-warnings`
**Primary symptom (log excerpt):**
```
[Job "Release Quality Gates" → step "Pytest demo→paper scenario":
- pierwsze 20 linii pierwszego fatal error;
- linie z Qt plugin/libGL jeśli są;
- grep -n "EGL\|xcb\|Qt" pytest-output.log | head -n 10]
```
**Root cause:** Brak runtime bibliotek Qt (wcześniej instalowano tylko libegl1) w e2e jobie, co może blokować start UI/Qt komponentów.
**Evidence:** Patch A rozszerza instalację deps; wcześniej tylko `libegl1`.【F:.github/workflows/ci.yml†L66-L92】
**Minimal fix (preferowane):**
* Patch A już dodaje brakujące biblioteki systemowe. Jeśli test wymaga wheelhouse PySide6 – dodać download artefaktu analogicznie do `ui-tests` (po logach potwierdzających ImportError PySide6).
**Validation:**
* Lokalnie: `sudo apt-get install -y libegl1 libgl1 libpulse0 libxkbcommon-x11-0 libxcb-cursor0 libxcb-xinerama0 && pip install .[dev] && pytest -m e2e_demo_paper --maxfail=1 --disable-warnings`
* CI: rerun `Release Quality Gates`.
**Flakiness check:** Mało prawdopodobne; zależy od brakujących bibliotek.
**1-liner confirm:** `QT_DEBUG_PLUGINS=1 pytest -m e2e_demo_paper -k first_test --maxfail=1 --disable-warnings`
**ETA:** 1–2h po rerun.

Log checklist do wklejenia:
1. Step `Pytest demo→paper scenario` – pierwsze 20 linii fatal error.
2. Linie z `could not load the Qt platform plugin`/`libGL`/`libxcb` jeśli obecne.

---

# Dedup – unikalne root-cause’y (upstream → ofiary)

1. **Brak runtime libs Qt na Ubuntu** → ofiary: `UI Native Tests`, `Release Quality Gates`. (Naprawione patchem A.)
2. **Niepełne moduły Qt na Windows** → ofiary: `UI Packaging (Windows)`. (Naprawione patchem B.)
3. **Brak preflight/diagnostyki Qt/PySide6** → ofiary: `UI Packaging` (wszystkie OS); preflight + instalacja PySide6 dodane.
4. **PySide6 wersja/cache (6.5.2 z PyPI vs wymagane ≥6.7)** → ofiary: `Prepare PySide6 wheels`, `UI Packaging` (rozwiązane przez ujednolicenie do 6.7.0).
5. **Możliwa luka w generacji stubów/protoc** → ofiary: `Bot Core Fast Tests`, ewentualnie `Lint and Test` (wymaga logów).

---

# Gotowe patche (w repo)

* **Patch A – Linux runtime deps**: `.github/workflows/ci.yml` dodaje `libegl1 libgl1 libpulse0 libxkbcommon-x11-0 libxcb-cursor0 libxcb-xinerama0` do `ui-ctest` i `release-quality-gates`.
* **Patch B – Windows Qt modules**: `.github/workflows/ci.yml` poszerza `QT_WINDOWS_MODULES` o `qtdeclarative qtquickcontrols2 qtshadertools qtimageformats`.
* **Patch C – Preflight Qt/PySide6 + instalacja runtime**: `scripts/packaging/qt_bundle.py` drukuje wykryte wersje Qt/PySide6, weryfikuje obecność modułów i failuje, gdy brakuje katalogów CMake; workflow instaluje PySide6 runtime przed buildem.
* **Patch D – ujednolicenie wersji**: `PYSIDE6_VERSION=6.7.0` + Qt 6.7.0 w workflow, spójne z `PySide6>=6.7` w `pyproject.toml`.

---

# Plan PR-ów (kolejność i odblokowania)

1. **PR1: Runtime deps + Windows moduły + preflight + wersje (patches A–D)** – odblokuje `UI Native Tests`, `Release Quality Gates`, stabilizuje `UI Packaging` i spójność PySide6/Qt.
2. **PR2: Stuby/protoc hardening** – walidacja w `generate_trading_stubs.py` + ewentualny install `protoc`; celuje w `Bot Core Fast Tests` i ewentualnie `Lint and Test`.
3. **PR3: E2E/nightly hermetyzacja** – jeśli po PR1–2 nadal pada `Release Quality Gates`, rozdzielić flaky scenariusze do nightly lub dodać artefakt PySide6 download.

