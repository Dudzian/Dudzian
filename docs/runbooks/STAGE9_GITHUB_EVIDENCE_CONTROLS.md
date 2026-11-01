# Stage 9 — ochrona GitHub dla ścieżki dowodowej Linux

To procedura dla właściciela repozytorium, bez automatycznej zmiany ustawień.
PR #3233 nie tworzy ochrony gałęzi, kont reviewerów, sekretów ani environments.
Kwalifikacja pozostaje BLOCKED do ich niezależnego skonfigurowania i sprawdzenia.
Nie wykonywać fizycznej instalacji ani restartu w ramach tej procedury.

## Stan zastany i znaczenie ustawień

Odczyt publicznego REST API 1 listopada 2026 r. potwierdził dla `main`:
`protected=false`, SHA `393113f4490aff42a74682823d84c8fcf92901cc`.
Lista rulesets była pusta, a `.github/CODEOWNERS` nie istniał.
Odczyt szczegółów branch protection przez użyty konektor zwrócił 403;
odczyt environments nie był obsługiwany przez ten konektor. Nie oznacza to,
że environments istnieją lub spełniają wymagania.

GitHub przy ustawieniu **Protected branches only** dopuszcza wszystkie gałęzie,
jeżeli żadna gałąź nie ma reguły ochrony. Dlatego sam
`deployment_branch_policy.protected_branches=true` nie jest dowodem ochrony.
`require-environment` osobno odczytuje rzeczywiste gałęzie i ich reguły.
[Dokumentacja GitHub o environments](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments)
opisuje również możliwość zapobiegania self-review i wyłączenia obejścia
reguł przez administratora.

## 1. Właściciele i bootstrap ochrony

Właściciel najpierw zatwierdza co najmniej dwa odrębne konta ludzi z jawnym
dostępem `write`, `maintain` albo `admin`. Nie wpisywać nieistniejących kont
ani nazw przykładowych jako rzeczywistych właścicieli. Używany profil wymaga
`.github/CODEOWNERS` z jedną globalną regułą, np. po zastąpieniu nazw:

```text
# Konta muszą istnieć i mieć zweryfikowany dostęp do zapisu.
* @ZATWIERDZONY-REVIEWER-1 @ZATWIERDZONY-REVIEWER-2
```

Ta reguła chroni również sam CODEOWNERS, workflow i politykę zaufania.
Walidator celowo odmawia dla wzorców wyjątków, późniejszych nadpisań reguł,
emaili i zespołów o niepotwierdzonym członkostwie. Nie udaje pełnego parsera
gitignore. Rozszerzenie obsługi wymaga osobnych testów i przeglądu.
[GitHub wymaga rzeczywistego dostępu write dla właścicieli kodu](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners).

Bootstrap z obecnego niechronionego `main` jest odrębną operacją właściciela:
przejrzeć stan, zatwierdzić ochronę gałęzi, a CODEOWNERS dostarczyć w
niezależnie przejrzanym PR. Samo dodanie pliku w PR nie ustanawia ochrony
przed scaleniem. W tej sesji nie wykonano tego bootstrapu ani merge.

## 2. Ochrona main i osobnej gałęzi polityki

Właściciel konfiguruje klasyczną regułę branch protection dla **dokładnej**
gałęzi `main` oraz osobno `stage9-linux-evidence-policy`. Druga gałąź powstaje
wyłącznie po świadomej zgodzie właściciela, z zatwierdzonego `main`.
W obu regułach wymagane są:

- Co najmniej dwa zatwierdzenia PR, wymagani code owners, odrzucanie starych
  zatwierdzeń i zgoda innej osoby na ostatni push.
- Egzekwowanie reguł także wobec administratorów; brak użytkowników, zespołów
  i aplikacji omijających wymagane review.
- Zakaz force push i usuwania gałęzi; wymagane rozwiązanie dyskusji.
- Strict required status checks, powiązane z konkretnym dodatnim `app_id`:
  `Ruff format (all tracked files)`, `Quality ratchet (combined coverage gate)`
  i `linux-evidence-contract`. Zachować także wszystkie wcześniejsze, mocniejsze
  kontrole repozytorium. Nie wpisywać fizycznego production gate jako kontroli,
  która miałaby otrzymać PASS z syntetycznych testów PR.

Rzeczywiste `app_id` odczytać z zakończonych check runs. Trzy wskazane wyżej
kontrole wymagają aplikacji GitHub Actions (`app_id=15368` na api.github.com),
nie dowolnej aplikacji o dodatnim ID. Nie wybierać
„any source”, nie używać `-1`, nie zastępować kontroli pustym statusem.
Wersja walidatora obsługuje powyższy profil klasycznej branch protection.
Sam ruleset lub nieczytelna konfiguracja pozostają BLOCKED, dopóki nie ma
osobno zweryfikowanej obsługi takiego profilu.
[Znaczenie branch protection i uprawnień API](https://docs.github.com/en/rest/branches/branch-protection#get-branch-protection).

Właściciel może sprawdzić ustawienia poleceniami tylko do odczytu:

```sh
gh api repos/Dudzian/Dudzian/branches/main
gh api repos/Dudzian/Dudzian/branches/main/protection
gh api repos/Dudzian/Dudzian/branches/stage9-linux-evidence-policy
gh api repos/Dudzian/Dudzian/branches/stage9-linux-evidence-policy/protection
```

Nie usuwać wymagania kontroli dlatego, że konto lub token nie może jej odczytać.

## 3. Wersjonowana polityka bez cyklicznego SHA

Pełny plik polityki jest przechowywany jako `policy.json` na osobnej chronionej
gałęzi `stage9-linux-evidence-policy`. Zawiera zatwierdzony finalny SHA kodu,
instalację, materiały, publiczny klucz dedykowanego producenta, tożsamość hosta
i niezależne autorytety tożsamości wydania. Nie zawiera kluczy prywatnych,
haseł ani poświadczeń. Fixture TEST_ONLY nie stanowi zatwierdzenia produktu.

Rozdzielenie gałęzi jest konieczne: commit kodu nie może zawierać w swoim
własnym pliku polityki wymaganego SHA tego samego commitu. Najpierw znany jest
przejrzany SHA kodu, potem niezależny PR aktualizuje policy.json na gałęzi
polityki. Aktualizacja obejmuje cały kontrakt i materiały, nie samo przepisywanie
skrótów z otrzymanego bundle. Właściciele kodu zatwierdzają przeznaczenie klucza,
fizyczność hosta i autentyczność materiałów poza raportem producenta.

Po niezależnym zatwierdzeniu i ewentualnym scaleniu przez uprawnionych ludzi
zanotować pełny 40-znakowy SHA **commitu gałęzi polityki** oraz kanoniczny
SHA-256 obiektu JSON. Wyzwanie przypina kod i hash polityki, a workflow
pobiera policy.json z dokładnego commitu. Walidator dodatkowo wymaga,
aby aktualny HEAD gałęzi polityki nadal był równy pinowi: rotacja lub odwołanie
polityki unieważnia kontynuację wcześniejszej próby.

## 4. Dwa environments i niezależne zatwierdzanie

Właściciel tworzy `stage9-linux-challenge` i `stage9-linux-evidence`.
W obu włącza wymaganych reviewerów, **Prevent self-review**, **Protected
branches only** i wyłącza **Allow administrators to bypass configured
protection rules**. Reviewer jest jednoznacznie zidentyfikowanym kontem człowieka;
obecny profil odmawia dla zespołów o niezweryfikowanym członkostwie.
GitHub wymaga zgody jednego z podanych reviewerów, nie wszystkich jednocześnie.

Wyłączenie administrator bypass sprawdzane jest jako
`can_admins_bypass=false` w odpowiedzi odczytu środowiska. Brak pola,
nieznany schemat odpowiedzi albo brak dostępu do ustawień oznaczają BLOCKED.
Nie zastępować tych danych deklaracją w zmiennej środowiskowej.
[Procedura GitHub konfiguracji environments](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments).

W obu environments właściciel ustawia publiczne zmienne:

- `STAGE9_LINUX_EVIDENCE_POLICY_COMMIT`: dokładny SHA aktualnego HEAD gałęzi polityki.
- `STAGE9_LINUX_EVIDENCE_POLICY_SHA256`: niezależnie zatwierdzony kanoniczny digest policy.json.
- W `stage9-linux-challenge` także `STAGE9_LINUX_EVIDENCE_HOST_ID`.

Nie stosować dawnego `STAGE9_LINUX_EVIDENCE_POLICY_JSON` jako źródła polityki:
zmienna repozytorium nie ma procesu review pliku na chronionej gałęzi.
Zmiana pinów wymaga niezależnego sprawdzenia zatwierdzonego PR polityki.
Samo przestawienie zmiennej nie pozwala podać alternatywnych bajtów policy.json.

## 5. Odczyt kontroli bez uprawnień do ich zmiany

Automatyczny `GITHUB_TOKEN` z `contents:read` i `actions:read` służy transportowi
artefaktów. Nie ma on wymaganych uprawnień Administration:read do pełnego
branch protection. Właściciel osobno zatwierdza ograniczony token fine-grained
lub token instalacji aplikacji **wyłącznie do odczytu** repozytorium
`Dudzian/Dudzian`: Administration:read, Actions:read, Contents:read i wymagane
Metadata:read. Nie dodawać uprawnień zapisu. Ustawić termin ważności i rotację.

W obu chronionych environments token jest sekretem
`STAGE9_GITHUB_CONTROLS_READ_TOKEN`. Tylko kroki `require-environment` używają
go jako `GH_TOKEN`, już po aprobacie environment. Nie przekazywać go do Della,
policy.json, archive redirect ani artefaktów. Wartości nigdy nie drukować.
Token nie daje możliwości skonfigurowania lub „naprawienia” ochrony przez workflow.
Odmowa API 403/404 jest odmową kwalifikacji.

## 6. Odtwarzalny odczyt kontroli

Z chronionego, przejrzanego checkoutu i z osobno skonfigurowanym tokenem
odczytowym w `GH_TOKEN`, bez wypisywania jego wartości:

```sh
python3 -B -m deployment.linux_evidence_handoff require-environment \
  --repository Dudzian/Dudzian --name stage9-linux-challenge \
  --policy-commit ZATWIERDZONY_40_ZNAKOWY_SHA_POLITYKI
python3 -B -m deployment.linux_evidence_handoff require-environment \
  --repository Dudzian/Dudzian --name stage9-linux-evidence \
  --policy-commit ZATWIERDZONY_40_ZNAKOWY_SHA_POLITYKI
```

Weryfikacja nie zmienia ustawień ani nie kwalifikuje Linux produkcyjnie.
Zatwierdzenie planu fizycznego poprzedza pierwszą zgodę na wydanie challenge.
Niezależna analiza kompletu dowodów poprzedza zgodę po zakończeniu ingress.
Przed decyzją końcowej bramki kontrola ustawień jest wykonywana ponownie.
Zaufanie do właściciela repozytorium pozostaje jawne: właściciel może zmieniać
ustawienia GitHub. Ten workflow nie jest niezależną od właściciela atestacją
administracji GitHub i nie może gwarantować niezmienności ustawień po odczycie.

Nie wykonano żadnej zmiany konfiguracji GitHub ani Della w ramach tej remediacji.
`LINUX_PRODUCTION_READY=NO` pozostaje obowiązujące.
