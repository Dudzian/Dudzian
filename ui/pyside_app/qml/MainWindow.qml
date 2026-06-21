import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Effects
import "components" as Components
import "components/layout" as LayoutComponents
import Styles 1.0 as StylesModule
import "views" as Views

ApplicationWindow {
    id: root
    width: 1280
    height: 720
    visible: true
    title: qsTr("Dudzian Product Preview — safe dry-run")
    color: designSystem.color("background")
    property var contextGrpcBridge: (typeof grpcBridge !== "undefined" ? grpcBridge : null)
    property var runtimeService: contextGrpcBridge && contextGrpcBridge.runtimeService ? contextGrpcBridge.runtimeService : null
    property var contextRuntimeState: (typeof runtimeState !== "undefined" ? runtimeState : null)
    property var actionDispatchContextBridge: (typeof paperRuntimeActionDispatchBridge !== "undefined" ? paperRuntimeActionDispatchBridge : null)
    property string defaultPanelId: "sidePanel"
    property string currentPanelId: defaultPanelId
    property string currentLanguage: "PL"
    property var languageOptions: [
        ({ code: "PL", label: "Polski", flag: "🇵🇱", display: "🇵🇱 PL" }),
        ({ code: "EN", label: "English", flag: "🌐", display: "🌐 EN" })
    ]
    property var translationDictionary: ({
        "PL": ({
            "app.title": "Dudzian Bot Preview",
            "safety.summary": "Live trading wyłączony • Połączenie z giełdą wyłączone • Składanie zleceń wyłączone",
            "nav.dashboard": "Dashboard",
            "nav.aiCenter": "AI Center",
            "nav.universe": "Trading Universe",
            "nav.marketScanner": "Okazje / Market Scanner",
            "nav.portfolio": "Portfel / Wyniki",
            "nav.terminal": "Paper Terminal",
            "nav.strategies": "Strategie",
            "nav.risk": "Ryzyko",
            "nav.decisions": "Decyzje",
            "nav.telemetry": "Telemetria",
            "nav.alerts": "Alerty",
            "nav.diagnostics": "Diagnostyka",
            "nav.settings": "Ustawienia",
            "nav.runtimeControl": "Runtime / Sesja",
            "nav.help": "Pomoc / Słownik",
            "settings.title": "Ustawienia",
            "settings.description": "Ustawienia działają lokalnie w preview. Konfiguracja runtime nie jest zapisywana.",
            "onboarding.title": "Onboarding preview",
            "quick.actions": "Szybkie akcje",
            "app.status": "Status aplikacji",
            "language.label": "Język",
            "refresh.preview": "Odśwież preview",
            "help.title": "Pomoc / Słownik",
            "help.description": "Prosty słowniczek pojęć dla nietechnicznego użytkownika. Wszystko działa lokalnie w Paper Preview.",
            "category.trading": "Trading",
            "category.risk": "Ryzyko",
            "category.ai": "AI / Governor",
            "category.strategies": "Strategie",
            "category.paper": "Paper / Live",
            "category.exchange": "Giełdy / API",
            "category.diagnostics": "Diagnostyka"
        }),
        "EN": ({
            "app.title": "Dudzian Bot Preview",
            "safety.summary": "Live trading disabled • Exchange I/O disabled • Order submission disabled",
            "nav.dashboard": "Dashboard",
            "nav.aiCenter": "AI Center",
            "nav.universe": "Trading Universe",
            "nav.marketScanner": "Market Scanner",
            "nav.portfolio": "Portfolio / Results",
            "nav.terminal": "Paper Terminal",
            "nav.strategies": "Strategies",
            "nav.risk": "Risk",
            "nav.decisions": "Decisions",
            "nav.telemetry": "Telemetry",
            "nav.alerts": "Alerts",
            "nav.diagnostics": "Diagnostics",
            "nav.settings": "Settings",
            "nav.runtimeControl": "Runtime / Session",
            "nav.help": "Help / Glossary",
            "settings.title": "Settings",
            "settings.description": "Settings are local preview only. No runtime config is written.",
            "onboarding.title": "Onboarding preview",
            "quick.actions": "Quick actions",
            "app.status": "App status",
            "language.label": "Language",
            "refresh.preview": "Refresh preview",
            "help.title": "Help / Glossary",
            "help.description": "A simple glossary for non-technical users. Everything stays local in Paper Preview.",
            "category.trading": "Trading",
            "category.risk": "Risk",
            "category.ai": "AI / Governor",
            "category.strategies": "Strategies",
            "category.paper": "Paper / Live",
            "category.exchange": "Exchanges / API",
            "category.diagnostics": "Diagnostics"
        })
    })
    property var previewTooltips: ({
        "Start Paper Preview": "Uruchamia lokalną sesję testową Paper Preview. Nie składa prawdziwych zleceń.",
        "Pause": "Wstrzymuje lokalne ticki bez kasowania wyników sesji.",
        "Stop": "Zatrzymuje lokalną sesję testową. Runtime produkcyjny nadal nie startuje.",
        "Reset": "Czyści lokalny stan sesji paper i zostawia blokady bezpieczeństwa aktywne.",
        "Generate Next Tick": "Dodaje jeden przykładowy krok rynku i decyzji tylko w pamięci UI.",
        "Run 10 paper ticks": "Szybko generuje 10 lokalnych kroków paper bez kontaktu z giełdą.",
        "Generate governor recommendation": "Tworzy przykładową rekomendację AI/Governor w UI. To nie jest sygnał live.",
        "Import markets preview": "Ładuje lokalny katalog par do podglądu, bez pobierania danych z giełdy.",
        "Select top 20": "Zaznacza 20 lokalnych kandydatów z listy preview.",
        "Blacklist selected": "Dodaje zaznaczone pary do lokalnej listy blokad.",
        "Whitelist selected": "Dodaje zaznaczone pary do lokalnej listy dozwolonych par.",
        "Simulate buy/sell order": "Dopisuje tylko lokalną symulację kupna lub sprzedaży. Brak prawdziwego zlecenia.",
        "Risk profile Conservative": "Najostrożniejsze limity: mniejsze pozycje i ciaśniejsze ryzyko.",
        "Risk profile Balanced": "Środkowy profil ryzyka dla zbalansowanego Paper Preview.",
        "Risk profile Aggressive": "Wyższe limity preview; nadal bez live tradingu i prawdziwych zleceń.",
        "Custom risk": "Własny profil: użytkownik ręcznie ustawia limity ryzyka. Zmiany są local-only preview.",
        "AI recommended risk": "AI dobiera lokalnie ostrożne limity na podstawie scenariusza, readiness, coverage, PnL i drawdown. Brak inferencji backendowej.",
        "Risk profile Custom": "Własny profil ryzyka: edytowalne limity tylko w pamięci UI preview.",
        "Risk profile AI Recommended": "AI dobiera lokalnie limity i pokazuje nietechniczne uzasadnienie; nie używa API ani modelu backendowego.",
        "max position": "Największa pojedyncza pozycja dopuszczona w lokalnym preview.",
        "stop loss": "Preview poziomu ograniczenia straty dla pozycji.",
        "take profit": "Preview poziomu realizacji zysku dla pozycji.",
        "slippage": "Maksymalna różnica ceny akceptowana w lokalnej symulacji.",
        "drawdown": "Maksymalny spadek kapitału tolerowany przez lokalny risk guard.",
        "daily loss limit": "Dzienny limit straty w Paper Preview.",
        "confidence floor": "Minimalna pewność lokalnej decyzji, zanim tick pokaże PAPER BUY/SELL zamiast HOLD/NO ORDER.",
        "kill-switch": "Awaryjna blokada: tick generuje tylko BLOCKED local preview event.",
        "allow AI override": "Preview przełącznik pozwalający AI nadpisać część własnych limitów; bez runtime i bez zleceń.",
        "Custom range": "Zakres raportu portfela. Nie zmienia aktywnej sesji Paper.",
        "Zastosuj zakres": "Stosuje zakres tylko do raportu Portfel / Wyniki, bez mutacji paper state.",
        "Live-like paper simulation": "Lokalna pętla paper, która wygląda jak live scanning, ale nie używa giełdy, sekretów ani prawdziwych zleceń.",
        "Simulation speed": "Szybkość lokalnego timera QML; zmienia tylko interwał preview.",
        "Market scenario": "Lokalny scenariusz rynku dla mock cen: balanced, bull, bear, volatility albo range.",
        "Paper loop": "Timer QML mieli ticki, decyzje, paper ordery, PnL i telemetrię tylko w pamięci UI.",
        "No real orders": "Żadne kliknięcie w preview nie składa prawdziwego zlecenia ani nie uruchamia runtime produkcyjnego.",
        "Start scanner": "Uruchamia lokalny skaner okazji preview; nie łączy się z giełdą.",
        "Pause scanner": "Wstrzymuje lokalny skaner bez kasowania kandydatów.",
        "Run scan tick": "Wykonuje jeden deterministyczny scan tick na lokalnym katalogu par.",
        "Run scan burst": "Wykonuje serię lokalnych ticków skanera bez realnych połączeń API.",
        "AI score": "Ocena lokalnego kandydata: trend, płynność i zgodność strategii.",
        "Risk score": "Niższa wartość oznacza bezpieczniejszy setup w preview.",
        "Liquidity score": "Lokalna ocena czy para ma wystarczający wolumen preview.",
        "Spread": "Preview różnicy bid/ask; szeroki spread obniża ocenę.",
        "Volatility": "Lokalna zmienność scenariusza, wpływa na ryzyko.",
        "Strategy match": "Strategia preview, która najlepiej pasuje do setupu.",
        "Watchlist": "Lokalna lista obserwowanych par w tym ekranie.",
        "Blacklist": "Lokalna blokada pary w skanerze preview.",
        "Explain candidate": "Pokazuje nietechniczne uzasadnienie wyboru lub odrzucenia pary.",
        "Explain decision": "Otwiera lokalny panel wyjaśnienia decyzji AI/Governora; bez backendowej inferencji i bez zleceń.",
        "Audit trail": "Czytelna oś kroków: snapshot danych, score, risk guard, paper decision i granice bezpieczeństwa.",
        "Risk checks": "Lista lokalnych kontroli ryzyka: profil, kill-switch, risk lock, limity i trasa zlecenia.",
        "Input snapshot": "Lokalny podgląd danych wejściowych użytych przez preview do deterministycznego wyjaśnienia.",
        "Paper impact": "Opisuje, czy preview zmieniłoby PnL, equity lub pozycję; nigdy nie składa realnego zlecenia.",
        "Mark all read": "Oznacza wszystkie alerty jako przeczytane tylko w lokalnym stanie preview.",
        "Clear alerts": "Czyści lokalną oś alertów w pamięci UI; nie dotyka backendu ani giełdy.",
        "Mute alerts": "Wycisza alerty wyłącznie jako przełącznik preview; nie używa systemowych powiadomień OS.",
        "Sound preview": "Włącza/wyłącza wyłącznie flagę dźwięku preview; żaden dźwięk systemowy nie jest odtwarzany.",
        "Desktop notification preview": "Włącza/wyłącza tylko lokalną flagę powiadomień desktop preview; brak systemowych powiadomień OS.",
        "Explain event": "Buduje lokalne wyjaśnienie zdarzenia bez backend inference, sieci, API i sekretów.",
        "Severity filter": "Filtruje lokalną oś alertów według Critical, Warning albo Info.",
        "Category filter": "Filtruje lokalną oś alertów według Trading, Risk, AI, Scanner, Paper, Portfolio, Telemetry, Diagnostics albo Safety.",
        "Alert Center": "Otwiera lokalne centrum alertów i oś zdarzeń Paper Preview.",
        "Settings": "Otwiera lokalne ustawienia preview; konfiguracja runtime nie jest zapisywana.",
        "Onboarding": "Prowadzi przez lokalny first-run preview wizard bez backendu i bez live tradingu.",
        "Demo Preview": "Tryb demonstracyjny safe preview: wszystkie akcje pozostają lokalne.",
        "Paper Preview": "Tryb paper trading preview: symulowane ticki i paper ordery bez giełdy.",
        "Sandbox planned": "Planowany sandbox/testnet; w tym preview nadal brak połączeń giełda/API.",
        "Live disabled": "Tryb live pozostaje niedostępny i zablokowany w UI Preview.",
        "Base currency": "Waluta bazowa do lokalnego formatowania podglądu; nie zmienia runtime.",
        "UI density": "Kompaktowość UI wyłącznie w lokalnym stanie preview.",
        "App mode": "Wybór trybu aplikacji preview; live trading pozostaje wyłączony.",
        "Local preview state": "Stan zapamiętany tylko w QML preview podczas sesji UI.",
        "Reset local preview state": "Czyści lokalne ustawienia i paper state bez sekretów, backendu i runtime.",
        "App mode selector": "Wybiera Demo Preview, Paper Preview, Sandbox planned albo Live disabled tylko lokalnie.",
        "Base currency selector": "Wybiera PLN, USD, EUR albo USDT tylko dla preview.",
        "UI density selector": "Wybiera Compact, Comfortable albo Large tylko dla UI preview.",
        "Theme preview": "Dark preview działa teraz; Light planned jest tylko zapowiedzią.",
        "Apply preview settings": "Zastosuj ustawienia tylko lokalnie; no runtime config is written.",
        "Start onboarding": "Uruchamia local-only onboarding preview wizard.",
        "Complete onboarding": "Kończy onboarding lokalnie i przechodzi do Dashboardu.",
        "Open Settings": "Otwiera panel Settings/Ustawienia.",
        "Open Alerts": "Otwiera lokalne centrum alertów.",
        "Open Help": "Otwiera Pomoc / Słownik.",
        "Generate diagnostic bundle": "Generuje lokalny status paczki diagnostycznej bez sekretów, backendu, giełdy i runtime."
    })
    property var glossaryCategories: [
        ({ key: "category.trading", terms: ["PnL", "ROI", "spread", "order book", "fee/prowizja", "equity", "available balance", "in positions"] }),
        ({ key: "category.risk", terms: ["drawdown", "risk guard", "kill-switch", "TP", "SL", "Custom risk", "AI Recommended risk", "confidence floor", "exposure", "daily loss limit", "cooldown", "risk override"] }),
        ({ key: "category.ai", terms: ["governor", "confidence", "Market Scanner", "AI score", "Candidate", "Rejected setup", "explainability", "audit trail", "lineage", "input snapshot", "risk check", "decision source", "alternative candidate", "paper impact"] }),
        ({ key: "category.strategies", terms: ["strategy", "Strategy match", "Trend", "Volatility", "Liquidity", "Risk score"] }),
        ({ key: "category.paper", terms: ["paper trading", "sandbox/testnet", "Demo Preview", "Paper Preview", "Sandbox planned", "Live disabled", "Live-like paper simulation", "Simulation speed", "Market scenario", "Paper loop", "No real orders", "Onboarding"] }),
        ({ key: "category.exchange", terms: ["API key", "Base currency", "App mode", "UI density", "Local preview state", "Reset local preview state", "slippage", "blacklist", "whitelist", "Watchlist", "Blacklist"] }),
        ({ key: "category.diagnostics", terms: ["Settings", "Runtime loop not started", "Exchange I/O disabled", "Order submission disabled"] })
    ]
    property var glossaryDescriptions: ({
        "PL": ({
            "PnL": "Zysk lub strata z transakcji albo sesji.",
            "ROI": "Procentowy wynik względem użytego kapitału.",
            "drawdown": "Spadek kapitału od ostatniego szczytu.",
            "slippage": "Różnica między oczekiwaną ceną a ceną wykonania.",
            "spread": "Różnica między najlepszą ceną kupna i sprzedaży.",
            "order book": "Lista ofert kupna i sprzedaży widoczna na rynku.",
            "paper trading": "Handel testowy na danych preview, bez prawdziwych pieniędzy.",
            "sandbox/testnet": "Bezpieczne środowisko testowe giełdy lub aplikacji.",
            "API key": "Klucz dostępu do giełdy; w tym preview nie jest wymagany.",
            "governor": "Warstwa AI, która podpowiada lub blokuje decyzje według zasad ryzyka.",
            "confidence": "Poziom pewności modelu dla danej rekomendacji.",
            "confidence floor": "Minimalny próg pewności wymagany, aby lokalna decyzja preview mogła pokazać PAPER BUY/SELL.",
            "Custom risk": "Własny profil ryzyka z limitami edytowanymi wyłącznie w lokalnym UI preview.",
            "AI Recommended risk": "Lokalny profil, w którym preview AI dobiera limity bez połączeń z backendem lub giełdą.",
            "exposure": "Część kapitału zaangażowana w pozycję, symbol lub strategię.",
            "daily loss limit": "Maksymalna dzienna strata dopuszczona w lokalnej symulacji paper.",
            "cooldown": "Czas przerwy po decyzji lub blokadzie, zanim preview pokaże kolejną agresywniejszą akcję.",
            "risk override": "Nadpisanie limitu ryzyka; w tym UI jest tylko bezpiecznym przełącznikiem preview.",
            "strategy": "Zestaw reguł mówiących, kiedy obserwować, kupić, sprzedać lub czekać.",
            "risk guard": "Bezpiecznik ograniczający zbyt ryzykowne działania.",
            "kill-switch": "Awaryjny wyłącznik blokujący handel lub akcje systemu.",
            "TP": "Take Profit: poziom planowanego zamknięcia z zyskiem.",
            "SL": "Stop Loss: poziom planowanego ograniczenia straty.",
            "fee/prowizja": "Koszt transakcji pobierany przez giełdę lub symulowany w raporcie.",
            "equity": "Łączna wartość konta: wolne środki plus pozycje.",
            "available balance": "Środki dostępne do użycia, niezamrożone w pozycjach.",
            "in positions": "Kapitał aktualnie zaangażowany w otwarte pozycje.",
            "blacklist": "Lista par pomijanych lub blokowanych w preview.",
            "whitelist": "Lista par dopuszczonych do obserwacji w preview.",
            "Runtime loop not started": "Produkcyjna pętla pracy bota nie została uruchomiona.",
            "Exchange I/O disabled": "UI nie wysyła ani nie pobiera danych z prawdziwej giełdy.",
            "Order submission disabled": "Składanie zleceń jest zablokowane.",
            "Live-like paper simulation": "Lokalna symulacja pracy bota podobna do live: ticki, skan par, decyzje, paper ordery, PnL i telemetria bez giełdy.",
            "Simulation speed": "Interwał lokalnego timera QML sterujący szybkością preview.",
            "Market scenario": "Wybrany mock reżim rynku, który wpływa na lokalne zmiany ceny.",
            "Paper loop": "Bezpieczna pętla paper w UI; produkcyjny runtime loop nie startuje.",
            "No real orders": "Brak prawdziwych zleceń, tras egzekucji, API giełdy i sekretów.",
            "explainability": "Warstwa prostego wyjaśnienia, dlaczego lokalny bot preview wybrał, odrzucił lub zablokował decyzję.",
            "audit trail": "Lista kroków audytu pokazująca snapshot danych, wynik AI, risk guard, paper action i granice bezpieczeństwa.",
            "lineage": "Powiązania decyzji z lokalnym źródłem: skanerem, governorem, ryzykiem, symulacją albo Paper Terminalem.",
            "input snapshot": "Zapis lokalnych wartości wejściowych użytych do wyjaśnienia decyzji w preview.",
            "risk check": "Pojedyncza kontrola ryzyka, np. profil, kill-switch, risk lock, spread albo limit pewności.",
            "decision source": "Miejsce, z którego pochodzi decyzja: Scanner, Governor, Risk, Paper Terminal albo Simulation.",
            "alternative candidate": "Inna para z lokalnego rankingu, która była rozważana, ale nie wygrała.",
            "paper impact": "Wpływ wyłącznie na Paper Preview: możliwy log, paper fill lub brak zmiany PnL/equity.",
            "Alert Center": "Centrum alertów pokazujące lokalną oś zdarzeń Paper Preview.",
            "Critical alert": "Alert krytyczny: kill-switch, safety boundary albo blokada ryzyka wymagają uwagi.",
            "Warning alert": "Alert ostrzegawczy: drawdown, stale heartbeat albo odrzucony setup wymagają obserwacji.",
            "Info alert": "Informacyjny wpis osi zdarzeń, np. heartbeat, diagnostyka albo zmiana zakresu portfela.",
            "unread": "Liczba lokalnych alertów nieoznaczonych jako przeczytane.",
            "event timeline": "Chronologiczna lista lokalnych zdarzeń preview.",
            "muted alerts": "Wyciszone alerty jako stan preview; brak systemowych powiadomień OS.",
            "desktop notification preview": "Przełącznik demonstracyjny powiadomień desktop; nie wysyła powiadomień OS.",
            "stale heartbeat": "Heartbeat telemetryczny uznany za nieświeży w lokalnym preview.",
            "drawdown warning": "Ostrzeżenie o spadku equity/PnL względem lokalnego limitu preview.",
            "Settings": "Panel ustawień produktu preview; działa local-only i nie zapisuje konfiguracji runtime.",
            "Onboarding": "Pierwszy przewodnik po wyborze języka, waluty, trybu, giełdy preview i profilu ryzyka.",
            "Demo Preview": "Tryb demonstracyjny bez live tradingu, zleceń, sekretów i połączeń API.",
            "Paper Preview": "Tryb lokalnej symulacji paper trading bez prawdziwych zleceń.",
            "Sandbox planned": "Planowany sandbox/testnet; w tej wersji preview nadal wyłączony dla połączeń giełda/API.",
            "Live disabled": "Live trading jest wyłączony i nie można go uruchomić z UI Preview.",
            "Base currency": "Waluta bazowa lokalnego podglądu, np. PLN, USD, EUR lub USDT.",
            "UI density": "Rozmiar i gęstość elementów UI wyłącznie w lokalnym preview.",
            "App mode": "Lokalny wybór trybu aplikacji preview; nie uruchamia backendu ani runtime.",
            "Local preview state": "Stan QML w pamięci UI, bez zapisu realnej konfiguracji.",
            "Reset local preview state": "Powrót ustawień i sesji preview do bezpiecznych wartości domyślnych."
        }),
        "EN": ({
            "PnL": "Profit or loss from a trade or session.",
            "ROI": "Percentage result compared with the capital used.",
            "drawdown": "Capital drop from a recent high point.",
            "slippage": "Difference between expected price and execution price.",
            "spread": "Gap between the best buy and sell prices.",
            "order book": "Visible list of buy and sell offers in a market.",
            "paper trading": "Test trading in preview without real money.",
            "sandbox/testnet": "Safe exchange or app testing environment.",
            "API key": "Exchange access key; not required in this preview.",
            "governor": "AI layer that recommends or blocks actions using risk rules.",
            "confidence": "How sure the model is about a recommendation.",
            "confidence floor": "Minimum confidence required before local preview can show PAPER BUY/SELL.",
            "Custom risk": "A user-edited risk profile stored only in local UI preview state.",
            "AI Recommended risk": "A local AI preview profile that chooses limits without backend or exchange calls.",
            "exposure": "Capital used by a position, symbol or strategy.",
            "daily loss limit": "Maximum daily loss allowed in the local paper simulation.",
            "cooldown": "Pause after a decision or block before preview shows a more aggressive action.",
            "risk override": "Risk limit override; in this UI it is only a safe preview toggle.",
            "strategy": "Rules for when to watch, buy, sell or wait.",
            "risk guard": "Safety guard that limits risky actions.",
            "kill-switch": "Emergency stop that blocks trading or system actions.",
            "TP": "Take Profit: planned exit level for a gain.",
            "SL": "Stop Loss: planned level for limiting loss.",
            "fee/prowizja": "Transaction cost charged by an exchange or simulated in the report.",
            "equity": "Total account value: free funds plus positions.",
            "available balance": "Funds available to use, not locked in positions.",
            "in positions": "Capital currently used in open positions.",
            "blacklist": "Pairs skipped or blocked in preview.",
            "whitelist": "Pairs allowed for observation in preview.",
            "Runtime loop not started": "The production bot loop has not been started.",
            "Exchange I/O disabled": "The UI does not send to or fetch from a real exchange.",
            "Order submission disabled": "Submitting orders is blocked.",
            "Live-like paper simulation": "A local bot-like paper loop: ticks, pair scans, decisions, paper orders, PnL and telemetry without exchange access.",
            "Simulation speed": "The local QML timer interval controlling preview speed.",
            "Market scenario": "A mock market regime that influences local price movement.",
            "Paper loop": "Safe paper loop inside the UI; the production runtime loop is not started.",
            "No real orders": "No real orders, execution route, exchange API or secret material.",
            "explainability": "A plain-language layer explaining why the local preview bot chose, rejected or blocked a decision.",
            "audit trail": "A step list showing the data snapshot, AI score, risk guard, paper action and safety boundary.",
            "lineage": "Links from a decision to its local source: scanner, governor, risk, simulation or Paper Terminal.",
            "input snapshot": "Local input values used to build the deterministic preview explanation.",
            "risk check": "One risk control, such as profile, kill-switch, risk lock, spread or confidence limit.",
            "decision source": "Where the decision came from: Scanner, Governor, Risk, Paper Terminal or Simulation.",
            "alternative candidate": "Another local-ranked pair that was considered but did not win.",
            "paper impact": "Impact inside Paper Preview only: possible log, paper fill, or no PnL/equity change.",
            "Market Scanner": "Product scanner that ranks local preview pairs without live trading.",
            "AI score": "Local opportunity score for a candidate pair.",
            "Risk score": "Local risk estimate; lower is safer in preview.",
            "Liquidity": "How easy a local preview pair appears to trade without wide friction.",
            "Volatility": "Local preview price movement range.",
            "Trend": "Direction and strength of the local preview move.",
            "Watchlist": "Pairs kept for observation in local preview.",
            "Blacklist": "Pairs blocked or skipped by local preview.",
            "Candidate": "A pair good enough to be reviewed by the scanner.",
            "Rejected setup": "A weak setup ignored by the scanner.",
            "Strategy match": "The preview strategy that best explains the candidate.",
            "Alert Center": "A local Paper Preview event center with an alert timeline.",
            "Critical alert": "A critical alert for kill-switch, safety boundary or risk block events.",
            "Warning alert": "A warning alert for drawdown, stale heartbeat or rejected setups.",
            "Info alert": "An informational event timeline entry such as heartbeat, diagnostics or range changes.",
            "unread": "The count of local alerts not yet marked read.",
            "event timeline": "A chronological list of local preview events.",
            "muted alerts": "Muted alerts as preview state only; no OS notifications are sent.",
            "desktop notification preview": "A demo desktop-notification toggle that never sends OS notifications.",
            "stale heartbeat": "Telemetry heartbeat treated as stale by local preview state.",
            "drawdown warning": "A warning about PnL/equity drawdown against a local preview limit.",
            "Settings": "Preview product settings panel; local-only and does not write runtime config.",
            "Onboarding": "First-run guide for language, base currency, app mode, preview exchange and risk profile.",
            "Demo Preview": "Demo mode without live trading, orders, secrets or API calls.",
            "Paper Preview": "Local paper trading simulation without real orders.",
            "Sandbox planned": "Planned sandbox/testnet; exchange/API connections remain disabled in this preview.",
            "Live disabled": "Live trading is disabled and cannot be started from UI Preview.",
            "Base currency": "Local preview base currency such as PLN, USD, EUR or USDT.",
            "UI density": "UI element density stored only in local preview state.",
            "App mode": "Local preview app mode selection; does not start backend or runtime.",
            "Local preview state": "QML UI memory state with no real runtime configuration write.",
            "Reset local preview state": "Return settings and session preview to safe defaults."
        })
    })
    readonly property var rootDesignSystem: designSystem

    // UI-PREVIEW-7.5 local-only product preview state. Safe dry-run/paper preview only: live trading disabled, exchange route disabled, order submission disabled, API keys not required.
    property var selectedExchanges: ["Paper Preview Catalog"]
    property var selectedPairs: ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    property var whitelistPairs: ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    property var blacklistPairs: []
    property var activeStrategies: ["Momentum Guard", "Range Guard"]
    property string paperSessionState: "stopped"
    property int paperTicks: 0
    property int decisionSequence: 0
    property int telemetryTick: 0
    readonly property int previewTelemetryFeedLimit: 12
    readonly property int previewAlertFeedLimit: 12
    readonly property int previewDecisionFeedLimit: 12
    readonly property int previewPaperOrderFeedLimit: 12
    readonly property int previewScannerRowsLimit: 60
    readonly property int previewTerminalLogLimit: 12
    readonly property int previewOpenPositionFeedLimit: 12
    readonly property int previewClosedTradeFeedLimit: 12
    property real previewEquity: 102020.0
    property real previewPnl: 2020.0
    property alias mockEquity: root.previewEquity
    property alias mockPnl: root.previewPnl
    property int paperOrdersCount: 0
    property int blockedOrdersCount: 0
    property int noOrderCount: 1
    property int simulatedOrdersCount: 0
    property var paperOrdersPreview: [
        ({ timestamp: "12:00:01Z", pair: "BTC/USDT", action: "HOLD", status: "no order", confidence: "0.81", reason: "Preview guard held the setup; order submission disabled." })
    ]
    property var openPaperPositions: [
        ({ pair: "BTC/USDT", side: "paper long", size: "0.012", pnl: "+42.10", label: "simulated" }),
        ({ pair: "SOL/USDT", side: "watch", size: "0", pnl: "0.00", label: "no order" })
    ]
    property var closedPaperTrades: [
        ({ pair: "ETH/USDT", side: "paper sell", pnl: "+18.44", label: "simulated" })
    ]
    // UI-PREVIEW-7.7 local-only paper bridge/state shared by Dashboard, Decisions, Telemetry and Paper Terminal.
    // Paper Preview only: live trading disabled, exchange I/O disabled, order submission disabled, API keys not required, runtime loop not started, no real orders.
    property string paperSessionStatus: "stopped"
    property real paperEquity: 102020.0
    property real paperPnl: 2020.0
    property int paperSessionTicks: 0
    property int paperBlockedCount: 0
    property int paperNoOrderCount: 1
    property int paperSimulatedCount: 0
    property var paperOpenPositions: [
        ({ pair: "BTC/USDT", side: "paper long", size: "0.012", pnl: "+42.10", label: "simulated" }),
        ({ pair: "SOL/USDT", side: "watch", size: "0", pnl: "0.00", label: "no order" })
    ]
    property var paperClosedTrades: [
        ({ pair: "ETH/USDT", side: "paper sell", pnl: "+18.44", label: "simulated" })
    ]
    property var paperOrderRows: [
        ({ timestamp: "12:00:01Z", pair: "BTC/USDT", action: "HOLD", status: "no order", confidence: "0.81", reason: "Preview guard held the setup; order submission disabled." })
    ]
    property var paperTelemetryRows: [
        ({ timestamp: "12:04:18Z", message: "local-only paper bridge/state ready • runtime loop not started" }),
        ({ timestamp: "12:03:58Z", message: "Paper Preview only • exchange I/O disabled • no real orders" })
    ]
    // UI-PREVIEW-8.0D QML-only live-like paper simulation loop. Local timer only: no backend runtime loop, no exchange I/O, no order submission, no API keys, no secret reads.
    property bool simulationRunning: false
    property bool simulationPaused: false
    property int simulationSpeed: 1
    property int simulationTickIntervalMs: 1200
    property string simulationScenario: "Balanced preview"
    property int simulationTickCount: 0
    property string simulationLastTickAt: "not started"
    property string simulationMarketMode: "balanced"
    property string simulationStatusLabel: "stopped • local paper loop only"
    property var simulationEvents: []
    property var simulationScenarios: ["Balanced preview", "Bull trend", "Bear trend", "High volatility", "Sideways/range"]
    property string simulationLastPair: "—"
    property string simulationLastAction: "—"
    property string simulationLastOrder: "—"
    property string simulationSafetyBoundary: "Local paper loop only • no exchange API • no real orders • no secrets • production runtime loop not started"
    property var simulationPrices: ({ "BTC/USDT": 68240.50, "ETH/USDT": 3560.10, "SOL/USDT": 154.20 })
    property var mockTerminalCandles: [
        ({ x: 36, open: 145, close: 104, high: 78, low: 174, vol: 44 }), ({ x: 78, open: 112, close: 168, high: 94, low: 194, vol: 72 }),
        ({ x: 120, open: 164, close: 122, high: 96, low: 184, vol: 58 }), ({ x: 162, open: 128, close: 88, high: 64, low: 148, vol: 38 }),
        ({ x: 204, open: 92, close: 154, high: 82, low: 178, vol: 82 }), ({ x: 246, open: 158, close: 128, high: 106, low: 188, vol: 66 }),
        ({ x: 288, open: 132, close: 86, high: 70, low: 154, vol: 48 }), ({ x: 330, open: 90, close: 142, high: 78, low: 168, vol: 74 }),
        ({ x: 372, open: 146, close: 98, high: 76, low: 166, vol: 62 }), ({ x: 414, open: 102, close: 72, high: 56, low: 132, vol: 36 }),
        ({ x: 456, open: 76, close: 128, high: 66, low: 158, vol: 70 }), ({ x: 498, open: 132, close: 92, high: 74, low: 152, vol: 54 })
    ]
    // UI-PREVIEW-8.0A local-only portfolio/performance preview state. No backend, no exchange I/O, no order submission, no secrets/env/keychain reads.
    property string portfolioBaseCurrency: "USD"
    property real portfolioStartingEquityUsd: 100000.0
    property real portfolioTotalEquityUsd: 102020.0
    property real portfolioTotalEquityPln: 402468.90
    property real portfolioAvailableBalanceUsd: 68115.0
    property real portfolioInPositionsUsd: 30580.0
    property real portfolioReservedMarginUsd: 3325.0
    property real portfolioTradingEquityUsdt: 102020.0
    property real portfolioTradingAvailableUsdt: 68115.0
    property real portfolioRealizedPnlUsd: 1900.0
    property real portfolioUnrealizedPnlUsd: 220.0
    property real portfolioFeesUsd: 86.40
    property real portfolioFundingOtherCostsUsd: 13.60
    property real portfolioNetPnlUsd: 2020.0
    property real portfolioSessionPnlUsd: 2020.0
    property real portfolioLastCyclePnlUsd: 126.75
    property string portfolioLastCycleId: "CYCLE-20260602-0810"
    property string portfolioLastCycleTimestamp: "2026-06-02 08:45 UTC"
    property int portfolioLastCycleTradesCount: 7
    property int portfolioLastCycleWinners: 5
    property int portfolioLastCycleLosers: 2
    property real portfolioLastCycleFeesUsd: 8.40
    property real portfolioLastCycleNetUsd: 118.35
    property real portfolioAllTimePnlUsd: 2020.0
    property real portfolioRoiPercent: 2.02
    property string portfolioFiatAccountLabel: "102 020.00 USD / 402 468.90 PLN"
    property string portfolioCryptoAccountLabel: "BTC 0.412 • ETH 3.80 • SOL 92.4 • USDT 18 640"
    property string portfolioSelectedRange: "1d"
    property string portfolioRangeLabel: "Zakres: 1d preview • filtry raportowe nie zmieniają aktywnej sesji Paper"
    property string portfolioCustomFrom: "2026-06-01 00:00"
    property string portfolioCustomTo: "2026-06-02 23:59"
    property var portfolioTimeFilters: ["1h", "1d", "7d", "1m", "1y", "all", "custom"]
    property string portfolioWinRate: "64.6%"
    property string portfolioMaxDrawdown: "-3.8%"
    property string portfolioBestPair: "BTC/USDT +1 120.00 USD"
    property string portfolioWorstPair: "ARB/USDT -240.00 USD"
    property int portfolioTradeCount: 48
    property var portfolioPerformanceCards: [
        ({ title: "Fiat balance / equity", field: "fiat" }),
        ({ title: "Trading balance / equity", field: "trading" }),
        ({ title: "Available balance", field: "available" }),
        ({ title: "In positions", field: "positions" }),
        ({ title: "Reserved / margin preview", field: "reserved" }),
        ({ title: "PnL ostatniego cyklu", field: "lastCycle" }),
        ({ title: "Paper session PnL", field: "paperSession" }),
        ({ title: "PnL całkowity", field: "allTime" }),
        ({ title: "Win rate", field: "winRate" }),
        ({ title: "Liczba transakcji", field: "tradeCount" }),
        ({ title: "Prowizje", field: "fees" }),
        ({ title: "Max drawdown", field: "drawdown" }),
        ({ title: "Najlepsza para", field: "bestPair" }),
        ({ title: "Najgorsza para", field: "worstPair" })
    ]
    property var portfolioAllCycleRows: [
        ({ startTime: "2026-06-02 08:10", endTime: "2026-06-02 08:45", pair: "BTC/USDT", strategy: "Momentum Guard", result: "+126.75 USD", fee: "8.40 USD", status: "closed preview", closeReason: "TP" }),
        ({ startTime: "2026-06-02 07:15", endTime: "2026-06-02 07:58", pair: "ETH/USDT", strategy: "Range Guard", result: "+64.20 USD", fee: "5.80 USD", status: "closed preview", closeReason: "AI exit" }),
        ({ startTime: "2026-06-01 22:05", endTime: "2026-06-01 23:20", pair: "SOL/USDT", strategy: "Volatility Breakout Preview", result: "-42.10 USD", fee: "4.10 USD", status: "guarded preview", closeReason: "risk guard" }),
        ({ startTime: "2026-06-01 18:30", endTime: "2026-06-01 19:05", pair: "BNB/USDT", strategy: "Strategy governor", result: "+38.00 USD", fee: "3.60 USD", status: "manual preview", closeReason: "manual preview" }),
        ({ startTime: "2026-06-01 13:00", endTime: "2026-06-01 13:22", pair: "ARB/USDT", strategy: "Range Guard", result: "-58.30 USD", fee: "2.90 USD", status: "closed preview", closeReason: "SL" }),
        ({ startTime: "2026-05-31 10:05", endTime: "2026-05-31 11:10", pair: "LINK/USDT", strategy: "Momentum Guard", result: "+212.40 USD", fee: "9.10 USD", status: "closed preview", closeReason: "TP" })
    ]
    property var portfolioCycleRows: portfolioAllCycleRows.slice(0, 5)
    property var portfolioRangeSnapshots: ({
        "1h": ({ realized: 122.00, unrealized: 18.10, fees: 7.20, fundingOtherCosts: 2.90, sessionPnl: 130.00, lastCyclePnl: 38.40, tradeCount: 4, winRate: "75.0%", maxDrawdown: "-0.4%", bestPair: "BTC/USDT +82.00 USD", worstPair: "SOL/USDT -12.00 USD", cycleRows: portfolioAllCycleRows.slice(0, 1) }),
        "1d": ({ realized: 1900.00, unrealized: 220.00, fees: 86.40, fundingOtherCosts: 13.60, sessionPnl: 2020.00, lastCyclePnl: 126.75, lastCycleTrades: 7, lastCycleWinners: 5, lastCycleLosers: 2, lastCycleFees: 8.40, lastCycleNet: 118.35, tradeCount: 48, winRate: "64.6%", maxDrawdown: "-3.8%", bestPair: "BTC/USDT +1 120.00 USD", worstPair: "ARB/USDT -240.00 USD", cycleRows: portfolioAllCycleRows.slice(0, 5) }),
        "7d": ({ realized: 2640.00, unrealized: 330.00, fees: 118.80, fundingOtherCosts: 31.20, sessionPnl: 2820.00, lastCyclePnl: 212.40, lastCycleTrades: 9, lastCycleWinners: 7, lastCycleLosers: 2, lastCycleFees: 9.10, lastCycleNet: 203.30, tradeCount: 82, winRate: "67.1%", maxDrawdown: "-4.2%", bestPair: "LINK/USDT +1 420.00 USD", worstPair: "ARB/USDT -310.00 USD", cycleRows: portfolioAllCycleRows.slice(0, 6) }),
        "1m": ({ realized: 5120.00, unrealized: 680.00, fees: 246.50, fundingOtherCosts: 73.50, sessionPnl: 5480.00, lastCyclePnl: 310.20, lastCycleTrades: 12, lastCycleWinners: 8, lastCycleLosers: 4, lastCycleFees: 14.50, lastCycleNet: 295.70, tradeCount: 196, winRate: "62.8%", maxDrawdown: "-5.6%", bestPair: "BTC/USDT +2 840.00 USD", worstPair: "OP/USDT -520.00 USD", cycleRows: portfolioAllCycleRows.slice(0, 6) }),
        "1y": ({ realized: 18450.00, unrealized: 1120.00, fees: 940.00, fundingOtherCosts: 210.00, sessionPnl: 18420.00, lastCyclePnl: 640.80, lastCycleTrades: 18, lastCycleWinners: 13, lastCycleLosers: 5, lastCycleFees: 32.00, lastCycleNet: 608.80, tradeCount: 1180, winRate: "60.2%", maxDrawdown: "-8.9%", bestPair: "BTC/USDT +8 900.00 USD", worstPair: "ARB/USDT -1 460.00 USD", cycleRows: portfolioAllCycleRows.slice(0, 6) }),
        "all": ({ realized: 22600.00, unrealized: 1640.00, fees: 1180.00, fundingOtherCosts: 260.00, sessionPnl: 22800.00, lastCyclePnl: 640.80, lastCycleTrades: 18, lastCycleWinners: 13, lastCycleLosers: 5, lastCycleFees: 32.00, lastCycleNet: 608.80, tradeCount: 1460, winRate: "61.4%", maxDrawdown: "-9.4%", bestPair: "BTC/USDT +10 400.00 USD", worstPair: "ARB/USDT -1 720.00 USD", cycleRows: portfolioAllCycleRows.slice(0, 6) }),
        "custom": ({ realized: 820.00, unrealized: 95.00, fees: 36.00, fundingOtherCosts: 9.00, sessionPnl: 870.00, lastCyclePnl: 64.20, lastCycleTrades: 5, lastCycleWinners: 4, lastCycleLosers: 1, lastCycleFees: 5.80, lastCycleNet: 58.40, tradeCount: 21, winRate: "66.7%", maxDrawdown: "-1.9%", bestPair: "ETH/USDT +420.00 USD", worstPair: "SOL/USDT -96.00 USD", cycleRows: portfolioAllCycleRows.slice(1, 4) })
    })
    property string lastGovernorDecision: "BTC/USDT HOLD • confidence 0.81 • NO ORDER — preview only"
    property string autonomyMode: "Supervised dry-run"
    property int autonomyLevel: 2
    property int modelReadiness: 72
    property int trainingCoverage: 68
    property int dataCoverage: 74
    property int confidenceThreshold: 75
    property string decisionPolicyPreview: "Polityka zbalansowana"
    property string activeGovernorEngine: "Dudzian Governor Ensemble"
    property string modelVersionBuild: "preview-7.5 build local-2026.06"
    property bool riskLocked: false
    property string riskProfile: "Balanced"
    property string riskState: "Balanced preview • daily loss limit active • live blocked"
    property string riskBlockReason: "Risk gate clear: Balanced local preview limits allow paper actions within confidence, position and drawdown constraints."
    property string maxPosition: "2,500 USDT"
    property int maxOpenPositions: 4
    property string stopLoss: "2.4%"
    property string takeProfit: "4.8%"
    property string maxSlippage: "0.20%"
    property string maxDrawdown: "6.0%"
    property string dailyLossLimit: "1,200 USDT"
    property string perSymbolExposure: "18% equity"
    property string confidenceFloor: "75%"
    property string cooldown: "120s"
    property string maxAllocation: "18% equity"
    property bool allowAiOverride: false
    property string riskLimitSource: "Balanced"
    property string riskExplanation: "Dlaczego takie ustawienia? Balanced utrzymuje średnie limity dla lokalnego Paper Preview: umiarkowana pozycja, standardowy stop loss i aktywny daily loss limit."
    property var aiRecommendedChangedParameters: []
    property var customRiskState: ({ maxPosition: "2,500 USDT", maxOpenPositions: 4, stopLoss: "2.4%", takeProfit: "4.8%", maxSlippage: "0.20%", maxDrawdown: "6.0%", dailyLossLimit: "1,200 USDT", perSymbolExposure: "18% equity", confidenceFloor: "75%", cooldown: "120s", maxAllocation: "18% equity", allowAiOverride: false })
    property var riskActiveLimits: []
    property bool riskRuntimeConfigWritten: false
    property bool liveTradingDisabled: true
    property bool exchangeIoDisabled: true
    property bool orderSubmissionDisabled: true
    property bool apiKeysRequired: false
    property bool runtimeLoopStarted: false
    // UI-PREVIEW-8.0I settings/onboarding state. Local-only: no runtime config write, no secrets read, no exchange/API calls, no order submission.
    property bool settingsPanelOpen: currentPanelId === "settingsPanel"
    property string appModePreview: "Demo Preview"
    property string baseCurrency: "USDT"
    property string uiDensity: "Comfortable"
    property string themeModePreview: "Dark preview"
    property string defaultPreviewExchange: "Paper Preview Catalog"
    property string defaultTerminalPair: "BTC/USDT"
    property string defaultRiskProfile: "Balanced"
    property bool settingsDirty: false
    property string settingsLastUpdatedAt: "not applied yet"
    property string settingsSafetySummary: "Settings are local preview only • No runtime config is written • No secrets are read • No exchange/API calls • No order submission • Live trading remains disabled • Ustawienia działają lokalnie w preview • Konfiguracja runtime nie jest zapisywana • Sekrety nie są odczytywane • Brak połączeń giełda/API • Brak składania zleceń • Live trading pozostaje wyłączony"
    property bool firstRunWizardVisible: false
    property int onboardingStep: 1
    property bool onboardingCompletedPreview: false
    property var appModePreviewOptions: ["Demo Preview", "Paper Preview", "Sandbox planned", "Live disabled"]
    property var baseCurrencyOptions: ["PLN", "USD", "EUR", "USDT"]
    property var uiDensityOptions: ["Compact", "Comfortable", "Large"]
    property var themeModePreviewOptions: ["Dark preview", "Light planned"]
    property var defaultPreviewExchangeOptions: ["Paper Preview Catalog", "Binance preview catalog", "Coinbase preview catalog", "Kraken preview catalog"]
    property var defaultTerminalPairOptions: ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
    property var defaultRiskProfileOptions: ["Conservative", "Balanced", "Aggressive", "AI Recommended"]
    property var onboardingSteps: [
        "Wybierz język / Choose language",
        "Wybierz walutę bazową / Choose base currency",
        "Wybierz tryb / Choose app mode",
        "Wybierz giełdę preview / Choose preview exchange",
        "Wybierz profil ryzyka / Choose risk profile",
        "Uruchom Paper Preview / przejdź do Dashboard"
    ]
    property string globalSafetyBadgeSummary: "Live trading: disabled • Exchange I/O: disabled • Order submission: disabled • API keys: not required • Runtime loop: not started • Safety: safe preview"
    property string appStatusSummary: "Mode: " + appModePreview + " • Alerts: " + alertUnreadCount + " • Lang: " + currentLanguage + " • Base: " + baseCurrency + " • Risk: " + defaultRiskProfile + " • Simulation: " + simulationStatusLabel
    property bool settingsRuntimeConfigWritten: false
    property bool settingsSecretsRead: false
    property bool topNavigationHorizontalScroll: true
    property bool marketsImported: false
    property string marketSearch: ""
    property string marketQuoteFilter: "All"
    property string marketCategoryFilter: "All"
    property bool marketSelectedOnly: false
    property bool marketAiCandidatesOnly: false
    property bool marketExcludedOnly: false
    property string decisionFilter: "all"
    property string decisionPairFilter: "All pairs"
    property var previewMarketPairs: [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
        "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT", "LTC/USDT", "BCH/USDT",
        "ATOM/USDT", "NEAR/USDT", "ARB/USDT", "OP/USDT", "INJ/USDT", "APT/USDT",
        "SUI/USDT", "TON/USDT", "PEPE/USDT", "WIF/USDT", "FET/USDT", "RENDER/USDT",
        "TAO/USDT", "UNI/USDT", "AAVE/USDT", "ETC/USDT", "FIL/USDT", "ICP/USDT",
        "MATIC/USDT", "SEI/USDT", "TIA/USDT", "JUP/USDT", "PYTH/USDT", "WLD/USDT",
        "GRT/USDT", "RNDR/USDT", "MKR/USDT", "LDO/USDT", "CRV/USDT", "SAND/USDT",
        "MANA/USDT", "AXS/USDT", "GALA/USDT", "SHIB/USDT", "FLOKI/USDT", "BONK/USDT",
        "ENA/USDT", "JTO/USDT", "STX/USDT", "KAS/USDT", "ALGO/USDT", "VET/USDT",
        "HBAR/USDT", "EGLD/USDT", "RUNE/USDT", "FTM/USDT", "IMX/USDT", "STRK/USDT",
        "ZK/USDT", "MANTA/USDT", "BLUR/USDT", "DYDX/USDT", "SNX/USDT", "COMP/USDT",
        "YFI/USDT", "SUSHI/USDT", "1INCH/USDT", "CAKE/USDT", "PENDLE/USDT", "ONDO/USDT",
        "BEAM/USDT", "ROSE/USDT", "MINA/USDT", "AR/USDT", "ENS/USDT", "MASK/USDT",
        "CHZ/USDT", "APE/USDT", "BTC/USDC", "ETH/USDC", "SOL/USDC", "BNB/USDC",
        "XRP/USDC", "ADA/USDC", "DOGE/USDC", "AVAX/USDC", "DOT/USDC", "LINK/USDC",
        "LTC/USDC", "BCH/USDC", "ATOM/USDC", "NEAR/USDC", "ARB/USDC", "OP/USDC",
        "INJ/USDC", "APT/USDC", "SUI/USDC", "TON/USDC", "PEPE/USDC", "WIF/USDC",
        "FET/USDC", "RENDER/USDC", "TAO/USDC", "UNI/USDC", "AAVE/USDC", "ETC/USDC",
        "FIL/USDC", "ICP/USDC", "MATIC/USDC", "SEI/USDC", "TIA/USDC", "JUP/USDC",
        "PYTH/USDC", "WLD/USDC", "GRT/USDC", "RNDR/USDC", "MKR/USDC", "LDO/USDC",
        "CRV/USDC", "SAND/USDC", "MANA/USDC", "AXS/USDC", "GALA/USDC", "SHIB/USDC",
        "FLOKI/USDC", "BONK/USDC"
    ]
    property var previewExchanges: [
        "Binance", "Bybit", "OKX", "KuCoin", "Coinbase", "Kraken", "Bitget", "Gate.io", "MEXC", "Paper Preview Catalog"
    ]
    // UI-PREVIEW-8.0F local-only Market Scanner / Okazje preview state. No live trading, no exchange I/O, no order submission, no API keys, no real orders, no network/API calls.
    property string scannerStatus: "safe preview"
    property bool scannerActive: false
    property string scannerLastScanAt: "not scanned"
    property int scannerTickCount: 0
    property string scannerSelectedExchange: "Paper Preview Catalog"
    property int scannerUniverseCount: 0
    property int scannerCandidateCount: 0
    property int scannerRejectedCount: 0
    property int scannerWatchlistCount: 0
    property var scannerWatchlistPairs: []
    property string scannerBestOpportunity: "—"
    property var scannerRows: []
    property var scannerRejectedRows: []
    property var scannerWatchlistRows: []
    property var scannerAiCandidateRows: []
    property string scannerFilterMode: "All"
    property string scannerSortMode: "AI score"
    property real scannerMinAiScore: 70
    property real scannerMinLiquidityScore: 60
    property real scannerMaxRiskScore: 55
    property string scannerSelectedPair: "BTC/USDT"
    property string scannerExplanation: "Wybierz parę albo kliknij Explain candidate. Bot pokaże, co przemawia za, co przeciw, czy ryzyko przepuszcza, jaka strategia pasuje i co zrobiłby w paper preview."
    property string scannerSafetyBoundary: "Safe preview scanner • Live trading disabled • Exchange I/O disabled • Order submission disabled • API keys not required • No real orders • No network/API calls • Local preview catalog only"
    property var decisionPreviewRows: [
        ({ timestamp: "12:04:18Z", symbol: "BTC/USDT", action: "HOLD", confidence: "0.81", reason: "Momentum neutral; confidence floor not reached.", riskReason: "Position cap unused; drawdown guard inside preview limit.", strategy: "Momentum Guard", safety: "NO ORDER — preview only", paperState: "stopped" }),
        ({ timestamp: "12:03:42Z", symbol: "ETH/USDT", action: "WAIT", confidence: "0.74", reason: "Coverage check asks for a fresher candle batch.", riskReason: "Risk governor waits for lower slippage.", strategy: "Range Guard", safety: "Exchange route disabled", paperState: "stopped" }),
        ({ timestamp: "12:02:57Z", symbol: "SOL/USDT", action: "BLOCKED", confidence: "0.69", reason: "Live bridge is intentionally unavailable in preview.", riskReason: "Execution guard blocks the order route.", strategy: "Volatility Breakout Preview", safety: "Live trading disabled • Order submission disabled", paperState: "stopped" }),
        ({ timestamp: "12:01:33Z", symbol: "BNB/USDT", action: "NO ORDER", confidence: "0.61", reason: "Advisory preview rejected low confidence setup.", riskReason: "Below confidence threshold.", strategy: "Strategy governor", safety: "Preview only / no order", paperState: "stopped" })
    ]
    // UI-PREVIEW-8.0G QML-only decision explainability/audit drawer state. Local deterministic preview only: no backend AI inference, no network/API calls, no order submission, no secret reads.
    property bool decisionExplainDrawerOpen: false
    property string selectedDecisionId: ""
    property string selectedDecisionPair: "BTC/USDT"
    property string selectedDecisionAction: "HOLD"
    property string selectedDecisionSource: "Governor"
    property string selectedDecisionConfidence: "0.81"
    property string selectedDecisionRiskState: "Risk profile Balanced • kill-switch inactive • risk lock inactive"
    property string selectedDecisionStrategy: "Momentum Guard"
    property string selectedDecisionReason: "Bot nie kupił, bo score AI był za niski, a profil ryzyka wymaga wyższej pewności. Zmieniono tylko lokalny log preview; PnL i equity pozostają bez zmian."
    property var selectedDecisionAuditRows: []
    property var selectedDecisionInputSnapshot: []
    property var selectedDecisionAlternatives: []
    property var selectedDecisionRiskChecks: []
    property var selectedDecisionLineageLinks: []
    property string selectedDecisionPaperImpact: "brak fill / brak zmiany finansowej"
    property string selectedDecisionSafetySummary: "Explanation is local preview only • No backend AI inference • No exchange/API call • No order submission • No real orders • No secrets read • Wyjaśnienie działa lokalnie w preview • Brak backendowej inferencji AI • Brak połączenia z giełdą/API • Brak składania zleceń • Brak prawdziwych zleceń • Brak odczytu sekretów"
    property string telemetryHeartbeat: "12:04:18Z"
    property int telemetryReconnects: 0
    property string telemetryDowntime: "0 ms"
    property string telemetryFreshness: "freshness status: ready • safe preview feed"
    property var telemetryRows: [
        ({ timestamp: "12:04:18Z", message: "heartbeat #0 • feed: safe preview • runtime loop: not started" }),
        ({ timestamp: "12:03:58Z", message: "exchange route: disabled • paper bridge: not connected / planned" })
    ]
    property string diagnosticsBundleStatus: "Last bundle path/status: not generated"
    property string lastStrategySaveStatus: "Brak lokalnego zapisu preview"


    // UI-PREVIEW-7.6 local-only Product Trading Terminal / Paper Trading Cockpit state.
    // Safe preview only: live trading disabled, exchange I/O disabled, order submission disabled, API keys not required, runtime loop not started, no real orders.
    property string selectedTerminalPair: selectedPairs && selectedPairs.length > 0 ? selectedPairs[0] : "BTC/USDT"
    property string selectedTerminalPairLastWriter: "initial:selectedPairs-default"
    property string terminalSide: "BUY"
    property string terminalOrderType: "LIMIT"
    property string terminalPrice: "68240.50"
    property string terminalAmount: "0.010"
    property string terminalTotal: "682.41"
    property bool terminalAutoConfirm: false
    property string terminalSelectedBottomTab: "Positions"
    property string terminalPairSearch: ""
    property string terminalTimeframe: "15m"
    property string terminalTakeProfit: "2.80%"
    property string terminalStopLoss: "1.20%"
    property var mockOrderBookAsks: [
        ({ price: "68302.40", amount: "0.184", total: "12567.64", action: "Use ask" }),
        ({ price: "68295.10", amount: "0.312", total: "21308.07", action: "Use ask" }),
        ({ price: "68288.60", amount: "0.146", total: "9970.14", action: "Use ask" }),
        ({ price: "68281.90", amount: "0.428", total: "29229.65", action: "Use ask" }),
        ({ price: "68275.20", amount: "0.233", total: "15907.12", action: "Use ask" }),
        ({ price: "68269.40", amount: "0.517", total: "35304.28", action: "Use ask" }),
        ({ price: "68264.20", amount: "0.096", total: "6553.36", action: "Use ask" }),
        ({ price: "68258.50", amount: "0.287", total: "19591.19", action: "Use ask" }),
        ({ price: "68251.20", amount: "0.356", total: "24297.43", action: "Use ask" }),
        ({ price: "68245.80", amount: "0.142", total: "9690.90", action: "Use ask" })
    ]
    property var mockOrderBookBids: [
        ({ price: "68236.10", amount: "0.188", total: "12844.39", action: "Use bid" }),
        ({ price: "68231.70", amount: "0.431", total: "29400.86", action: "Use bid" }),
        ({ price: "68225.30", amount: "0.274", total: "18693.73", action: "Use bid" }),
        ({ price: "68220.40", amount: "0.265", total: "18078.41", action: "Use bid" }),
        ({ price: "68212.90", amount: "0.502", total: "34282.88", action: "Use bid" }),
        ({ price: "68206.60", amount: "0.118", total: "8048.38", action: "Use bid" }),
        ({ price: "68199.20", amount: "0.349", total: "23801.52", action: "Use bid" }),
        ({ price: "68192.50", amount: "0.622", total: "42455.74", action: "Use bid" }),
        ({ price: "68185.70", amount: "0.213", total: "14522.55", action: "Use bid" }),
        ({ price: "68178.30", amount: "0.461", total: "31430.20", action: "Use bid" })
    ]
    property var mockTerminalPositions: [
        ({ pair: "BTC/USDT", side: "paper long", size: "0.012", entry: "68110.20", pnl: "+42.10", status: "open paper" }),
        ({ pair: "ETH/USDT", side: "paper watch", size: "0.000", entry: "—", pnl: "0.00", status: "no real orders" })
    ]
    property var mockTerminalOrders: [
        ({ time: "12:02:10Z", pair: "BTC/USDT", side: "BUY", type: "LIMIT", price: "68180.00", amount: "0.010", status: "paper pending" }),
        ({ time: "12:03:22Z", pair: "SOL/USDT", side: "SELL", type: "MARKET", price: "preview", amount: "1.500", status: "simulated only" })
    ]
    property var mockTerminalHistory: [
        ({ time: "11:58:40Z", pair: "ETH/USDT", side: "SELL", price: "3560.10", amount: "0.250", result: "+18.44 paper" }),
        ({ time: "11:45:12Z", pair: "BTC/USDT", side: "BUY", price: "67980.00", amount: "0.006", result: "+9.12 paper" })
    ]
    property var mockTerminalReservedBalances: [
        ({ asset: "USDT", reserved: "682.41", reason: "paper limit preview", status: "local only" }),
        ({ asset: "BTC", reserved: "0.0100", reason: "paper position margin", status: "no exchange lock" }),
        ({ asset: "ETH", reserved: "0.0000", reason: "idle preview", status: "available" })
    ]
    property var terminalLogRows: [
        ({ time: "12:04:00Z", message: "Paper Preview loaded • live trading disabled • exchange I/O disabled" }),
        ({ time: "12:04:03Z", message: "Runtime loop not started • order submission disabled • API keys not required" })
    ]


    // UI-PREVIEW-8.0H Alert/Event Center state. Local-only QML preview: no OS notifications, no backend calls, no exchange/API calls, no order submission, no secrets read.
    property bool alertCenterOpen: false
    property var alertRows: [
        ({ time: "12:04:30Z", severity: "Critical", category: "Risk", source: "Risk Governor", pair: "BTC/USDT", title: "Risk blocked action", message: "Risk lock blocked a local preview action; PnL/equity unchanged.", action: "Review risk controls", read: false, status: "unread" }),
        ({ time: "12:04:24Z", severity: "Critical", category: "Safety", source: "Safety Boundary", pair: "—", title: "Kill-switch active", message: "Safety kill-switch active in safe paper preview.", action: "Keep live disabled", read: false, status: "unread" }),
        ({ time: "12:04:18Z", severity: "Warning", category: "Portfolio", source: "Portfolio Preview", pair: "—", title: "Drawdown warning", message: "Local drawdown warning generated from preview equity/PnL state.", action: "Inspect portfolio range", read: false, status: "unread" }),
        ({ time: "12:04:12Z", severity: "Info", category: "AI", source: "AI Governor", pair: "ETH/USDT", title: "AI decision generated", message: "AI/Governor generated a deterministic local preview decision.", action: "Explain event", read: false, status: "unread" }),
        ({ time: "12:04:06Z", severity: "Info", category: "Scanner", source: "Market Scanner", pair: "SOL/USDT", title: "Scanner candidate found", message: "Scanner found a candidate from the local preview catalog.", action: "Open scanner", read: true, status: "read" }),
        ({ time: "12:04:00Z", severity: "Warning", category: "Scanner", source: "Market Scanner", pair: "BNB/USDT", title: "Scanner rejected setup", message: "Scanner rejected a setup because local score/risk thresholds were not met.", action: "Review thresholds", read: true, status: "read" }),
        ({ time: "12:03:54Z", severity: "Info", category: "Paper", source: "Paper Terminal", pair: "BTC/USDT", title: "Paper order simulated", message: "Paper order simulated locally; no real order route exists.", action: "Open Paper Terminal", read: true, status: "read" }),
        ({ time: "12:03:48Z", severity: "Warning", category: "Paper", source: "Paper Terminal", pair: "ETH/USDT", title: "Paper order blocked", message: "Local validation or risk lock blocked a paper order preview.", action: "Inspect blocked order", read: true, status: "read" }),
        ({ time: "12:03:42Z", severity: "Info", category: "Portfolio", source: "Paper Portfolio", pair: "—", title: "PnL changed", message: "Preview PnL/equity changed after a local simulated paper event.", action: "Open portfolio", read: true, status: "read" }),
        ({ time: "12:03:36Z", severity: "Info", category: "Risk", source: "Risk Guard", pair: "—", title: "Risk profile changed", message: "Risk profile changed in local UI preview state.", action: "Review active limits", read: true, status: "read" }),
        ({ time: "12:03:30Z", severity: "Info", category: "Risk", source: "AI Risk", pair: "—", title: "AI Recommended risk applied", message: "AI Recommended risk applied locally without backend inference.", action: "Explain risk", read: true, status: "read" }),
        ({ time: "12:03:24Z", severity: "Info", category: "Portfolio", source: "Portfolio Range", pair: "—", title: "Portfolio range changed", message: "Portfolio report range changed without mutating paper session state.", action: "Review report", read: true, status: "read" }),
        ({ time: "12:03:18Z", severity: "Info", category: "Telemetry", source: "Telemetry Feed", pair: "—", title: "Telemetry heartbeat fresh", message: "Telemetry heartbeat is fresh in local preview.", action: "Ping feed", read: true, status: "read" }),
        ({ time: "12:03:12Z", severity: "Info", category: "Diagnostics", source: "Diagnostics", pair: "—", title: "Diagnostic bundle generated", message: "Diagnostic bundle status generated locally; secrets and environment values excluded.", action: "Open diagnostics", read: true, status: "read" }),
        ({ time: "12:03:06Z", severity: "Info", category: "Safety", source: "Safety Boundary", pair: "—", title: "Safety boundary reminder", message: "Alerts are local preview only; no OS notifications sent; no backend calls; no exchange/API calls; no order submission; no secrets read.", action: "Read safety copy", read: true, status: "read" })
    ]
    property int alertUnreadCount: 4
    property int alertCriticalCount: 2
    property int alertWarningCount: 3
    property int alertInfoCount: 10
    property string alertSelectedSeverity: "All"
    property string alertSelectedCategory: "All"
    property string alertLastEventAt: "12:04:30Z"
    property bool alertMutedPreview: false
    property bool alertSoundEnabledPreview: false
    property bool alertDesktopNotificationsPreview: false
    property var alertSelectedEvent: alertRows.length > 0 ? alertRows[0] : ({})
    property string alertEventExplanation: "Select an alert and click Explain event. Explanation is local preview only; no backend calls, no exchange/API calls, no order submission, no secrets read."
    property var alertSeverityFilters: ["All", "Critical", "Warning", "Info"]
    property var alertCategoryFilters: ["All", "Trading", "Risk", "AI", "Scanner", "Paper", "Portfolio", "Telemetry", "Diagnostics", "Safety"]
    property string alertSafetyBoundaryCopy: "LOCAL ALERT FEED ONLY • NO CLOUD SINK • NO EXTERNAL EXPORT • Alerts are local preview only • No OS notifications sent • No backend calls • No exchange/API calls • No order submission • No secrets read • Alerty działają lokalnie w preview • Brak systemowych powiadomień OS • Brak wywołań backendu • Brak połączeń giełda/API • Brak składania zleceń • Brak odczytu sekretów"

    function normalizeAlertSeverity(severity) {
        if (severity === "Critical" || severity === "critical") return "Critical"
        if (severity === "Warning" || severity === "warning") return "Warning"
        return "Info"
    }
    function normalizeAlertCategory(category) {
        var allowed = ["Trading", "Risk", "AI", "Scanner", "Paper", "Portfolio", "Telemetry", "Diagnostics", "Safety"]
        return allowed.indexOf(category) >= 0 ? category : "Trading"
    }
    function recomputeAlertCounts() {
        var unread = 0, critical = 0, warning = 0, info = 0
        for (var i = 0; i < alertRows.length; ++i) {
            var row = alertRows[i]
            if (!row.read) unread += 1
            if (row.severity === "Critical") critical += 1
            else if (row.severity === "Warning") warning += 1
            else info += 1
        }
        alertUnreadCount = unread
        alertCriticalCount = critical
        alertWarningCount = warning
        alertInfoCount = info
        alertLastEventAt = alertRows.length > 0 ? alertRows[0].time : "—"
        if (alertRows.length > 0 && (!alertSelectedEvent || !alertSelectedEvent.title))
            alertSelectedEvent = alertRows[0]
    }
    function appendPreviewAlert(severity, category, title, message, source, pair, action) {
        var normalizedSeverity = normalizeAlertSeverity(severity)
        var normalizedCategory = normalizeAlertCategory(category)
        var timestamp = previewTime(alertRows.length + decisionSequence + telemetryTick + paperSessionTicks + 1)
        var row = ({ time: timestamp, severity: normalizedSeverity, category: normalizedCategory, source: source || "UI Preview", pair: pair || "—", title: title || "Preview alert", message: message || "Local preview event", action: action || "Review event", read: false, status: "unread" })
        var rows = alertRows.slice()
        if (rows.length === 0 || rows[0].title !== row.title || rows[0].category !== row.category || rows[0].pair !== row.pair)
            rows.unshift(row)
        alertRows = rows.slice(0, previewAlertFeedLimit)
        alertSelectedEvent = alertRows[0]
        alertCenterOpen = alertCenterOpen
        recomputeAlertCounts()
        return row
    }
    function markAlertRead(index) {
        var rows = alertRows.slice()
        var visibleRows = visibleAlertRows()
        var row = visibleRows[index]
        if (!row)
            row = rows[index]
        if (!row)
            return
        for (var i = 0; i < rows.length; ++i) {
            if (rows[i].time === row.time && rows[i].title === row.title && rows[i].pair === row.pair) {
                rows[i].read = true
                rows[i].status = "read"
                alertSelectedEvent = rows[i]
                break
            }
        }
        alertRows = rows
        recomputeAlertCounts()
    }
    function markAllAlertsRead() {
        var rows = alertRows.slice()
        for (var i = 0; i < rows.length; ++i) {
            rows[i].read = true
            rows[i].status = "read"
        }
        alertRows = rows
        recomputeAlertCounts()
    }
    function clearPreviewAlerts() {
        alertRows = []
        alertSelectedEvent = ({})
        alertEventExplanation = "Alert timeline cleared locally. Alerts are local preview only; no OS notifications sent; no backend calls; no exchange/API calls; no order submission; no secrets read."
        recomputeAlertCounts()
    }
    function setAlertSeverityFilter(severity) {
        alertSelectedSeverity = alertSeverityFilters.indexOf(severity) >= 0 ? severity : "All"
    }
    function setAlertCategoryFilter(category) {
        alertSelectedCategory = alertCategoryFilters.indexOf(category) >= 0 ? category : "All"
    }
    function visibleAlertRows() {
        var out = []
        for (var i = 0; i < alertRows.length; ++i) {
            var row = alertRows[i]
            if (alertSelectedSeverity !== "All" && row.severity !== alertSelectedSeverity) continue
            if (alertSelectedCategory !== "All" && row.category !== alertSelectedCategory) continue
            out.push(row)
        }
        return out
    }
    function selectAlertEvent(index) {
        var rows = visibleAlertRows()
        var row = rows[index]
        if (!row)
            row = alertRows[index]
        if (!row)
            return
        alertSelectedEvent = row
    }
    function explainAlertEvent(index) {
        selectAlertEvent(index)
        var row = alertSelectedEvent || ({})
        var category = row.category || "Safety"
        var base = "Wyjaśnij zdarzenie / Explain event: " + (row.title || "Preview alert") + " • " + category + " • " + (row.severity || "Info") + ". " + (row.message || "Local-only preview event.")
        if (category === "AI" || category === "Scanner" || category === "Paper" || category === "Risk") {
            alertEventExplanation = base + " Explanation is local preview only and can use the existing decision explain drawer context; no backend inference, no network/API call, no order submission, no real orders, no secrets read."
            if (category === "AI" || category === "Scanner" || category === "Paper" || category === "Risk")
                openDecisionExplainDrawer(({ timestamp: row.time || previewTime(0), symbol: row.pair || selectedTerminalPair, action: row.title || "ALERT", confidence: "local", reason: row.message || base, source: row.source || category, riskReason: riskState, strategy: "Alert Center", safety: alertSafetyBoundaryCopy, status: row.status || "preview" }))
        } else {
            alertEventExplanation = base + " Local alert explanation only; no OS notifications sent, no backend calls, no exchange/API calls, no order submission, no secrets read."
        }
    }
    function toggleAlertMutePreview() {
        alertMutedPreview = !alertMutedPreview
        appendPreviewAlert("Info", "Safety", "Muted alerts changed", "Muted alerts preview flag is now " + (alertMutedPreview ? "on" : "off") + "; no OS notifications sent.", "Alert Center", "—", "Toggle mute")
    }
    function toggleAlertSoundPreview() {
        alertSoundEnabledPreview = !alertSoundEnabledPreview
        appendPreviewAlert("Info", "Safety", "Sound preview changed", "Sound preview flag is now " + (alertSoundEnabledPreview ? "on" : "off") + "; no system sound is played.", "Alert Center", "—", "Toggle sound")
    }
    function toggleDesktopNotificationsPreview() {
        alertDesktopNotificationsPreview = !alertDesktopNotificationsPreview
        appendPreviewAlert("Info", "Safety", "Desktop notification preview changed", "Desktop notification preview flag is now " + (alertDesktopNotificationsPreview ? "on" : "off") + "; No OS notifications sent.", "Alert Center", "—", "Toggle desktop preview")
    }

    function hasValue(list, value) {
        return list && list.indexOf(value) >= 0
    }

    function toggledList(list, value) {
        var copy = list ? list.slice() : []
        var idx = copy.indexOf(value)
        if (idx >= 0) copy.splice(idx, 1); else copy.push(value)
        return copy
    }

    function boundedRows(rows, limit) {
        var source = rows ? rows.slice() : []
        return source.slice(0, limit)
    }

    function parsePreviewNumber(value, fallbackValue) {
        var text = String(value || "").replace(/[^0-9.\-]/g, "")
        var parsed = Number(text)
        return isNaN(parsed) ? fallbackValue : parsed
    }

    function normalizePreviewAction(action) {
        if (action === "BLOCKED LIVE" || action === "BLOCKED PAPER PREVIEW")
            return "BLOCKED"
        return action || "NO ORDER"
    }

    function previewSessionIsActive() {
        return paperSessionStatus === "running" || paperSessionStatus === "active preview"
    }

    function previewRuntimeBoundaryOk() {
        return liveTradingDisabled && exchangeIoDisabled && orderSubmissionDisabled && !apiKeysRequired && !runtimeLoopStarted
    }

    function normalizedPaperSessionState() {
        if (paperSessionStatus === "running" || paperSessionState === "running")
            return "running"
        if (paperSessionStatus === "paused" || paperSessionState === "paused")
            return "paused"
        if (simulationStatusLabel.indexOf("running") >= 0)
            return "running"
        return "stopped"
    }

    function currentPaperSessionSnapshot() {
        return ({
            status: paperSessionStatus,
            state: paperSessionState,
            active: previewSessionIsActive(),
            normalizedState: normalizedPaperSessionState(),
            simulationStatusLabel: simulationStatusLabel,
            ticks: paperSessionTicks,
            simulatedCount: paperSimulatedCount,
            blockedCount: paperBlockedCount,
            noOrderCount: paperNoOrderCount,
            orderRows: paperOrderRows.length,
            latestOrderAction: paperOrderRows.length > 0 ? normalizePreviewAction(paperOrderRows[0].action) : "—",
            latestOrderStatus: paperOrderRows.length > 0 ? paperOrderRows[0].status : "—"
        })
    }

    function currentScannerSnapshot() {
        return ({
            status: scannerStatus,
            active: scannerActive,
            tickCount: scannerTickCount,
            rows: scannerRows.length,
            candidates: scannerCandidateCount,
            rejected: scannerRejectedCount,
            bestOpportunity: scannerBestOpportunity,
            selectedPair: scannerSelectedPair
        })
    }

    function currentGovernorSnapshot() {
        var latest = decisionPreviewRows.length > 0 ? decisionPreviewRows[0] : ({})
        return ({
            lastDecision: lastGovernorDecision,
            latestAction: latest.action ? normalizePreviewAction(latest.action) : "—",
            latestSymbol: latest.symbol || "—",
            latestReason: latest.reason || "—",
            riskProfile: riskProfile,
            riskState: riskState,
            riskBlockReason: riskBlockReason,
            decisionRows: decisionPreviewRows.length
        })
    }

    function currentPortfolioSnapshot() {
        return ({
            equity: paperEquity,
            pnl: paperPnl,
            startingEquity: portfolioStartingEquityUsd,
            orders: paperOrderRows.length,
            simulatedCount: paperSimulatedCount,
            blockedCount: paperBlockedCount,
            openPositions: paperOpenPositions.length,
            closedTrades: paperClosedTrades.length
        })
    }

    function currentAlertTelemetrySnapshot() {
        return ({
            alertRows: alertRows.length,
            unreadAlerts: alertUnreadCount,
            criticalAlerts: alertCriticalCount,
            warningAlerts: alertWarningCount,
            infoAlerts: alertInfoCount,
            latestAlertTitle: alertRows.length > 0 ? alertRows[0].title : "—",
            telemetryRows: paperTelemetryRows.length,
            latestTelemetry: paperTelemetryRows.length > 0 ? paperTelemetryRows[0].message : "—",
            telemetryTick: telemetryTick
        })
    }

    function currentPerPanelRuntimeSnapshot() {
        var latestDecision = decisionPreviewRows.length > 0 ? decisionPreviewRows[0] : ({})
        var latestOrder = paperOrderRows.length > 0 ? paperOrderRows[0] : ({})
        var latestTelemetry = paperTelemetryRows.length > 0 ? paperTelemetryRows[0].message : "—"
        var latestAlert = alertRows.length > 0 ? alertRows[0].message : "—"
        return ({
            dashboardBestScannerOpportunity: scannerBestOpportunity,
            dashboardLastGovernorDecision: lastGovernorDecision,
            aiCenterLastGovernorDecision: lastGovernorDecision,
            aiCenterRiskProfile: riskProfile,
            decisionLatestAction: latestDecision.action ? normalizePreviewAction(latestDecision.action) : "—",
            decisionLatestReason: latestDecision.reason || "—",
            terminalLatestOrderAction: latestOrder.action ? normalizePreviewAction(latestOrder.action) : "—",
            terminalLatestOrderStatus: latestOrder.status || "—",
            terminalLatestOrderReason: latestOrder.reason || "—",
            portfolioOrderCount: paperOrderRows.length,
            portfolioPnl: paperPnl,
            portfolioEquity: paperEquity,
            alertsLatestMessage: latestAlert,
            telemetryLatestMessage: latestTelemetry,
            riskBlockReason: riskBlockReason
        })
    }

    function safeColor(token, fallback) {
        if (designSystem && typeof designSystem.color === "function")
            return designSystem.color(token)
        return fallback
    }

    function pairQuote(pair) { return pair.split("/")[1] || "" }
    function pairBase(pair) { return pair.split("/")[0] || pair }
    function pairCategory(pair) {
        var base = pairBase(pair)
        if (["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE"].indexOf(base) >= 0) return "Major"
        if (["FET", "RENDER", "RNDR", "TAO", "WLD", "GRT", "AI"].indexOf(base) >= 0) return "AI"
        if (["DOGE", "SHIB", "PEPE", "WIF", "FLOKI", "BONK"].indexOf(base) >= 0) return "Meme"
        if (["UNI", "AAVE", "MKR", "LDO", "CRV", "SNX", "COMP", "YFI", "SUSHI", "1INCH", "CAKE", "PENDLE"].indexOf(base) >= 0) return "DeFi"
        if (["SOL", "AVAX", "DOT", "NEAR", "APT", "SUI", "TON", "ATOM", "SEI", "TIA", "KAS", "ALGO", "VET", "HBAR", "EGLD", "RUNE", "FTM"].indexOf(base) >= 0) return "Layer1"
        if (["ARB", "OP", "MATIC", "IMX", "STRK", "ZK", "MANTA"].indexOf(base) >= 0) return "Layer2"
        if (["BTC", "ETH", "SOL", "BNB", "XRP", "LINK", "LTC", "BCH", "DOGE", "ADA", "AVAX", "TON"].indexOf(base) >= 0) return "High volume"
        return "High volume"
    }
    function isAiCandidate(pair) { return ["BTC", "ETH", "SOL", "INJ", "TAO", "RENDER", "RNDR", "FET", "ARB", "OP", "PYTH", "WLD", "GRT"].indexOf(pairBase(pair)) >= 0 }
    function filteredMarketPairs() {
        var term = marketSearch.toLowerCase()
        var out = []
        for (var i = 0; i < previewMarketPairs.length; ++i) {
            var pair = previewMarketPairs[i]
            if (term.length > 0 && pair.toLowerCase().indexOf(term) < 0) continue
            if (marketQuoteFilter !== "All" && pairQuote(pair) !== marketQuoteFilter) continue
            if (marketCategoryFilter !== "All" && pairCategory(pair) !== marketCategoryFilter) continue
            if (marketSelectedOnly && !hasValue(selectedPairs, pair)) continue
            if (marketAiCandidatesOnly && !isAiCandidate(pair)) continue
            if (marketExcludedOnly && !hasValue(blacklistPairs, pair)) continue
            out.push(pair)
        }
        return out
    }

    function scannerUniversePairs() {
        var source = selectedPairs && selectedPairs.length >= 30 ? selectedPairs : previewMarketPairs
        if (source && source.length >= 30) return source.slice(0, previewScannerRowsLimit)
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT", "LTC/USDT", "BCH/USDT", "ATOM/USDT", "NEAR/USDT", "ARB/USDT", "OP/USDT", "INJ/USDT", "APT/USDT", "SUI/USDT", "TON/USDT", "PEPE/USDT", "WIF/USDT", "FET/USDT", "RENDER/USDT", "TAO/USDT", "UNI/USDT", "AAVE/USDT", "ETC/USDT", "FIL/USDT", "ICP/USDT"].slice(0, previewScannerRowsLimit)
    }
    function scannerTrendStrength(trend) { if (trend.indexOf("strong") >= 0) return 90; if (trend.indexOf("bull") >= 0 || trend.indexOf("bear") >= 0) return 72; if (trend.indexOf("range") >= 0) return 48; return 55 }
    function scannerScenarioAdjustments(index) { var mode = scenarioMode(simulationScenario); if (mode === "bull") return ({ trend: 10, risk: -4, volatility: 2 }); if (mode === "bear") return ({ trend: -3, risk: 12, volatility: 4 }); if (mode === "volatility") return ({ trend: 2, risk: 10, volatility: 12 }); if (mode === "sideways") return ({ trend: -8, risk: 3, volatility: -3 }); return ({ trend: index % 3, risk: 0, volatility: 0 }) }
    function scannerRecommendation(aiScore, liquidityScore, riskScore, pair) { if (hasValue(blacklistPairs, pair)) return "BLOCKED"; if (riskScore > scannerMaxRiskScore + 18 || liquidityScore < scannerMinLiquidityScore - 15) return "IGNORE"; if (aiScore >= scannerMinAiScore && liquidityScore >= scannerMinLiquidityScore && riskScore <= scannerMaxRiskScore) return "TRADE"; if (aiScore >= scannerMinAiScore - 10 && riskScore <= scannerMaxRiskScore + 10) return "WATCH"; return "IGNORE" }
    function scannerReason(row) { if (row.recommendation === "BLOCKED") return "Para jest na lokalnej blacklist; bot blokuje setup w preview."; if (row.recommendation === "TRADE") return "Wysoki AI score, płynność OK i ryzyko mieści się w progu."; if (row.recommendation === "WATCH") return "Setup obiecujący, ale bot czeka na lepszy trend lub niższy spread."; return "Słaby setup: wynik, płynność albo ryzyko nie spełnia lokalnych progów." }
    function buildScannerRow(pair, index) { var adj = scannerScenarioAdjustments(index); var quote = pairQuote(pair); var exchange = selectedExchanges && selectedExchanges.length > 0 ? selectedExchanges[0] : scannerSelectedExchange; var basePrice = pair.indexOf("BTC") >= 0 ? 68240 : (pair.indexOf("ETH") >= 0 ? 3560 : (pair.indexOf("SOL") >= 0 ? 154 : 12 + index * 3.7)); var tickShift = (scannerTickCount % 9) * (index % 2 === 0 ? 0.17 : -0.11); var price = Number((basePrice * (1 + tickShift / 100)).toFixed(basePrice > 1000 ? 2 : 4)); var volumeNumber = 850000 + ((index * 137000 + scannerTickCount * 41000) % 9200000); var spreadNumber = Number((0.03 + ((index + scannerTickCount) % 11) * 0.018).toFixed(3)); var liquidityScore = Math.max(20, Math.min(99, 58 + (index * 7) % 35 - Math.round(spreadNumber * 35))); var volatility = Math.max(8, Math.min(99, 24 + (index * 5) % 42 + adj.volatility)); var trendOptions = ["strong bull", "bull", "range", "bear", "sideways", "reversal"]; var trend = trendOptions[(index + scannerTickCount + (adj.trend > 5 ? 0 : 2)) % trendOptions.length]; var trendScore = Math.max(15, Math.min(99, scannerTrendStrength(trend) + adj.trend)); var aiScore = Math.max(30, Math.min(99, Math.round((liquidityScore * 0.38) + (trendScore * 0.44) + (isAiCandidate(pair) ? 9 : 0) + (index % 9)))); var riskScore = Math.max(12, Math.min(96, Math.round(31 + volatility * 0.48 + spreadNumber * 70 - liquidityScore * 0.18 + adj.risk))); var strategy = trend.indexOf("bull") >= 0 ? "Momentum Guard" : (trend.indexOf("range") >= 0 || trend.indexOf("sideways") >= 0 ? "Range Guard" : "Volatility Breakout Preview"); var recommendation = scannerRecommendation(aiScore, liquidityScore, riskScore, pair); var row = ({ pair: pair, exchange: exchange, price: price.toFixed(basePrice > 1000 ? 2 : 4) + " " + quote, volume: (volumeNumber / 1000000).toFixed(2) + "M", volumeValue: volumeNumber, spread: spreadNumber.toFixed(3) + "%", spreadValue: spreadNumber, liquidityScore: liquidityScore, volatility: volatility, trend: trend, trendStrength: trendScore, aiScore: aiScore, riskScore: riskScore, strategyMatch: strategy, recommendation: recommendation, reason: "", safetyState: "safe preview scanner • no real orders" }); row.reason = scannerReason(row); return row }
    function refreshScannerBuckets() { var candidates = [], rejected = [], watched = [], best = null; for (var i = 0; i < scannerRows.length; ++i) { var row = scannerRows[i]; if (row.recommendation === "TRADE" || row.aiScore >= scannerMinAiScore) candidates.push(row); if (row.recommendation === "IGNORE" || row.recommendation === "BLOCKED") rejected.push(row); if (hasValue(scannerWatchlistPairs, row.pair)) watched.push(row); if (row.recommendation !== "BLOCKED" && (!best || row.aiScore - row.riskScore > best.aiScore - best.riskScore)) best = row } scannerAiCandidateRows = candidates; scannerRejectedRows = rejected; scannerWatchlistRows = watched; scannerUniverseCount = scannerRows.length; scannerCandidateCount = candidates.length; scannerRejectedCount = rejected.length; scannerWatchlistCount = scannerWatchlistPairs.length; scannerBestOpportunity = best ? best.pair + " • AI " + best.aiScore + " • " + best.recommendation : "—" }
    function rebuildMarketScannerRows() { var pairs = scannerUniversePairs(); var rows = []; for (var i = 0; i < pairs.length; ++i) rows.push(buildScannerRow(pairs[i], i)); scannerRows = rows; refreshScannerBuckets(); if (!scannerSelectedPair && rows.length > 0) scannerSelectedPair = rows[0].pair; explainScannerCandidate(scannerSelectedPair) }
    function scannerRowByPair(pair) { for (var i = 0; i < scannerRows.length; ++i) if (scannerRows[i].pair === pair) return scannerRows[i]; return scannerRows.length > 0 ? scannerRows[0] : null }
    function sortScannerRows(rows) { var copy = rows.slice(); copy.sort(function(a, b) { if (scannerSortMode === "Risk score") return a.riskScore - b.riskScore; if (scannerSortMode === "Liquidity") return b.liquidityScore - a.liquidityScore; if (scannerSortMode === "Volume") return b.volumeValue - a.volumeValue; if (scannerSortMode === "Spread") return a.spreadValue - b.spreadValue; if (scannerSortMode === "Trend strength") return b.trendStrength - a.trendStrength; return b.aiScore - a.aiScore }); return copy }
    function visibleScannerRows() { var rows = []; for (var i = 0; i < scannerRows.length; ++i) { var row = scannerRows[i]; if (scannerFilterMode === "AI candidates" && row.aiScore < scannerMinAiScore) continue; if (scannerFilterMode === "Trade candidates" && row.recommendation !== "TRADE") continue; if (scannerFilterMode === "Watchlist" && !hasValue(scannerWatchlistPairs, row.pair)) continue; if (scannerFilterMode === "Rejected" && row.recommendation !== "IGNORE") continue; if (scannerFilterMode === "Blocked" && row.recommendation !== "BLOCKED") continue; if (scannerFilterMode === "High liquidity" && row.liquidityScore < 80) continue; if (scannerFilterMode === "Low risk" && row.riskScore > 35) continue; if (scannerFilterMode === "Top score" && row.aiScore < 85) continue; rows.push(row) } return sortScannerRows(rows) }
    function startMarketScannerPreview() { scannerActive = true; scannerStatus = "scanning"; if (scannerRows.length === 0) rebuildMarketScannerRows(); runMarketScannerTick() }
    function pauseMarketScannerPreview() { scannerActive = false; scannerStatus = "paused" }
    function stopMarketScannerPreview() { scannerActive = false; scannerStatus = "stopped" }
    function resetMarketScannerPreview() { scannerActive = false; scannerStatus = "safe preview"; scannerTickCount = 0; scannerLastScanAt = "not scanned"; scannerFilterMode = "All"; scannerSortMode = "AI score"; scannerMinAiScore = 70; scannerMinLiquidityScore = 60; scannerMaxRiskScore = 55; rebuildMarketScannerRows() }
    function runMarketScannerTick() {
        scannerTickCount += 1
        scannerLastScanAt = previewTime(scannerTickCount)
        if (scannerStatus === "stopped")
            scannerStatus = "safe preview"
        rebuildMarketScannerRows()
        var rows = visibleScannerRows()
        var bestRow = rows.length > 0 ? rows[0] : scannerRowByPair(scannerSelectedPair)
        if (!bestRow)
            return
        var rejected = bestRow.recommendation === "IGNORE" || bestRow.recommendation === "BLOCKED"
        appendPaperDecision("SCANNER", "Market Scanner local candidate " + bestRow.pair + " • " + bestRow.recommendation + " • " + bestRow.reason, bestRow.pair, "scanner preview / no order")
        appendPaperTelemetry("scanner tick " + scannerTickCount + " • candidates " + scannerCandidateCount + " • rejected " + scannerRejectedCount + " • no network/API calls")
        appendPreviewAlert(rejected ? "Warning" : "Info", "Scanner", rejected ? "Scanner rejected setup" : "Scanner candidate found", bestRow.pair + " • " + bestRow.recommendation + " • " + bestRow.reason, "Market Scanner", bestRow.pair, "Explain candidate")
    }
    function runMarketScannerBurst(count) { var ticks = Math.max(0, Number(count) || 0); for (var i = 0; i < ticks; ++i) runMarketScannerTick() }
    function selectScannerPair(pair) { var requestedPair = pair && pair.length > 0 ? pair : scannerSelectedPair; var row = scannerRowByPair(requestedPair); scannerSelectedPair = row ? row.pair : requestedPair; if (scannerSelectedPair && scannerSelectedPair.length > 0) { var selectedCopy = selectedPairs ? selectedPairs.slice() : []; var existingIndex = selectedCopy.indexOf(scannerSelectedPair); if (existingIndex >= 0) selectedCopy.splice(existingIndex, 1); selectedPairs = [scannerSelectedPair].concat(selectedCopy); whitelistPairs = selectedPairs.slice() } explainScannerCandidate(scannerSelectedPair); setTerminalPairFromSource(scannerSelectedPair, "selectScannerPair") }
    function addScannerPairToWatchlist(pair) { if (pair && !hasValue(scannerWatchlistPairs, pair)) scannerWatchlistPairs = scannerWatchlistPairs.concat([pair]); refreshScannerBuckets() }
    function removeScannerPairFromWatchlist(pair) { var copy = scannerWatchlistPairs ? scannerWatchlistPairs.slice() : []; var idx = copy.indexOf(pair); if (idx >= 0) copy.splice(idx, 1); scannerWatchlistPairs = copy; refreshScannerBuckets() }
    function blacklistScannerPair(pair) { if (pair && !hasValue(blacklistPairs, pair)) blacklistPairs = blacklistPairs.concat([pair]); rebuildMarketScannerRows() }
    function setScannerFilterMode(mode) { scannerFilterMode = mode && mode.length > 0 ? mode : "All" }
    function setScannerSortMode(mode) { scannerSortMode = mode && mode.length > 0 ? mode : "AI score" }
    function setScannerThreshold(name, value) { var numberValue = Math.max(0, Math.min(100, Number(value) || 0)); if (name === "minAiScore") scannerMinAiScore = numberValue; else if (name === "minLiquidityScore") scannerMinLiquidityScore = numberValue; else if (name === "maxRiskScore") scannerMaxRiskScore = numberValue; rebuildMarketScannerRows() }
    function explainScannerCandidate(pair) { var row = scannerRowByPair(pair); if (!row) { scannerExplanation = "Brak lokalnych wierszy skanera; uruchom scan tick preview."; return } scannerSelectedPair = row.pair; var riskPass = row.riskScore <= scannerMaxRiskScore ? "Ryzyko przepuszcza setup w lokalnym progu." : "Ryzyko nie przepuszcza setupu: score jest powyżej progu."; var paperAction = row.recommendation === "TRADE" ? "w paper preview bot pokazałby PAPER BUY/SELL jako symulację, bez realnego zlecenia" : (row.recommendation === "WATCH" ? "bot dodałby parę do obserwacji i czekał na kolejny tick" : "bot nie składałby żadnego zlecenia i zostawiłby parę poza akcją"); scannerExplanation = "Para " + row.pair + ": co przemawia za — AI score " + row.aiScore + ", płynność " + row.liquidityScore + ", trend " + row.trend + ". Co przemawia przeciw — risk score " + row.riskScore + ", zmienność " + row.volatility + ", spread " + row.spread + ". " + riskPass + " Pasująca strategia: " + row.strategyMatch + ". Rekomendacja: " + row.recommendation + ". Powód: " + row.reason + " W paper preview " + paperAction + ". Live trading disabled, order submission disabled." }

    function decisionValue(row, key, fallback) { if (!row) return fallback; var value = row[key]; return value === undefined || value === null || String(value).length === 0 ? fallback : value }
    function normalizedDecisionRow(row) {
        if (!row || typeof row !== "object") return decisionPreviewRows.length > 0 ? decisionPreviewRows[0] : ({})
        return row
    }
    function buildDecisionInputSnapshot(row) {
        row = normalizedDecisionRow(row)
        return [
            ({ label: "Market data snapshot", value: "local preview • " + decisionValue(row, "symbol", decisionValue(row, "pair", scannerSelectedPair)) }),
            ({ label: "AI score", value: decisionValue(row, "aiScore", Math.round(Number(decisionValue(row, "confidence", "0.70")) * 100)) }),
            ({ label: "Risk score", value: decisionValue(row, "riskScore", riskState + " • " + riskProfile) }),
            ({ label: "Liquidity score", value: decisionValue(row, "liquidityScore", "n/a for governor row") }),
            ({ label: "Spread", value: decisionValue(row, "spread", "preview spread check only") }),
            ({ label: "Strategy match", value: decisionValue(row, "strategyMatch", decisionValue(row, "strategy", "Strategy governor")) })
        ]
    }
    function buildDecisionRiskChecks(row) {
        row = normalizedDecisionRow(row)
        var riskScoreValue = Number(decisionValue(row, "riskScore", 0))
        var action = decisionValue(row, "action", decisionValue(row, "recommendation", "HOLD"))
        return [
            ({ label: "Risk profile applied", status: riskProfile, detail: "Profil ryzyka obowiązuje tylko w lokalnym preview." }),
            ({ label: "Kill-switch", status: riskLocked ? "ACTIVE / blokada" : "inactive", detail: riskLocked ? "Bot blokuje akcję i nie zmienia PnL/equity." : "Brak blokady awaryjnej w tym ticku." }),
            ({ label: "Risk lock", status: riskState, detail: decisionValue(row, "riskReason", "Risk guard checked local thresholds.") }),
            ({ label: "Confidence floor", status: decisionValue(row, "confidence", "local score"), detail: "Decyzja musi przekroczyć próg pewności profilu." }),
            ({ label: "Order route check", status: "disabled", detail: "Trasa zlecenia jest zablokowana; brak realnego zlecenia." }),
            ({ label: "Live mode check", status: liveTradingDisabled ? "disabled" : "blocked by preview", detail: "Live mode nie jest dostępny w UI Preview." }),
            ({ label: "Risk score threshold", status: riskScoreValue > scannerMaxRiskScore ? "watch/block" : "pass", detail: action.indexOf("BLOCK") >= 0 ? "Risk guard zablokował akcję." : "Ryzyko ocenione deterministycznie lokalnie." })
        ]
    }
    function buildDecisionAuditRows(row) {
        row = normalizedDecisionRow(row)
        var action = decisionValue(row, "action", decisionValue(row, "recommendation", "HOLD"))
        return [
            ({ step: "Market data snapshot: local preview", result: "Zebrano lokalny snapshot wejść dla " + decisionValue(row, "symbol", decisionValue(row, "pair", "—")) }),
            ({ step: "AI score computed: local deterministic preview", result: "Score policzony lokalnie, bez backendowej inferencji AI." }),
            ({ step: "Risk profile applied", result: riskProfile + " • " + riskState }),
            ({ step: "Risk guard result", result: decisionValue(row, "riskReason", "Risk checks evaluated locally.") }),
            ({ step: "Order route check", result: "disabled • brak składania zleceń" }),
            ({ step: "Live mode check", result: "disabled • safe preview only" }),
            ({ step: "Paper action decision", result: action + " • " + selectedDecisionPaperImpact }),
            ({ step: "Financial impact", result: selectedDecisionPaperImpact }),
            ({ step: "Safety boundary", result: selectedDecisionSafetySummary })
        ]
    }
    function buildDecisionAlternatives(row) {
        row = normalizedDecisionRow(row)
        var selectedPair = decisionValue(row, "symbol", decisionValue(row, "pair", scannerSelectedPair))
        var sourceRows = scannerRows && scannerRows.length > 0 ? sortScannerRows(scannerRows).slice(0, 8) : decisionPreviewRows
        var alternatives = []
        for (var i = 0; i < sourceRows.length && alternatives.length < 3; ++i) {
            var alt = sourceRows[i]
            var altPair = decisionValue(alt, "pair", decisionValue(alt, "symbol", "—"))
            if (altPair === selectedPair) continue
            alternatives.push(({ pair: altPair, score: decisionValue(alt, "aiScore", decisionValue(alt, "confidence", "—")), reason: "Nie wygrała: " + decisionValue(alt, "reason", "niższy score, większe ryzyko albo słabsza płynność w lokalnym preview.") }))
        }
        return alternatives
    }
    function explainDecisionRow(row) {
        row = normalizedDecisionRow(row)
        var action = decisionValue(row, "action", decisionValue(row, "recommendation", "HOLD"))
        var confidence = decisionValue(row, "confidence", decisionValue(row, "aiScore", "0.70"))
        var aiScore = decisionValue(row, "aiScore", Math.round(Number(confidence) * 100))
        var riskScore = decisionValue(row, "riskScore", "local risk guard")
        var source = decisionValue(row, "source", row.pair ? "Scanner" : (action.indexOf("PAPER") >= 0 ? "Paper Terminal" : (action.indexOf("BLOCK") >= 0 ? "Risk" : "Governor")))
        if (action.indexOf("BLOCK") >= 0 || riskLocked)
            return "Bot zablokował akcję, bo kill-switch lub risk lock jest aktywny albo trasa live jest niedostępna. Zmieniono tylko logi i telemetrykę, bez zmiany PnL/equity. Score AI: " + aiScore + ", risk score: " + riskScore + ". Profil ryzyka " + riskProfile + " wymaga bezpieczniejszego setupu."
        if (action.indexOf("TRADE") >= 0 || action.indexOf("PAPER") >= 0)
            return "Bot wskazał paper action, bo lokalny score AI jest wystarczający, płynność wygląda poprawnie, a risk guard mieści się w profilu " + riskProfile + ". To nadal tylko Paper Preview: brak połączenia z giełdą i brak realnych zleceń. Źródło decyzji: " + source + "."
        if (action.indexOf("WATCH") >= 0 || action === "SCANNER")
            return "Bot dodał parę do obserwacji, bo trend jest dobry, ale ryzyko, zmienność albo spread wymagają kolejnego lokalnego ticka. Paper Preview nie zmienia PnL/equity i nie wysyła zleceń."
        return "Bot nie kupił, bo score AI był za niski, spread lub ryzyko były zbyt wysokie, a profil ryzyka wymaga wyższej pewności. Decyzja jest lokalnym preview; brak fill i brak zmiany finansowej."
    }
    function openDecisionExplainDrawer(row) {
        row = normalizedDecisionRow(row)
        selectedDecisionId = decisionValue(row, "timestamp", previewTime(decisionSequence)) + " • " + decisionValue(row, "symbol", decisionValue(row, "pair", scannerSelectedPair))
        selectedDecisionPair = decisionValue(row, "symbol", decisionValue(row, "pair", scannerSelectedPair))
        selectedDecisionAction = decisionValue(row, "action", decisionValue(row, "recommendation", "HOLD"))
        selectedDecisionSource = decisionValue(row, "source", row.pair ? "Scanner" : (selectedDecisionAction.indexOf("PAPER") >= 0 ? "Paper Terminal" : (selectedDecisionAction.indexOf("BLOCK") >= 0 ? "Risk" : "Governor")))
        selectedDecisionConfidence = decisionValue(row, "confidence", decisionValue(row, "aiScore", "0.70"))
        selectedDecisionRiskState = riskProfile + " • " + riskState + " • kill-switch " + (riskLocked ? "active" : "inactive")
        selectedDecisionStrategy = decisionValue(row, "strategy", decisionValue(row, "strategyMatch", "Strategy governor"))
        selectedDecisionPaperImpact = selectedDecisionAction.indexOf("PAPER") >= 0 || selectedDecisionAction === "TRADE" ? "paper preview event/log only • możliwa lokalna pozycja paper, brak realnego fill" : "brak fill / brak zmiany finansowej"
        selectedDecisionReason = explainDecisionRow(row)
        selectedDecisionInputSnapshot = buildDecisionInputSnapshot(row)
        selectedDecisionRiskChecks = buildDecisionRiskChecks(row)
        selectedDecisionAlternatives = buildDecisionAlternatives(row)
        selectedDecisionLineageLinks = [
            ({ label: "decision source", value: selectedDecisionSource }),
            ({ label: "paper terminal", value: selectedDecisionAction.indexOf("PAPER") >= 0 ? "linked local paper row" : "no paper fill" }),
            ({ label: "telemetry", value: telemetryHeartbeat + " • local audit event" })
        ]
        selectedDecisionAuditRows = buildDecisionAuditRows(row)
        decisionExplainDrawerOpen = true
        if (decisionExplainDrawer) decisionExplainDrawer.open()
        appendPaperTelemetry("decision explainability audit opened for " + selectedDecisionPair + " • local preview only • no backend AI inference")
    }
    function closeDecisionExplainDrawer() {
        decisionExplainDrawerOpen = false
        if (decisionExplainDrawer) decisionExplainDrawer.close()
    }
    function explainScannerCandidateDecision(pair) {
        var row = scannerRowByPair(pair)
        if (!row) {
            rebuildMarketScannerRows()
            row = scannerRowByPair(pair)
        }
        if (row) {
            explainScannerCandidate(row.pair)
            openDecisionExplainDrawer(row)
        }
    }
    function explainPaperOrderDecision(order) {
        var row = (!order || typeof order !== "object") ? (paperOrderRows.length > 0 ? paperOrderRows[0] : null) : order
        if (!row) row = ({ pair: selectedTerminalPair, action: "NO ORDER", confidence: "0.00", reason: "No paper order row selected.", source: "Paper Terminal" })
        row.source = "Paper Terminal"
        openDecisionExplainDrawer(row)
    }


    function currentTerminalPair() { return selectedPairs && selectedPairs.length > 0 ? selectedPairs[0] : "BTC/USDT" }
    function preferredTerminalPair() {
        if (selectedTerminalPair && selectedTerminalPair.length > 0 && hasValue(selectedPairs, selectedTerminalPair))
            return selectedTerminalPair
        if (scannerSelectedPair && scannerSelectedPair.length > 0 && (!selectedPairs || selectedPairs.length === 0 || hasValue(selectedPairs, scannerSelectedPair)))
            return scannerSelectedPair
        if (selectedPairs && selectedPairs.length > 0)
            return selectedPairs[0]
        return "BTC/USDT"
    }
    function setTerminalPairFromSource(pair, writer) {
        selectedTerminalPair = pair && pair.length > 0 ? pair : preferredTerminalPair()
        selectedTerminalPairLastWriter = writer && writer.length > 0 ? writer : "setTerminalPairFromSource"
    }
    function setTerminalPair(pair) { setTerminalPairFromSource(pair, "setTerminalPair") }
    function setTerminalSide(side) { terminalSide = side === "SELL" ? "SELL" : "BUY" }
    function setTerminalOrderType(type) { terminalOrderType = type === "MARKET" ? "MARKET" : "LIMIT" }
    function setTerminalTimeframe(timeframe) { terminalTimeframe = timeframe && timeframe.length > 0 ? timeframe : "15m" }
    function terminalPairCandidates() {
        var sourcePairs = selectedPairs && selectedPairs.length > 0 ? selectedPairs : previewMarketPairs
        var term = terminalPairSearch.toLowerCase()
        var out = []
        for (var i = 0; i < sourcePairs.length && out.length < 16; ++i) {
            var pair = sourcePairs[i]
            if (term.length > 0 && pair.toLowerCase().indexOf(term) < 0)
                continue
            out.push(pair)
        }
        if (out.length === 0)
            out.push("BTC/USDT")
        return out
    }
    function recalcTerminalTotal() {
        var price = Number(terminalPrice)
        var amount = Number(terminalAmount)
        if (!isNaN(price) && !isNaN(amount))
            terminalTotal = (price * amount).toFixed(2)
    }
    function setTerminalPrice(price) { terminalPrice = String(price); recalcTerminalTotal() }
    function setTerminalAmount(amount) { terminalAmount = String(amount); recalcTerminalTotal() }
    function applyTerminalPercent(percent) {
        var baseAmount = 0.02
        var pct = Number(percent)
        if (isNaN(pct))
            pct = 10
        terminalAmount = (baseAmount * pct / 100).toFixed(4)
        recalcTerminalTotal()
        var logCopy = terminalLogRows.slice()
        logCopy.unshift(({ time: previewTime(logCopy.length + 1), message: "Applied local paper percent chip " + pct + "% • no real orders" }))
        terminalLogRows = logCopy.slice(0, previewTerminalLogLimit)
    }
    function syncPaperBridgeState() {
        paperSessionState = paperSessionStatus
        paperTicks = paperSessionTicks
        previewEquity = paperEquity
        previewPnl = paperPnl
        blockedOrdersCount = paperBlockedCount
        noOrderCount = paperNoOrderCount
        simulatedOrdersCount = paperSimulatedCount
        paperOrdersCount = paperOrderRows.length
        paperOrdersPreview = paperOrderRows.slice(0, previewPaperOrderFeedLimit)
        openPaperPositions = paperOpenPositions.slice(0, previewOpenPositionFeedLimit)
        closedPaperTrades = paperClosedTrades.slice(0, previewClosedTradeFeedLimit)
        telemetryRows = paperTelemetryRows.slice(0, previewTelemetryFeedLimit)
        syncPortfolioPerformanceState()
    }
    function markSettingsDirty() { settingsDirty = true }
    function setAppModePreview(mode) { appModePreview = mode; markSettingsDirty(); if (mode === "Paper Preview") paperSessionStatus = paperSessionStatus === "stopped" ? "ready" : paperSessionStatus }
    function setBaseCurrency(currency) { baseCurrency = currency; markSettingsDirty() }
    function setUiDensity(density) { uiDensity = density; markSettingsDirty() }
    function setThemeModePreview(mode) { themeModePreview = mode; markSettingsDirty() }
    function setDefaultPreviewExchange(exchange) { defaultPreviewExchange = exchange; selectedExchanges = [exchange]; markSettingsDirty() }
    function setDefaultTerminalPair(pair) { defaultTerminalPair = pair; terminalSelectedPair = pair; if (selectedPairs.indexOf(pair) < 0) selectedPairs = selectedPairs.concat([pair]); markSettingsDirty() }
    function setDefaultRiskProfile(profile) { defaultRiskProfile = profile; if (profile !== "AI Recommended") applyRiskProfile(profile); else applyAiRecommendedRiskProfile(); markSettingsDirty() }
    function applyPreviewSettings() { settingsDirty = false; settingsLastUpdatedAt = previewTime(telemetryTick + simulationTickCount + paperSessionTicks + 1); appendPreviewAlert("Info", "Settings", "Preview settings applied", settingsSafetySummary, "Settings", defaultTerminalPair, "Open Settings") }
    function resetPreviewSettings() { appModePreview = "Demo Preview"; baseCurrency = "USDT"; uiDensity = "Comfortable"; themeModePreview = "Dark preview"; defaultPreviewExchange = "Paper Preview Catalog"; defaultTerminalPair = "BTC/USDT"; defaultRiskProfile = "Balanced"; settingsDirty = true; settingsLastUpdatedAt = "reset local preview settings" }
    function resetLocalPreviewState() { resetPreviewSettings(); resetPaperPreview(); firstRunWizardVisible = false; onboardingStep = 1; onboardingCompletedPreview = false; appendPreviewAlert("Warning", "Settings", "Local preview state reset", settingsSafetySummary, "Settings", "—", "Open Settings") }
    function startOnboardingPreview() { firstRunWizardVisible = true; onboardingStep = 1; onboardingCompletedPreview = false; currentPanelId = "settingsPanel" }
    function nextOnboardingStep() { onboardingStep = Math.min(onboardingStep + 1, onboardingSteps.length) }
    function previousOnboardingStep() { onboardingStep = Math.max(onboardingStep - 1, 1) }
    function completeOnboardingPreview() { onboardingCompletedPreview = true; firstRunWizardVisible = false; appModePreview = appModePreview === "Demo Preview" ? "Paper Preview" : appModePreview; applyPreviewSettings(); showPanel("sidePanel") }
    function skipOnboardingPreview() { onboardingCompletedPreview = true; firstRunWizardVisible = false; settingsLastUpdatedAt = "onboarding skipped local-only" }

    function setLanguage(lang) {
        for (var i = 0; i < languageOptions.length; ++i) {
            if (languageOptions[i].code === lang) {
                currentLanguage = lang
                return
            }
        }
    }
    function trText(key) {
        var langTable = translationDictionary[currentLanguage] || translationDictionary["PL"]
        return langTable && langTable[key] ? langTable[key] : key
    }
    function previewT(key) { return trText(key) }
    function tooltipText(key) { return previewTooltips[key] || key }
    function glossaryDescription(term) {
        var table = glossaryDescriptions[currentLanguage] || glossaryDescriptions["PL"]
        return table && table[term] ? table[term] : term
    }
    function languageDisplay(code) {
        for (var i = 0; i < languageOptions.length; ++i) {
            if (languageOptions[i].code === code)
                return languageOptions[i].display
        }
        return "🌐 " + code
    }
    function groupedNumber(value) {
        var fixed = Math.abs(Number(value)).toFixed(2)
        var parts = fixed.split(".")
        var whole = parts[0]
        var grouped = ""
        while (whole.length > 3) {
            grouped = " " + whole.slice(whole.length - 3) + grouped
            whole = whole.slice(0, whole.length - 3)
        }
        return whole + grouped + "." + parts[1]
    }
    function formatMoney(value, currency) {
        var numeric = Number(value)
        var sign = numeric < 0 ? "-" : ""
        return sign + groupedNumber(numeric) + " " + currency
    }
    function formatUsd(value) {
        var numeric = Number(value)
        var sign = numeric >= 0 ? "+" : "-"
        return sign + groupedNumber(numeric) + " USD"
    }
    function portfolioCardDescription(field) {
        if (field === "fiat") return portfolioFiatAccountLabel
        if (field === "crypto") return portfolioCryptoAccountLabel
        if (field === "trading") return formatMoney(portfolioTradingEquityUsdt, "USDT") + " • equity preview"
        if (field === "available") return formatMoney(portfolioAvailableBalanceUsd, portfolioBaseCurrency)
        if (field === "positions") return formatMoney(portfolioInPositionsUsd, portfolioBaseCurrency)
        if (field === "reserved") return formatMoney(portfolioReservedMarginUsd, portfolioBaseCurrency)
        if (field === "lastCycle") return formatUsd(portfolioLastCyclePnlUsd)
        if (field === "paperSession") return formatUsd(paperPnl) + " • equity " + formatMoney(paperEquity, "USD")
        if (field === "session") return formatUsd(portfolioSessionPnlUsd)
        if (field === "allTime") return formatUsd(portfolioAllTimePnlUsd)
        if (field === "winRate") return portfolioWinRate
        if (field === "tradeCount") return String(portfolioTradeCount)
        if (field === "fees") return formatMoney(portfolioFeesUsd, portfolioBaseCurrency)
        if (field === "drawdown") return portfolioMaxDrawdown
        if (field === "bestPair") return portfolioBestPair
        if (field === "worstPair") return portfolioWorstPair
        return "preview"
    }
    function recomputePortfolioTotals() {
        portfolioNetPnlUsd = Number((portfolioRealizedPnlUsd + portfolioUnrealizedPnlUsd - portfolioFeesUsd - portfolioFundingOtherCostsUsd).toFixed(2))
        portfolioAllTimePnlUsd = portfolioNetPnlUsd
        portfolioTotalEquityUsd = Number((portfolioStartingEquityUsd + portfolioAllTimePnlUsd).toFixed(2))
        portfolioTotalEquityPln = Number((portfolioTotalEquityUsd * 3.945).toFixed(2))
        portfolioRoiPercent = Number(((portfolioAllTimePnlUsd / portfolioStartingEquityUsd) * 100).toFixed(2))
        portfolioAvailableBalanceUsd = Number((portfolioTotalEquityUsd - portfolioInPositionsUsd - portfolioReservedMarginUsd).toFixed(2))
        portfolioTradingEquityUsdt = portfolioTotalEquityUsd
        portfolioTradingAvailableUsdt = portfolioAvailableBalanceUsd
        portfolioFiatAccountLabel = formatMoney(portfolioTotalEquityUsd, "USD") + " / " + formatMoney(portfolioTotalEquityPln, "PLN")
    }
    function normalizedPortfolioRange(range) {
        if (range === "24h") return "1d"
        if (range === "30d") return "1m"
        if (range === "All") return "all"
        if (range === "Custom") return "custom"
        return range
    }
    function applyPortfolioSnapshot(range, snapshot) {
        var normalizedRange = normalizedPortfolioRange(range)
        portfolioSelectedRange = normalizedRange
        portfolioRangeLabel = normalizedRange === "custom" ? "Zakres własny: preview • " + portfolioCustomFrom + " → " + portfolioCustomTo : "Zakres: " + normalizedRange + " preview • filtry raportowe nie zmieniają aktywnej sesji Paper"
        portfolioRealizedPnlUsd = Number(snapshot.realized)
        portfolioUnrealizedPnlUsd = Number(snapshot.unrealized)
        portfolioFeesUsd = Number(snapshot.fees)
        portfolioFundingOtherCostsUsd = Number(snapshot.fundingOtherCosts)
        portfolioSessionPnlUsd = Number(snapshot.sessionPnl)
        portfolioLastCyclePnlUsd = Number(snapshot.lastCyclePnl)
        portfolioLastCycleTradesCount = Number(snapshot.lastCycleTrades || portfolioLastCycleTradesCount)
        portfolioLastCycleWinners = Number(snapshot.lastCycleWinners || portfolioLastCycleWinners)
        portfolioLastCycleLosers = Number(snapshot.lastCycleLosers || portfolioLastCycleLosers)
        portfolioLastCycleFeesUsd = Number(snapshot.lastCycleFees || portfolioLastCycleFeesUsd)
        portfolioLastCycleNetUsd = Number(snapshot.lastCycleNet || (portfolioLastCyclePnlUsd - portfolioLastCycleFeesUsd))
        portfolioTradeCount = Number(snapshot.tradeCount)
        portfolioWinRate = snapshot.winRate
        portfolioMaxDrawdown = snapshot.maxDrawdown
        portfolioBestPair = snapshot.bestPair
        portfolioWorstPair = snapshot.worstPair
        portfolioCycleRows = snapshot.cycleRows.slice(0, 12)
        recomputePortfolioTotals()
    }
    function setPortfolioTimeRange(range) {
        var normalizedRange = normalizedPortfolioRange(range)
        var snapshot = portfolioRangeSnapshots[normalizedRange] || portfolioRangeSnapshots["1d"]
        applyPortfolioSnapshot(normalizedRange, snapshot)
        appendPreviewAlert("Info", "Portfolio", "Portfolio range changed", "Portfolio report range changed to " + normalizedRange + "; paper state was not mutated.", "Portfolio Performance", "—", "Review report")
    }
    function applyPortfolioCustomRange(fromValue, toValue) {
        portfolioCustomFrom = fromValue && fromValue.length > 0 ? fromValue : portfolioCustomFrom
        portfolioCustomTo = toValue && toValue.length > 0 ? toValue : portfolioCustomTo
        setPortfolioTimeRange("custom")
    }
    function syncPortfolioPerformanceState() {
        recomputePortfolioTotals()
    }
    function appendTerminalLog(message) {
        var logCopy = terminalLogRows.slice()
        logCopy.unshift(({ time: previewTime(logCopy.length + paperSessionTicks + 1), message: message }))
        terminalLogRows = logCopy.slice(0, previewTerminalLogLimit)
    }
    function appendPaperTelemetry(message) {
        telemetryTick += 1
        telemetryHeartbeat = previewTime(telemetryTick + paperSessionTicks)
        telemetryFreshness = "freshness status: paper bridge event #" + telemetryTick + " • " + telemetryHeartbeat
        var rows = paperTelemetryRows.slice()
        rows.unshift(({ timestamp: telemetryHeartbeat, message: message + " • local-only paper bridge/state • exchange I/O disabled • runtime loop not started" }))
        paperTelemetryRows = rows.slice(0, previewTelemetryFeedLimit)
        telemetryRows = paperTelemetryRows.slice(0, previewTelemetryFeedLimit)
    }
    function appendPaperDecision(action, reason, pair, status) {
        decisionSequence += 1
        var symbol = pair && pair.length > 0 ? pair : currentTerminalPair()
        var strategy = activeStrategies.length > 0 ? activeStrategies[decisionSequence % activeStrategies.length] : "Strategy governor"
        var confidence = action === "BLOCKED" || action === "NO ORDER" ? "0.00" : (0.70 + ((decisionSequence % 7) * 0.021)).toFixed(2)
        var normalizedAction = normalizePreviewAction(action)
        var row = ({ timestamp: previewTime(decisionSequence + paperSessionTicks), symbol: symbol, action: normalizedAction, confidence: confidence, reason: reason, riskReason: riskBlockReason, strategy: strategy, safety: "Paper Preview only • Live trading disabled • Exchange I/O disabled • Order submission disabled", paperState: paperSessionStatus, status: status, orderEvent: status && status.indexOf("paper simulated") >= 0 ? "Paper Terminal order event for " + symbol : "no real order emitted" })
        var rows = decisionPreviewRows.slice()
        if (rows.length === 0 || rows[0].timestamp !== row.timestamp || rows[0].action !== row.action || rows[0].symbol !== row.symbol)
            rows.unshift(row)
        decisionPreviewRows = rows.slice(0, previewDecisionFeedLimit)
        lastGovernorDecision = symbol + " " + normalizedAction + " • confidence " + confidence + " • " + reason
        return row
    }
    function validatePaperOrderPreview() {
        var price = Number(terminalPrice)
        var amount = Number(terminalAmount)
        var total = Number(terminalTotal)
        if (isNaN(amount) || amount <= 0)
            return ({ ok: false, action: "NO ORDER", status: "no order", reason: "Local validation failed: amount must be > 0." })
        if (terminalOrderType === "LIMIT" && (isNaN(price) || price <= 0))
            return ({ ok: false, action: "NO ORDER", status: "no order", reason: "Local validation failed: price must be > 0 for LIMIT." })
        if (isNaN(total) || total <= 0) {
            total = (isNaN(price) ? 0 : price) * amount
            terminalTotal = total.toFixed(2)
        }
        if (total > paperEquity)
            return ({ ok: false, action: "NO ORDER", status: "no order", reason: "Local validation failed: total exceeds paper balance." })
        var action = terminalSide === "SELL" ? "PAPER SELL" : "PAPER BUY"
        var scannerRow = scannerRowByPair(selectedTerminalPair || currentTerminalPair())
        var riskGate = evaluateLocalRiskGate(action, scannerRow, 0.86, total)
        riskState = riskProfile + " preview • daily loss limit active • paper actions evaluated by local risk gate • live blocked • " + riskBlockReason
        if (riskGate.blocked)
            return ({ ok: false, action: "BLOCKED", status: "blocked", reason: riskGate.reason })
        return ({ ok: true, action: action, status: "paper simulated / no real order", reason: "Paper simulated order accepted locally; " + riskGate.reason + " Live trading disabled and order submission disabled." })
    }
    function updatePaperEquityPreview(status, total) {
        var delta = status === "blocked" ? 0 : (terminalSide === "BUY" ? 7.25 : 5.10)
        paperPnl = Number((paperPnl + delta).toFixed(2))
        paperEquity = Number((portfolioStartingEquityUsd + paperPnl).toFixed(2))
        previewPnl = paperPnl
        previewEquity = paperEquity
        syncPortfolioPerformanceState()
    }
    function updatePaperPositionPreview(event) {
        if (event.status === "blocked" || event.status === "no order")
            return
        if (event.action === "PAPER BUY") {
            var positions = paperOpenPositions.slice()
            positions.unshift(({ pair: event.pair, side: "paper long", size: event.amount, pnl: paperPnl >= 0 ? "+" + paperPnl.toFixed(2) : paperPnl.toFixed(2), label: "paper simulated" }))
            paperOpenPositions = positions.slice(0, previewOpenPositionFeedLimit)
        } else {
            var trades = paperClosedTrades.slice()
            trades.unshift(({ pair: event.pair, side: "paper sell", pnl: paperPnl >= 0 ? "+" + paperPnl.toFixed(2) : paperPnl.toFixed(2), label: "paper simulated" }))
            paperClosedTrades = trades.slice(0, previewClosedTradeFeedLimit)
        }
    }
    function submitLocalPaperOrder() {
        setTerminalPair(selectedTerminalPair || currentTerminalPair())
        recalcTerminalTotal()
        paperSessionStatus = paperSessionStatus === "stopped" ? "running" : paperSessionStatus
        paperSessionTicks += 1
        var validation = validatePaperOrderPreview()
        var timestamp = previewTime(paperSessionTicks)
        var event = ({ timestamp: timestamp, time: timestamp, pair: selectedTerminalPair, action: validation.action, side: terminalSide, type: terminalOrderType, price: terminalOrderType === "MARKET" ? "paper market preview" : terminalPrice, amount: terminalAmount, total: terminalTotal, status: validation.status, confidence: validation.ok ? "0.86" : "0.00", reason: validation.reason })
        var orderRows = paperOrderRows.slice()
        orderRows.unshift(event)
        paperOrderRows = orderRows.slice(0, previewPaperOrderFeedLimit)
        var terminalRows = mockTerminalOrders.slice()
        terminalRows.unshift(event)
        mockTerminalOrders = terminalRows.slice(0, previewPaperOrderFeedLimit)
        if (event.status === "blocked")
            paperBlockedCount += 1
        else if (event.status === "no order")
            paperNoOrderCount += 1
        else
            paperSimulatedCount += 1
        updatePaperEquityPreview(event.status, Number(event.total))
        updatePaperPositionPreview(event)
        appendTerminalLog(event.action + " " + event.pair + " • " + event.status + " • " + event.reason)
        appendPaperTelemetry("paper order event " + event.action + " " + event.pair + " • " + event.status + " • " + event.reason)
        appendPaperDecision(event.action, event.reason, event.pair, event.status)
        appendPreviewAlert(event.status === "blocked" ? "Warning" : "Info", "Paper", event.status === "blocked" ? "Paper order blocked" : "Paper order simulated", event.action + " " + event.pair + " • " + event.reason, "Paper Terminal", event.pair, "Explain event")
        terminalSelectedBottomTab = "Orders"
        syncPaperBridgeState()
        return event
    }
    function simulateTerminalOrder() { submitLocalPaperOrder() }
    function selectTerminalBottomTab(tab) { terminalSelectedBottomTab = tab }
    function useOrderBookPrice(price) {
        terminalPrice = String(price)
        recalcTerminalTotal()
        var logCopy = terminalLogRows.slice()
        logCopy.unshift(({ time: previewTime(logCopy.length + 1), message: "Copied mock order book price " + price + " into local Order Form • exchange I/O disabled" }))
        terminalLogRows = logCopy.slice(0, previewTerminalLogLimit)
    }


    function visiblePairsCount() { return filteredMarketPairs().length }
    function selectedPairsCount() { return selectedPairs.length }
    function whitelistedPairsCount() { return whitelistPairs.length }
    function blacklistedPairsCount() { return blacklistPairs.length }
    function aiCandidatesCount() {
        var count = 0
        for (var i = 0; i < previewMarketPairs.length; ++i)
            if (isAiCandidate(previewMarketPairs[i]))
                count += 1
        return count
    }
    function saveStrategyPreview(name) { lastStrategySaveStatus = "Zapisano lokalny preview strategii: " + name + " • runtime config write disabled • live execution disabled" }
    function ensureSelectedTerminalPair() {
        var preferredPair = preferredTerminalPair()
        if (preferredPair && preferredPair.length > 0 && selectedTerminalPair !== preferredPair) {
            setTerminalPairFromSource(preferredPair, "ensureSelectedTerminalPair")
        }
    }
    function syncUniverseSelectionState(message) {
        ensureSelectedTerminalPair()
        if (decisionPairFilter !== "All pairs" && !hasValue(selectedPairs, decisionPairFilter))
            decisionPairFilter = selectedTerminalPair
        lastGovernorDecision = selectedTerminalPair + " UNIVERSE UPDATE • " + selectedPairs.length + " selected preview pairs • " + message
        appendPaperTelemetry(message + " • selected pairs update Dashboard, Decisions and Paper Terminal")
    }
    function toggleExchange(exchange) {
        selectedExchanges = toggledList(selectedExchanges, exchange)
        marketsImported = true
        syncUniverseSelectionState("Exchange toggle preview: " + exchange + " • sandbox/testnet/API planned/disabled")
    }
    function togglePair(pair) {
        selectedPairs = toggledList(selectedPairs, pair)
        whitelistPairs = selectedPairs.slice()
        if (hasValue(blacklistPairs, pair) && hasValue(selectedPairs, pair))
            blacklistPairs = blacklistPairs.filter(function(item) { return item !== pair })
        syncUniverseSelectionState("Pair toggle preview: " + pair)
    }
    function toggleBlacklist(pair) {
        blacklistPairs = toggledList(blacklistPairs, pair)
        if (hasValue(blacklistPairs, pair))
            selectedPairs = selectedPairs.filter(function(item) { return item !== pair })
        whitelistPairs = selectedPairs.slice()
        syncUniverseSelectionState("Blacklist toggle preview: " + pair)
    }
    function importMarketsPreview() {
        marketsImported = true
        marketAiCandidatesOnly = true
        syncUniverseSelectionState("Import markets preview complete; AI scans eligible local pairs only")
    }
    function selectAllVisiblePairs() {
        marketsImported = true
        selectedPairs = filteredMarketPairs().filter(function(pair) { return blacklistPairs.indexOf(pair) < 0 })
        whitelistPairs = selectedPairs.slice()
        syncUniverseSelectionState("Select all visible preview pairs")
    }
    function selectAllPairs() { selectAllVisiblePairs() }
    function clearSelectedPairs() {
        selectedPairs = []
        whitelistPairs = []
        decisionPairFilter = "All pairs"
        syncUniverseSelectionState("Clear selected preview pairs")
    }
    function selectTop20Pairs() {
        marketsImported = true
        selectedPairs = filteredMarketPairs().filter(function(pair) { return blacklistPairs.indexOf(pair) < 0 }).slice(0, 20)
        whitelistPairs = selectedPairs.slice()
        syncUniverseSelectionState("Select top 20 preview pairs")
    }
    function blacklistSelectedPairs() {
        blacklistPairs = uniqueList(blacklistPairs.concat(selectedPairs))
        selectedPairs = []
        whitelistPairs = []
        decisionPairFilter = "All pairs"
        syncUniverseSelectionState("Blacklist selected preview pairs")
    }
    function whitelistSelectedPairs() {
        whitelistPairs = selectedPairs.slice()
        blacklistPairs = blacklistPairs.filter(function(pair) { return selectedPairs.indexOf(pair) < 0 })
        syncUniverseSelectionState("Whitelist selected preview pairs")
    }
    function setAutonomyMode(mode) { autonomyMode = mode; autonomyLevel = mode === "Advisory" ? 1 : (mode === "Supervised dry-run" ? 2 : (mode === "Autonomous paper" ? 4 : 0)) }
    function setDecisionPolicy(policy) { decisionPolicyPreview = policy }
    function setConfidenceThreshold(value) { confidenceThreshold = value }
    function riskProfileDescription(profile) {
        if (profile === "Conservative") return "Dlaczego takie ustawienia? Conservative ogranicza ekspozycję, liczbę pozycji, poślizg i drawdown dla ostrożnego Paper Preview."
        if (profile === "Aggressive") return "Dlaczego takie ustawienia? Aggressive pokazuje wyższe limity preview dla bardziej aktywnej symulacji, ale live trading, exchange I/O i order submission nadal są wyłączone."
        if (profile === "Custom") return "Dlaczego takie ustawienia? Użytkownik ręcznie ustawia wartości; zmiany są local-only preview i nie zapisują real runtime config."
        if (profile === "AI Recommended") return riskExplanation
        return "Dlaczego takie ustawienia? Balanced utrzymuje średnie limity dla lokalnego Paper Preview: umiarkowana pozycja, standardowy stop loss i aktywny daily loss limit."
    }
    function refreshRiskActiveLimits(source, comments) {
        var localSource = source || riskProfile
        var note = comments || ({})
        riskActiveLimits = [
            ({ parameter: "max position", value: maxPosition, source: localSource, comment: note.maxPosition || "limit pojedynczej pozycji preview" }),
            ({ parameter: "max open positions", value: String(maxOpenPositions), source: localSource, comment: note.maxOpenPositions || "ile pozycji może pokazać lokalna symulacja" }),
            ({ parameter: "stop loss", value: stopLoss, source: localSource, comment: note.stopLoss || "lokalny próg ograniczenia straty" }),
            ({ parameter: "take profit", value: takeProfit, source: localSource, comment: note.takeProfit || "lokalny próg realizacji zysku" }),
            ({ parameter: "max slippage", value: maxSlippage, source: localSource, comment: note.maxSlippage || "preview akceptowanego poślizgu" }),
            ({ parameter: "max drawdown", value: maxDrawdown, source: localSource, comment: note.maxDrawdown || "lokalny limit obsunięcia kapitału" }),
            ({ parameter: "daily loss limit", value: dailyLossLimit, source: localSource, comment: note.dailyLossLimit || "dzienny bezpiecznik Paper Preview" }),
            ({ parameter: "per-symbol exposure", value: perSymbolExposure, source: localSource, comment: note.perSymbolExposure || "ekspozycja na jedną parę" }),
            ({ parameter: "confidence floor", value: confidenceFloor, source: localSource, comment: note.confidenceFloor || "wpływa deterministycznie na PAPER BUY/SELL/HOLD" }),
            ({ parameter: "cooldown", value: cooldown, source: localSource, comment: note.cooldown || "preview przerwy między ryzykowniejszymi akcjami" })
        ]
    }
    function captureCustomRiskState() {
        customRiskState = ({ maxPosition: maxPosition, maxOpenPositions: maxOpenPositions, stopLoss: stopLoss, takeProfit: takeProfit, maxSlippage: maxSlippage, maxDrawdown: maxDrawdown, dailyLossLimit: dailyLossLimit, perSymbolExposure: perSymbolExposure, confidenceFloor: confidenceFloor, cooldown: cooldown, maxAllocation: maxAllocation, allowAiOverride: allowAiOverride })
    }
    function setCustomRiskValue(field, value) {
        riskProfile = "Custom"
        riskLimitSource = "Custom"
        if (field === "maxPosition") maxPosition = String(value)
        else if (field === "maxOpenPositions") maxOpenPositions = Math.max(1, Number(value))
        else if (field === "stopLoss") stopLoss = String(value)
        else if (field === "takeProfit") takeProfit = String(value)
        else if (field === "maxSlippage") maxSlippage = String(value)
        else if (field === "maxDrawdown") maxDrawdown = String(value)
        else if (field === "dailyLossLimit") dailyLossLimit = String(value)
        else if (field === "perSymbolExposure") perSymbolExposure = String(value)
        else if (field === "confidenceFloor") { confidenceFloor = String(value); confidenceThreshold = Math.max(1, Math.min(99, parseInt(value))) }
        else if (field === "cooldown") cooldown = String(value)
        else if (field === "maxAllocation") maxAllocation = String(value)
        else if (field === "allowAiOverride") allowAiOverride = Boolean(value)
        captureCustomRiskState()
        riskState = "Custom preview • user-edited local limits • live blocked"
        riskLocked = false
        riskExplanation = riskProfileDescription("Custom")
        refreshRiskActiveLimits("Custom")
    }
    function setLocalRiskKillSwitch(enabled, reason) {
        riskLocked = enabled === true || enabled === "true" || enabled === "1"
        riskBlockReason = riskLocked ? (reason || "Risk gate blocked: local preview kill-switch enabled for smoke/risk validation.") : "Risk gate clear: local preview kill-switch inactive."
        riskState = riskProfile + " preview • daily loss limit active • paper actions evaluated by local risk gate • live blocked • " + riskBlockReason
    }

    function setLocalPaperPnlForRiskPreview(value) {
        paperPnl = Number(value)
        paperEquity = Number((portfolioStartingEquityUsd + paperPnl).toFixed(2))
        previewPnl = paperPnl
        previewEquity = paperEquity
        syncPortfolioPerformanceState()
    }

    function setRiskProfile(profile) {
        if (profile === "AI Recommended") { applyAiRecommendedRiskProfile(); return }
        riskProfile = profile
        riskLimitSource = profile
        if (profile === "Conservative") { maxPosition = "1,000 USDT"; maxOpenPositions = 2; stopLoss = "1.5%"; takeProfit = "3.0%"; maxSlippage = "0.10%"; maxDrawdown = "3.0%"; dailyLossLimit = "500 USDT"; perSymbolExposure = "10% equity"; confidenceFloor = "82%"; cooldown = "180s"; maxAllocation = "10% equity" }
        else if (profile === "Aggressive") { maxPosition = "5,000 USDT"; maxOpenPositions = 8; stopLoss = "4.0%"; takeProfit = "8.0%"; maxSlippage = "0.35%"; maxDrawdown = "10.0%"; dailyLossLimit = "2,500 USDT"; perSymbolExposure = "28% equity"; confidenceFloor = "68%"; cooldown = "60s"; maxAllocation = "28% equity" }
        else if (profile === "Custom") { maxPosition = customRiskState.maxPosition; maxOpenPositions = customRiskState.maxOpenPositions; stopLoss = customRiskState.stopLoss; takeProfit = customRiskState.takeProfit; maxSlippage = customRiskState.maxSlippage; maxDrawdown = customRiskState.maxDrawdown; dailyLossLimit = customRiskState.dailyLossLimit; perSymbolExposure = customRiskState.perSymbolExposure; confidenceFloor = customRiskState.confidenceFloor; cooldown = customRiskState.cooldown; maxAllocation = customRiskState.maxAllocation; allowAiOverride = customRiskState.allowAiOverride }
        else { maxPosition = "2,500 USDT"; maxOpenPositions = 4; stopLoss = "2.4%"; takeProfit = "4.8%"; maxSlippage = "0.20%"; maxDrawdown = "6.0%"; dailyLossLimit = "1,200 USDT"; perSymbolExposure = "18% equity"; confidenceFloor = "75%"; cooldown = "120s"; maxAllocation = "18% equity" }
        confidenceThreshold = Math.max(1, Math.min(99, parseInt(confidenceFloor)))
        riskLocked = false
        riskBlockReason = "Risk gate clear: " + profile + " local preview limits allow paper actions unless confidence, position, risk-score or daily-loss limits are breached."
        riskState = profile + " preview • daily loss limit active • paper actions evaluated by local risk gate • live blocked • " + riskBlockReason
        riskExplanation = riskProfileDescription(profile)
        refreshRiskActiveLimits(profile)
    }
    function applyAiRecommendedRiskProfile() {
        riskProfile = "AI Recommended"
        riskLimitSource = "AI Recommended"
        simulationMarketMode = scenarioMode(simulationScenario)
        var cautious = simulationMarketMode === "volatility" || simulationMarketMode === "bear" || modelReadiness < 75 || trainingCoverage < 70 || dataCoverage < 75 || portfolioMaxDrawdown.indexOf("-3") === 0 || paperPnl < 0
        if (cautious) { maxPosition = "900 USDT"; maxOpenPositions = 2; stopLoss = "1.4%"; takeProfit = "3.2%"; maxSlippage = "0.08%"; maxDrawdown = "3.0%"; dailyLossLimit = "450 USDT"; perSymbolExposure = "8% equity"; confidenceFloor = "84%"; cooldown = "240s"; maxAllocation = "8% equity" }
        else { maxPosition = "2,000 USDT"; maxOpenPositions = 4; stopLoss = "2.2%"; takeProfit = "4.4%"; maxSlippage = "0.16%"; maxDrawdown = "5.0%"; dailyLossLimit = "900 USDT"; perSymbolExposure = "14% equity"; confidenceFloor = "78%"; cooldown = "150s"; maxAllocation = "14% equity" }
        confidenceThreshold = Math.max(1, Math.min(99, parseInt(confidenceFloor)))
        allowAiOverride = false
        riskLocked = cautious
        var reasons = []
        reasons.push("scenariusz rynku: " + simulationScenario + " / " + simulationMarketMode)
        reasons.push("model readiness " + modelReadiness + "%")
        reasons.push("training coverage " + trainingCoverage + "%")
        reasons.push("data coverage " + dataCoverage + "%")
        reasons.push("confidence threshold " + confidenceThreshold + "%")
        reasons.push("paper PnL " + formatUsd(paperPnl))
        reasons.push("portfolio max drawdown " + portfolioMaxDrawdown)
        aiRecommendedChangedParameters = ["max position", "max open positions", "stop loss", "take profit", "max slippage", "max drawdown", "daily loss limit", "per-symbol exposure", "confidence floor", "cooldown", "max allocation"]
        riskExplanation = "Dlaczego takie ustawienia? AI dobrało " + (cautious ? "profil ostrożny" : "profil zbalansowany") + ", ponieważ " + reasons.join(", ") + ". Zmienione parametry: " + aiRecommendedChangedParameters.join(", ") + ". To lokalny preview bez backend model inference."
        riskBlockReason = "Risk gate clear: AI Recommended local preview uses " + (cautious ? "reduced exposure and stricter confidence limits" : "balanced exposure limits") + " unless scanner risk or daily-loss limits are breached."
        riskState = "AI Recommended preview • " + (cautious ? "exposure reduced" : "balanced exposure") + " • live blocked • " + riskBlockReason
        refreshRiskActiveLimits("AI Recommended", ({ maxPosition: cautious ? "obniżone przez high volatility/bear/readiness guard" : "dobrane lokalnie do stabilniejszego preview", confidenceFloor: "ustawione z uwzględnieniem model readiness i coverage", cooldown: cautious ? "wydłużony przez lokalny risk guard" : "umiarkowany cooldown preview", perSymbolExposure: cautious ? "ekspozycja obniżona przez AI preview" : "ekspozycja zbalansowana" }))
        appendPaperDecision("NO ORDER", riskExplanation, "Risk profile", "AI Recommended local-only")
        appendPreviewAlert("Info", "Risk", "AI Recommended risk applied", riskExplanation, "Risk Guard", "—", "Explain risk")
        appendPaperTelemetry("AI Recommended risk profile applied • local-only • no backend inference • no exchange I/O")
        appendTerminalLog("AI Recommended risk applied • " + maxPosition + " • confidence floor " + confidenceFloor + " • no real orders")
    }
    function setStrategyActive(name, enabled) {
        var active = activeStrategies.slice()
        var idx = active.indexOf(name)
        if (enabled && idx < 0) active.push(name)
        if (!enabled && idx >= 0) active.splice(idx, 1)
        activeStrategies = active
    }
    function previewTime(offset) {
        var total = 12 * 3600 + 5 * 60 + paperTicks * 7 + decisionSequence * 3 + telemetryTick + (offset || 0)
        var h = Math.floor(total / 3600) % 24
        var m = Math.floor((total % 3600) / 60)
        var sec = total % 60
        return (h < 10 ? "0" + h : h) + ":" + (m < 10 ? "0" + m : m) + ":" + (sec < 10 ? "0" + sec : sec) + "Z"
    }
    function actionStatus(action) {
        if (action === "BLOCKED" || action === "BLOCKED LIVE" || action === "BLOCKED PAPER PREVIEW") return "blocked"
        if (action === "NO ORDER" || action === "HOLD" || action === "WAIT") return "no order"
        return "simulated"
    }
    function riskDailyLossLimitValue() {
        return parsePreviewNumber(dailyLossLimit, 0)
    }

    function riskPositionLimitValue() {
        return parsePreviewNumber(maxPosition, paperEquity)
    }

    function evaluateLocalRiskGate(action, scannerRow, confidenceValue, orderTotal) {
        var normalizedAction = normalizePreviewAction(action)
        var confidenceNumber = Number(confidenceValue)
        if (confidenceNumber > 1)
            confidenceNumber = confidenceNumber / 100.0
        var floorNumber = Math.max(1, Math.min(99, parseInt(confidenceFloor || confidenceThreshold))) / 100.0
        var riskScore = scannerRow && scannerRow.riskScore !== undefined ? Number(scannerRow.riskScore) : 0
        var total = Number(orderTotal || 0)
        var dailyLimit = riskDailyLossLimitValue()
        var positionLimit = riskPositionLimitValue()
        if (riskLocked || riskState.indexOf("kill-switch") >= 0) {
            riskBlockReason = "Risk gate blocked: local kill-switch/risk lock is active after preview risk evaluation; only BLOCKED/no-order preview events are allowed."
            return ({ blocked: true, reason: riskBlockReason })
        }
        if ((normalizedAction === "PAPER BUY" || normalizedAction === "PAPER SELL") && confidenceNumber > 0 && confidenceNumber < floorNumber) {
            riskBlockReason = "Risk gate blocked: confidence " + confidenceNumber.toFixed(2) + " is below floor " + confidenceFloor + " for " + riskProfile + "."
            return ({ blocked: true, reason: riskBlockReason })
        }
        if (riskScore > 0 && riskScore > scannerMaxRiskScore) {
            riskBlockReason = "Risk gate blocked: scanner risk score " + riskScore + " exceeds local limit " + scannerMaxRiskScore + " for " + riskProfile + "."
            return ({ blocked: true, reason: riskBlockReason })
        }
        if (total > 0 && total > positionLimit) {
            riskBlockReason = "Risk gate blocked: local order total " + formatUsd(total) + " exceeds max position " + maxPosition + " for " + riskProfile + "."
            return ({ blocked: true, reason: riskBlockReason })
        }
        if (dailyLimit > 0 && paperPnl <= -dailyLimit) {
            riskBlockReason = "Risk gate blocked: paper session PnL " + formatUsd(paperPnl) + " breached daily loss limit " + dailyLossLimit + "."
            return ({ blocked: true, reason: riskBlockReason })
        }
        if (riskProfile === "Custom" && !allowAiOverride && normalizedAction === "PAPER BUY" && riskScore > Math.max(1, scannerMaxRiskScore - 8)) {
            riskBlockReason = "Risk gate blocked: Custom profile has AI override off and scanner risk score " + riskScore + " is too close to limit " + scannerMaxRiskScore + "."
            return ({ blocked: true, reason: riskBlockReason })
        }
        riskBlockReason = "Risk gate clear: " + riskProfile + " local preview limits passed for " + normalizedAction + "."
        return ({ blocked: false, reason: riskBlockReason })
    }

    function previewRiskGateScenarioInput(path) {
        setRiskProfile("Custom")
        setLocalRiskKillSwitch(false, "Risk gate clear: preparing local risk path " + path + ".")
        scannerMaxRiskScore = 55
        setCustomRiskValue("confidenceFloor", "70%")
        setCustomRiskValue("maxPosition", "2,500 USDT")
        setCustomRiskValue("dailyLossLimit", "1,200 USDT")
        setLocalPaperPnlForRiskPreview(0)
        var row = ({ pair: selectedTerminalPair || currentTerminalPair(), recommendation: "TRADE", aiScore: 92, riskScore: 20, reason: "local risk gate scenario " + path })
        var action = "PAPER BUY"
        var confidence = 0.92
        var orderTotal = 250
        if (path === "confidence_floor") {
            setCustomRiskValue("confidenceFloor", "99%")
            confidence = 0.50
        } else if (path === "scanner_score") {
            row.riskScore = 95
        } else if (path === "daily_loss") {
            setCustomRiskValue("dailyLossLimit", "100 USDT")
            setLocalPaperPnlForRiskPreview(-150)
        } else if (path === "max_position") {
            setCustomRiskValue("maxPosition", "100 USDT")
            orderTotal = 500
        } else if (path === "kill_switch") {
            setLocalRiskKillSwitch(true, "Risk gate blocked: local preview kill-switch enabled for " + path + ".")
        }
        return ({ path: path, action: action, scannerRow: row, confidence: confidence, orderTotal: orderTotal })
    }

    function exerciseRiskGatePreviewPath(path) {
        var input = previewRiskGateScenarioInput(path)
        var gate = evaluateLocalRiskGate(input.action, input.scannerRow, input.confidence, input.orderTotal)
        if (!gate.blocked)
            return ({ path: path, blocked: false, reason: gate.reason })
        var pair = input.scannerRow.pair
        var timestamp = previewTime(paperSessionTicks + telemetryTick + decisionSequence + 1)
        var event = ({ timestamp: timestamp, time: timestamp, pair: pair, action: "BLOCKED", side: terminalSide, type: "PAPER", price: terminalPrice, amount: terminalAmount, total: String(input.orderTotal), status: "blocked", confidence: "0.00", reason: gate.reason })
        var orderRows = paperOrderRows.slice()
        orderRows.unshift(event)
        paperOrderRows = orderRows.slice(0, previewPaperOrderFeedLimit)
        var terminalRows = mockTerminalOrders.slice()
        terminalRows.unshift(event)
        mockTerminalOrders = terminalRows.slice(0, previewPaperOrderFeedLimit)
        appendPaperDecision("BLOCKED", gate.reason, pair, "blocked")
        appendPaperTelemetry("risk gate " + path + " BLOCKED " + pair + " • " + gate.reason)
        appendPreviewAlert("Warning", "Risk", "Risk gate blocked " + path, gate.reason, "Risk Governor", pair, "Review risk controls")
        appendTerminalLog("risk gate " + path + " BLOCKED " + pair + " • " + gate.reason)
        recountOrderCounters()
        return ({ path: path, blocked: true, action: "BLOCKED", reason: gate.reason, pair: pair })
    }

    function recountOrderCounters() {
        var blocked = 0, none = 0, simulated = 0
        for (var i = 0; i < paperOrderRows.length; ++i) {
            var status = paperOrderRows[i].status
            if (status === "blocked") blocked += 1
            else if (status === "no order") none += 1
            else if (status.indexOf("paper simulated") >= 0 || status === "simulated") simulated += 1
        }
        paperOrdersCount = paperOrderRows.length
        paperBlockedCount = blocked
        paperNoOrderCount = none
        paperSimulatedCount = simulated
        syncPaperBridgeState()
    }
    function addDecision(action, reason) {
        decisionSequence += 1
        var pair = selectedPairs.length > 0 ? selectedPairs[decisionSequence % selectedPairs.length] : "BTC/USDT"
        var strategy = activeStrategies.length > 0 ? activeStrategies[decisionSequence % activeStrategies.length] : "Strategy governor"
        var confidence = (0.58 + ((decisionSequence % 11) * 0.033)).toFixed(2)
        var row = ({ timestamp: previewTime(decisionSequence), symbol: pair, action: action, confidence: confidence, reason: reason, riskReason: riskState, strategy: strategy, safety: "Live trading disabled • Exchange route disabled • Order submission disabled", paperState: paperSessionStatus, orderEvent: action.indexOf("PAPER") >= 0 ? "Paper Terminal order event preview" : "no order / blocked live" })
        var rows = decisionPreviewRows.slice()
        rows.unshift(row)
        decisionPreviewRows = rows.slice(0, previewDecisionFeedLimit)
        lastGovernorDecision = pair + " " + action + " • confidence " + confidence + " • " + reason
        appendPaperTelemetry("paper decision row " + action + " " + pair)
        return row
    }
    function buildGovernorPreviewDecisionReason(scannerRow, action) {
        var pair = scannerRow ? scannerRow.pair : currentTerminalPair()
        var scannerPart = scannerRow ? ("scanner " + scannerRow.recommendation + " • AI " + scannerRow.aiScore + " • risk " + scannerRow.riskScore) : "scanner state unavailable"
        var riskPart = "risk profile " + riskProfile + " • confidence floor " + confidenceFloor + " • " + riskState
        if (action === "BLOCKED")
            return "Risk gate blocked " + pair + " because " + riskPart + " • " + scannerPart + " • no real order emitted."
        if (action === "NO ORDER" || action === "HOLD" || action === "WAIT")
            return "Governor chose " + action + " for " + pair + " from local " + scannerPart + " and " + riskPart + " • no order emitted."
        return "Governor selected " + action + " for " + pair + " from local " + scannerPart + " and " + riskPart + " • paper recommendation only."
    }
    function chooseGovernorPreviewAction(scannerRow) {
        if (!scannerRow)
            return "WAIT"
        var floorValue = Math.max(1, Math.min(99, parseInt(confidenceFloor || confidenceThreshold)))
        var candidateAction = "HOLD"
        if (scannerRow.recommendation === "TRADE" && scannerRow.aiScore >= floorValue)
            candidateAction = decisionSequence % 2 === 0 ? "PAPER BUY" : "PAPER SELL"
        else if (scannerRow.recommendation === "WATCH" || scannerRow.aiScore >= floorValue - 8)
            candidateAction = "WAIT"
        else
            candidateAction = "NO ORDER"
        var gate = evaluateLocalRiskGate(candidateAction, scannerRow, scannerRow.aiScore / 100.0, 0)
        riskState = riskProfile + " preview • daily loss limit active • paper actions evaluated by local risk gate • live blocked • " + riskBlockReason
        if (scannerRow.recommendation === "BLOCKED" || gate.blocked)
            return "BLOCKED"
        if (scannerRow.aiScore < floorValue)
            return candidateAction
        return candidateAction
    }
    function generateGovernorRecommendation() {
        if (scannerRows.length === 0)
            rebuildMarketScannerRows()
        var scannerRow = scannerRowByPair(scannerSelectedPair) || (scannerAiCandidateRows.length > 0 ? scannerAiCandidateRows[0] : (scannerRows.length > 0 ? scannerRows[0] : null))
        var action = chooseGovernorPreviewAction(scannerRow)
        var pair = scannerRow ? scannerRow.pair : currentTerminalPair()
        var row = appendPaperDecision(action, buildGovernorPreviewDecisionReason(scannerRow, action), pair, actionStatus(action))
        appendPaperTelemetry("governor decision " + action + " " + pair + " • scanner candidates " + scannerCandidateCount + " • risk profile " + riskProfile)
        appendPreviewAlert(action === "BLOCKED" ? "Warning" : "Info", action === "BLOCKED" ? "Risk" : "AI", action === "BLOCKED" ? "Governor decision blocked" : "AI decision generated", row.symbol + " " + action + " • " + row.reason, "AI Governor", row.symbol, "Explain event")
    }
    function generateNextDecision() {
        generateGovernorRecommendation()
    }
    function scenarioMode(scenario) {
        if (scenario === "Bull trend") return "bull"
        if (scenario === "Bear trend") return "bear"
        if (scenario === "High volatility") return "volatility"
        if (scenario === "Sideways/range") return "range"
        return "balanced"
    }
    function scenarioDrift(mode, tick) {
        if (mode === "bull") return 0.004 + (tick % 3) * 0.001
        if (mode === "bear") return -0.004 - (tick % 3) * 0.001
        if (mode === "volatility") return (tick % 2 === 0 ? 0.011 : -0.009) + (tick % 5) * 0.001
        if (mode === "range") return tick % 2 === 0 ? 0.0018 : -0.0016
        return (tick % 4 - 1.5) * 0.0015
    }
    function setSimulationScenario(scenario) {
        simulationScenario = scenario
        simulationMarketMode = scenarioMode(scenario)
        simulationStatusLabel = (simulationRunning ? "running" : (simulationPaused ? "paused" : "stopped")) + " • " + simulationScenario + " • local paper loop only"
        appendSimulationEvent("market scenario changed to " + simulationScenario + " • no exchange I/O")
        appendPreviewAlert("Info", "Trading", "Market scenario changed", "Market scenario changed to " + simulationScenario + " in local preview.", "Simulation", "—", "Review scenario")
    }
    function setSimulationSpeed(speed) {
        var numeric = Number(speed)
        if (isNaN(numeric) || numeric < 1)
            numeric = 1
        simulationSpeed = Math.min(5, Math.round(numeric))
        simulationTickIntervalMs = Math.max(250, Math.round(1200 / simulationSpeed))
        simulationStatusLabel = (simulationRunning ? "running" : (simulationPaused ? "paused" : "stopped")) + " • speed x" + simulationSpeed + " • local paper loop only"
    }
    function appendSimulationEvent(message) {
        var rows = simulationEvents.slice()
        rows.unshift(({ timestamp: previewTime(simulationTickCount + 1), message: message + " • local paper simulation • no real orders" }))
        simulationEvents = rows.slice(0, previewDecisionFeedLimit)
    }
    function updateSimulationOrderBook(lastPrice) {
        var asks = []
        var bids = []
        for (var i = 0; i < 10; ++i) {
            var ask = Number((lastPrice + 4.5 + i * 3.7).toFixed(2))
            var bid = Number((lastPrice - 4.5 - i * 3.5).toFixed(2))
            var amount = Number((0.12 + ((simulationTickCount + i) % 8) * 0.047).toFixed(3))
            asks.push(({ price: ask.toFixed(2), amount: amount.toFixed(3), total: (ask * amount).toFixed(2), action: "Use ask" }))
            bids.push(({ price: bid.toFixed(2), amount: amount.toFixed(3), total: (bid * amount).toFixed(2), action: "Use bid" }))
        }
        mockOrderBookAsks = asks
        mockOrderBookBids = bids
    }
    function updateSimulationCandles(lastPrice) {
        var candles = mockTerminalCandles.slice(1)
        var open = 95 + (simulationTickCount % 8) * 12
        var close = open + (scenarioDrift(simulationMarketMode, simulationTickCount) >= 0 ? -38 : 34)
        candles.push(({ x: 498, open: open, close: close, high: Math.max(48, Math.min(open, close) - 22), low: Math.min(198, Math.max(open, close) + 34), vol: 36 + (simulationTickCount % 7) * 9 }))
        for (var i = 0; i < candles.length; ++i)
            candles[i].x = 36 + i * 42
        mockTerminalCandles = candles.slice(0, 12)
        terminalPrice = Number(lastPrice).toFixed(2)
        recalcTerminalTotal()
    }
    function paperOrderEventFromSimulation(action, status, pair, price, confidence, reason) {
        var side = action === "PAPER SELL" ? "SELL" : (action === "PAPER BUY" ? "BUY" : "NONE")
        var amount = status === "simulated" ? (pair.indexOf("BTC") >= 0 ? "0.010" : (pair.indexOf("ETH") >= 0 ? "0.120" : "1.500")) : "0.000"
        var total = (Number(price) * Number(amount)).toFixed(2)
        var eventStatus = status === "simulated" ? "paper simulated / no real order" : status
        return ({ timestamp: simulationLastTickAt, time: simulationLastTickAt, pair: pair, action: action, side: side, type: "PAPER", price: Number(price).toFixed(2), amount: amount, total: total, status: eventStatus, confidence: confidence, reason: reason })
    }
    function runSimulationTick() {
        simulationTickCount += 1
        paperSessionTicks += 1
        paperSessionStatus = "running"
        paperSessionState = "running"
        simulationRunning = true
        simulationPaused = false
        simulationMarketMode = scenarioMode(simulationScenario)
        var pair = selectedPairs.length > 0 ? selectedPairs[(simulationTickCount - 1) % selectedPairs.length] : "BTC/USDT"
        setTerminalPairFromSource(pair, "runSimulationTick")
        var prices = simulationPrices
        var currentPrice = Number(prices[pair] || (pair.indexOf("BTC") >= 0 ? 68240.50 : (pair.indexOf("ETH") >= 0 ? 3560.10 : 154.20)))
        var nextPrice = Number((currentPrice * (1 + scenarioDrift(simulationMarketMode, simulationTickCount))).toFixed(2))
        prices[pair] = nextPrice
        simulationPrices = prices
        simulationLastTickAt = previewTime(simulationTickCount)
        simulationLastPair = pair
        var actions = ["PAPER BUY", "PAPER SELL", "HOLD", "WAIT", "NO ORDER", "BLOCKED"]
        var action = actions[(simulationTickCount + (simulationMarketMode === "bull" ? 0 : simulationMarketMode === "bear" ? 1 : 2)) % actions.length]
        var status = actionStatus(action)
        var confidence = (0.62 + ((simulationTickCount % 9) * 0.035)).toFixed(2)
        var floorValue = Math.max(1, Math.min(99, parseInt(confidenceFloor || confidenceThreshold))) / 100.0
        var confidenceNumber = Number(confidence)
        var selectedScannerRow = scannerRowByPair(pair)
        var riskGate = evaluateLocalRiskGate(action, selectedScannerRow, confidenceNumber, action === "PAPER BUY" || action === "PAPER SELL" ? nextPrice * 0.01 : 0)
        riskState = riskProfile + " preview • daily loss limit active • paper actions evaluated by local risk gate • live blocked • " + riskBlockReason
        var riskBlocked = riskGate.blocked
        if (riskBlocked) {
            action = "BLOCKED"
            status = "blocked"
        } else if (confidenceNumber < floorValue && (action === "PAPER BUY" || action === "PAPER SELL")) {
            action = confidenceNumber < floorValue - 0.08 ? "NO ORDER" : "HOLD"
            status = "no order"
        } else if (riskProfile === "AI Recommended" && (simulationMarketMode === "volatility" || simulationMarketMode === "bear") && action === "PAPER BUY") {
            action = "HOLD"
            status = "no order"
        }
        var reason = "simulation tick " + simulationTickCount + " • " + simulationScenario + " • risk profile " + riskProfile + " • confidence " + confidence + " vs floor " + confidenceFloor + " • mock price " + nextPrice.toFixed(2) + " • no exchange I/O"
        if (riskBlocked) reason = riskGate.reason + " • BLOCKED local-only event • no exchange I/O"
        simulationLastAction = action
        updateSimulationOrderBook(nextPrice)
        updateSimulationCandles(nextPrice)
        var delta = status === "simulated" ? (action === "PAPER BUY" ? 12.40 : 8.80) : (status === "blocked" ? 0.0 : -1.35)
        if (status !== "blocked") {
            paperPnl = Number((paperPnl + delta).toFixed(2))
            paperEquity = Number((portfolioStartingEquityUsd + paperPnl).toFixed(2))
            previewPnl = paperPnl
            previewEquity = paperEquity
        }
        var order = paperOrderEventFromSimulation(action, status, pair, nextPrice, confidence, reason)
        var orders = paperOrderRows.slice()
        orders.unshift(order)
        paperOrderRows = orders.slice(0, previewPaperOrderFeedLimit)
        var terminalOrders = mockTerminalOrders.slice()
        terminalOrders.unshift(order)
        mockTerminalOrders = terminalOrders.slice(0, previewPaperOrderFeedLimit)
        updatePaperPositionPreview(order)
        simulationLastOrder = status === "simulated" ? order.action + " " + order.pair + " @ " + order.price : (status === "blocked" ? "blocked local preview route" : "no order emitted")
        appendPaperDecision(action, reason, pair, order ? order.status : status)
        appendPaperTelemetry("heartbeat/tick " + simulationTickCount + " • simulation event " + action + " " + pair + " • no exchange I/O • order submission disabled • local paper loop only")
        appendTerminalLog("paper loop tick " + simulationTickCount + " • " + pair + " " + action + " • price " + nextPrice.toFixed(2) + " • no real orders")
        appendSimulationEvent("tick " + simulationTickCount + " scanned " + pair + " action " + action + " price " + nextPrice.toFixed(2))
        if (riskBlocked) appendPreviewAlert("Critical", "Risk", "Risk blocked action", reason, "Risk Governor", pair, "Review risk controls")
        else if (status === "simulated") appendPreviewAlert("Info", "Paper", "Paper order simulated", action + " " + pair + " at mock price " + nextPrice.toFixed(2) + "; no real order.", "Simulation", pair, "Explain event")
        else if (simulationTickCount % 2 === 0) appendPreviewAlert("Info", "AI", "AI decision generated", action + " " + pair + " • " + reason, "Simulation", pair, "Explain event")
        if (status !== "blocked" && Math.abs(delta) > 0) appendPreviewAlert(Math.abs(paperPnl) > 2500 ? "Warning" : "Info", "Portfolio", Math.abs(paperPnl) > 2500 ? "Drawdown warning" : "PnL changed", "Paper PnL now " + paperPnl.toFixed(2) + " and equity " + paperEquity.toFixed(2) + " in local preview.", "Portfolio Preview", pair, "Open portfolio")
        terminalSelectedBottomTab = order ? "Orders" : "Log"
        recountOrderCounters()
        if (simulationTickCount % 3 === 0)
            runMarketScannerTick()
        simulationStatusLabel = "running • speed x" + simulationSpeed + " • " + simulationScenario + " • local paper loop only"
    }
    function startLiveLikePaperSimulation() {
        simulationRunning = true
        simulationPaused = false
        paperSessionStatus = "running"
        paperSessionState = "running"
        simulationStatusLabel = "running • speed x" + simulationSpeed + " • " + simulationScenario + " • local paper loop only"
        appendPaperTelemetry("paper session started • local preview state running • production runtime loop not started")
        appendPreviewAlert("Info", "Paper", "Paper session started", "Local Paper Preview session started; live trading disabled, exchange I/O disabled, no real orders.", "Paper Preview", selectedTerminalPair || currentTerminalPair(), "Open Dashboard")
        syncPaperBridgeState()
        simulationTimer.restart()
        runSimulationTick()
    }
    function pauseLiveLikePaperSimulation() {
        simulationRunning = false
        simulationPaused = true
        paperSessionStatus = "paused"
        paperSessionState = "paused"
        simulationStatusLabel = "paused • state preserved • local paper loop only"
        simulationTimer.stop()
        appendSimulationEvent("paper loop paused")
        appendPaperTelemetry("paper session paused • local QML timer stopped")
        syncPaperBridgeState()
    }
    function stopLiveLikePaperSimulation() {
        simulationRunning = false
        simulationPaused = false
        paperSessionStatus = "stopped"
        paperSessionState = "stopped"
        simulationStatusLabel = "stopped • local paper loop only"
        simulationTimer.stop()
        appendSimulationEvent("paper loop stopped")
        appendPaperTelemetry("paper session stopped • production runtime loop not started")
        syncPaperBridgeState()
    }
    function resetLiveLikePaperSimulation() {
        simulationTimer.stop()
        simulationRunning = false
        simulationPaused = false
        simulationTickCount = 0
        simulationLastTickAt = "not started"
        simulationLastPair = "—"
        simulationLastAction = "—"
        simulationLastOrder = "—"
        simulationStatusLabel = "stopped • local paper loop only"
        simulationEvents = []
        paperSessionStatus = "stopped"
        paperSessionState = "stopped"
        paperSessionTicks = 0
        paperTicks = 0
        paperEquity = portfolioStartingEquityUsd
        paperPnl = 0.0
        paperOrderRows = []
        paperOpenPositions = []
        paperClosedTrades = []
        mockTerminalOrders = []
        terminalLogRows = []
        decisionPreviewRows = []
        telemetryTick = 0
        telemetryHeartbeat = previewTime(0)
        paperTelemetryRows = [({ timestamp: telemetryHeartbeat, message: "paper preview state reset • ticks 0 • orders 0 • scanner rows reset to local catalog baseline • local-only bounded feed" })]
        telemetryRows = paperTelemetryRows.slice(0, previewTelemetryFeedLimit)
        resetMarketScannerPreview()
        alertRows = []
        recomputeAlertCounts()
        appendPreviewAlert("Info", "Paper", "Paper preview reset", "Local preview state reset: ticks 0, orders 0, scanner rows rebuilt, telemetry kept a single reset event, alerts kept exactly this reset alert.", "Paper Preview", "—", "Open Dashboard")
        recountOrderCounters()
        syncPaperBridgeState()
    }
    function runSimulationBurst(count) {
        var ticks = Math.max(0, Number(count) || 0)
        for (var i = 0; i < ticks; ++i)
            runSimulationTick()
    }
    function generatePaperTick() { runSimulationTick() }
    function runTenMockTicks() { runSimulationBurst(10) }
    function startPaperPreview() { startLiveLikePaperSimulation() }
    function pausePaperPreview() { pauseLiveLikePaperSimulation() }
    function stopPaperPreview() { stopLiveLikePaperSimulation() }
    function resetPaperPreview() { resetLiveLikePaperSimulation() }
    function pingTelemetryFeed() {
        telemetryTick += 1
        telemetryHeartbeat = previewTime(telemetryTick)
        telemetryFreshness = "freshness status: heartbeat #" + telemetryTick + " • updated " + telemetryHeartbeat
        var messages = [
            "heartbeat #" + telemetryTick + " • feed: safe preview • runtime loop: not started",
            "market catalog scan #" + telemetryTick + " • exchange route: disabled",
            "paper bridge check #" + telemetryTick + " • not connected / planned",
            "decision stream pulse #" + telemetryTick + " • order submission disabled"
        ]
        appendPaperTelemetry(messages[telemetryTick % messages.length])
        appendPreviewAlert(telemetryTick % 5 === 0 ? "Warning" : "Info", "Telemetry", telemetryTick % 5 === 0 ? "Telemetry heartbeat stale" : "Telemetry heartbeat fresh", telemetryFreshness + " • no backend calls", "Telemetry Feed", "—", "Ping feed")
    }
    function generateDiagnosticBundle() { diagnosticsBundleStatus = "Last bundle path/status: var/tmp/preview-diagnostic-bundle-ui-only.zip • generated local diagnostic status • included: UI state, telemetry snapshot, governor rows, config preview metadata • excluded: secrets, env files, keychain, real environment values, exchange state"; appendPreviewAlert("Info", "Diagnostics", "Diagnostic bundle generated", diagnosticsBundleStatus + " • no secrets read", "Diagnostics", "—", "Open diagnostics") }

    function showPanel(panelId) {
        if (!panelId)
            return
        if (panelId === "terminalPanel")
            ensureSelectedTerminalPair()
        currentPanelId = panelId
        if (layoutController)
            layoutController.setPanelVisibility(panelId, true)
    }

    function showOperatorDashboard() {
        showPanel(defaultPanelId)
    }

    function selectedPanelComponent() {
        if (panelRegistry && panelRegistry[currentPanelId] && panelRegistry[currentPanelId].component)
            return panelRegistry[currentPanelId].component
        return sidePanelComponent
    }

    property var panelMetadata: [
        ({ panelId: "sidePanel", title: qsTr("Dashboard"), titleKey: "nav.dashboard", icon: "fingerprint", defaultColumn: 0, defaultOrder: 0 }),
        ({ panelId: "aiCenterPanel", title: qsTr("AI Center"), titleKey: "nav.aiCenter", icon: "mode_wizard", defaultColumn: 0, defaultOrder: 1 }),
        ({ panelId: "tradingUniversePanel", title: qsTr("Trading Universe"), titleKey: "nav.universe", icon: "cloud", defaultColumn: 0, defaultOrder: 2 }),
        ({ panelId: "marketScannerPanel", title: qsTr("Okazje / Market Scanner"), titleKey: "nav.marketScanner", icon: "cloud", defaultColumn: 0, defaultOrder: 3 }),
        ({ panelId: "portfolioPerformancePanel", title: qsTr("Portfel / Wyniki"), titleKey: "nav.portfolio", icon: "package", defaultColumn: 0, defaultOrder: 4 }),
        ({ panelId: "terminalPanel", title: qsTr("Paper Terminal"), titleKey: "nav.terminal", icon: "package", defaultColumn: 0, defaultOrder: 5 }),
        ({ panelId: "strategiesPanel", title: qsTr("Strategie"), titleKey: "nav.strategies", icon: "strategy_manager", defaultColumn: 0, defaultOrder: 6 }),
        ({ panelId: "riskControlsPanel", title: qsTr("Ryzyko"), titleKey: "nav.risk", icon: "shield", defaultColumn: 0, defaultOrder: 7 }),
        ({ panelId: "aiDecisionsPanel", title: qsTr("Decyzje"), titleKey: "nav.decisions", icon: "mode_wizard", defaultColumn: 0, defaultOrder: 8 }),
        ({ panelId: "telemetryPanel", title: qsTr("Telemetria"), titleKey: "nav.telemetry", icon: "diagnostics", defaultColumn: 0, defaultOrder: 9 }),
        ({ panelId: "alertsPanel", title: qsTr("Alerty"), titleKey: "nav.alerts", icon: "diagnostics", defaultColumn: 0, defaultOrder: 10 }),
        ({ panelId: "diagnosticsPanel", title: qsTr("Diagnostyka"), titleKey: "nav.diagnostics", icon: "diagnostics", defaultColumn: 0, defaultOrder: 11 }),
        ({ panelId: "settingsPanel", title: qsTr("Ustawienia"), titleKey: "nav.settings", icon: "diagnostics", defaultColumn: 0, defaultOrder: 12 }),
        ({ panelId: "runtimeSessionControlPanel", title: qsTr("Runtime / Sesja"), titleKey: "nav.runtimeControl", icon: "diagnostics", defaultColumn: 0, defaultOrder: 13 }),
        ({ panelId: "helpGlossaryPanel", title: qsTr("Pomoc / Słownik"), titleKey: "nav.help", icon: "diagnostics", defaultColumn: 0, defaultOrder: 14 })
    ]

    property var productTabs: panelMetadata

    property var panelRegistry: ({
        "sidePanel": { title: qsTr("Dashboard"), icon: "fingerprint", component: sidePanelComponent },
        "aiCenterPanel": { title: qsTr("AI Center"), icon: "mode_wizard", component: aiCenterPanelComponent },
        "tradingUniversePanel": { title: qsTr("Trading Universe"), icon: "cloud", component: tradingUniversePanelComponent },
        "marketScannerPanel": { title: qsTr("Okazje / Market Scanner"), icon: "cloud", component: marketScannerPanelComponent },
        "portfolioPerformancePanel": { title: qsTr("Portfel / Wyniki"), icon: "package", component: portfolioPerformancePanelComponent },
        "terminalPanel": { title: qsTr("Paper Terminal"), icon: "package", component: terminalPanelComponent },
        "strategiesPanel": { title: qsTr("Strategie"), icon: "strategy_manager", component: strategiesPanelComponent },
        "riskControlsPanel": { title: qsTr("Ryzyko"), icon: "shield", component: riskControlsPanelComponent },
        "aiDecisionsPanel": { title: qsTr("Decyzje"), icon: "mode_wizard", component: aiDecisionsPanelComponent },
        "telemetryPanel": { title: qsTr("Telemetria"), icon: "diagnostics", component: telemetryPanelComponent },
        "diagnosticsPanel": { title: qsTr("Diagnostyka"), icon: "diagnostics", component: diagnosticsPanelComponent },
        "alertsPanel": { title: qsTr("Alerty"), icon: "diagnostics", component: alertsPanelComponent },
        "settingsPanel": { title: qsTr("Ustawienia"), icon: "diagnostics", component: settingsPanelComponent },
        "runtimeSessionControlPanel": { title: qsTr("Runtime / Sesja"), icon: "diagnostics", component: runtimeSessionControlPanelComponent },
        "helpGlossaryPanel": { title: qsTr("Pomoc / Słownik"), icon: "diagnostics", component: helpGlossaryPanelComponent },
        "chartView": { title: qsTr("Strumień decyzji"), icon: "cloud", component: chartViewComponent },
        "strategyWorkbench": { title: qsTr("Warsztat strategii"), icon: "package", component: strategyWorkbenchComponent },
        "modeWizardPanel": { title: qsTr("Tryby pracy"), icon: "mode_wizard", component: modeWizardPanelComponent },
        "strategyManagerPanel": { title: qsTr("Menedżer strategii"), icon: "strategy_manager", component: strategyManagerPanelComponent }
    })

    StylesModule.DesignSystem {
        id: designSystem
        themeBridge: theme
    }

    Drawer {
        id: decisionExplainDrawer
        objectName: "decisionExplainabilityDrawer"
        width: Math.min(root.width * 0.42, 560)
        height: root.height
        edge: Qt.RightEdge
        modal: false
        interactive: true
        onOpened: root.decisionExplainDrawerOpen = true
        onClosed: root.decisionExplainDrawerOpen = false
        background: Rectangle {
            color: root.safeColor("surface", "#10141f")
            border.color: root.safeColor("border", "#3C3F44")
        }

        Components.StyledScrollView {
            anchors.fill: parent
            anchors.margins: 16
            contentWidth: availableWidth
            ColumnLayout {
                width: parent.availableWidth
                spacing: 12
                RowLayout {
                    Layout.fillWidth: true
                    Label { text: qsTr("Dlaczego bot tak zdecydował?"); color: root.safeColor("textPrimary", "#ffffff"); font.pixelSize: 24; font.bold: true; Layout.fillWidth: true }
                    Components.IconButton { designSystem: root.designSystem; text: qsTr("Close"); subtle: true; onClicked: root.closeDecisionExplainDrawer() }
                }
                Label { text: qsTr("Explanation is local preview only • Wyjaśnienie działa lokalnie w preview"); color: root.safeColor("accent", "#5BC8FF"); font.bold: true; wrapMode: Text.WordWrap; Layout.fillWidth: true }
                Rectangle {
                    Layout.fillWidth: true
                    radius: 14
                    color: root.safeColor("surfaceMuted", "#242936")
                    border.color: root.safeColor("border", "#3C3F44")
                    implicitHeight: summaryColumn.implicitHeight + 24
                    ColumnLayout {
                        id: summaryColumn
                        anchors.fill: parent
                        anchors.margins: 12
                        spacing: 8
                        Label { text: root.selectedDecisionPair + " • " + root.selectedDecisionAction; color: root.safeColor("textPrimary", "#ffffff"); font.bold: true; font.pixelSize: 18; Layout.fillWidth: true; wrapMode: Text.WordWrap }
                        Label { text: qsTr("source: %1 • confidence: %2").arg(root.selectedDecisionSource).arg(root.selectedDecisionConfidence); color: root.safeColor("textSecondary", "#c5cad3"); Layout.fillWidth: true; wrapMode: Text.WordWrap }
                        Label { text: qsTr("AI score: %1 • risk score: %2 • liquidity score: %3").arg(root.selectedDecisionInputSnapshot.length > 1 ? root.selectedDecisionInputSnapshot[1].value : "local").arg(root.selectedDecisionInputSnapshot.length > 2 ? root.selectedDecisionInputSnapshot[2].value : "local").arg(root.selectedDecisionInputSnapshot.length > 3 ? root.selectedDecisionInputSnapshot[3].value : "n/a"); color: root.safeColor("textSecondary", "#c5cad3"); Layout.fillWidth: true; wrapMode: Text.WordWrap }
                        Label { text: qsTr("strategy match: %1").arg(root.selectedDecisionStrategy); color: root.safeColor("textSecondary", "#c5cad3"); Layout.fillWidth: true; wrapMode: Text.WordWrap }
                        Label { text: qsTr("risk profile / kill-switch / risk lock: %1").arg(root.selectedDecisionRiskState); color: root.safeColor("textSecondary", "#c5cad3"); Layout.fillWidth: true; wrapMode: Text.WordWrap }
                        Label { text: qsTr("expected paper action: %1").arg(root.selectedDecisionAction.indexOf("PAPER") >= 0 || root.selectedDecisionAction === "TRADE" ? "local paper preview event only" : "no order / observe / wait"); color: root.safeColor("textPrimary", "#ffffff"); Layout.fillWidth: true; wrapMode: Text.WordWrap }
                        Label { text: qsTr("paper impact: %1").arg(root.selectedDecisionPaperImpact); color: root.safeColor("textPrimary", "#ffffff"); Layout.fillWidth: true; wrapMode: Text.WordWrap }
                    }
                }
                Components.PreviewCard {
                    objectName: "decisionExplainHumanExplanation"
                    designSystem: root.designSystem
                    title: qsTr("Human explanation")
                    description: root.selectedDecisionReason
                }
                Components.PreviewCard {
                    objectName: "decisionExplainSafetyBoundary"
                    designSystem: root.designSystem
                    title: qsTr("Safety / Granice bezpieczeństwa")
                    description: qsTr("live disabled, order submission disabled, no real orders • %1").arg(root.selectedDecisionSafetySummary)
                    Flow { Layout.fillWidth: true; spacing: 8; Repeater { model: ["No backend AI inference", "No exchange/API call", "No order submission", "No real orders", "No secrets read", "Brak backendowej inferencji AI", "Brak połączenia z giełdą/API", "Brak składania zleceń", "Brak prawdziwych zleceń", "Brak odczytu sekretów"]; delegate: Rectangle { required property string modelData; radius: 10; height: 30; width: Math.max(145, safetyChip.implicitWidth + 18); color: Qt.rgba(0.33, 0.78, 1, 0.12); border.color: root.safeColor("border", "#3C3F44"); Label { id: safetyChip; anchors.centerIn: parent; text: modelData; color: root.safeColor("textPrimary", "#ffffff"); font.pixelSize: 11 } } } }
                }
                Components.PreviewCard {
                    objectName: "decisionExplainAuditTrail"
                    designSystem: root.designSystem
                    title: qsTr("Audit trail")
                    description: qsTr("Market data snapshot: local preview • AI score computed: local deterministic preview • Risk profile applied • Risk guard result • Order route check • Live mode check • Paper action decision • Financial impact • Safety boundary")
                    ColumnLayout { Layout.fillWidth: true; spacing: 6; Repeater { model: root.selectedDecisionAuditRows; delegate: Label { required property var modelData; text: "• " + modelData.step + " — " + modelData.result; color: root.safeColor("textSecondary", "#c5cad3"); wrapMode: Text.WordWrap; Layout.fillWidth: true } } }
                }
                Components.PreviewCard {
                    objectName: "decisionExplainRiskChecks"
                    designSystem: root.designSystem
                    title: qsTr("Risk checks")
                    description: qsTr("Risk profile, kill-switch, risk lock, confidence floor and order route checks are local-only.")
                    ColumnLayout { Layout.fillWidth: true; spacing: 6; Repeater { model: root.selectedDecisionRiskChecks; delegate: Label { required property var modelData; text: "• " + modelData.label + ": " + modelData.status + " — " + modelData.detail; color: root.safeColor("textSecondary", "#c5cad3"); wrapMode: Text.WordWrap; Layout.fillWidth: true } } }
                }
                Components.PreviewCard {
                    objectName: "decisionExplainInputSnapshot"
                    designSystem: root.designSystem
                    title: qsTr("Input snapshot")
                    description: qsTr("Local preview inputs used for this explanation; no backend inference and no API calls.")
                    ColumnLayout { Layout.fillWidth: true; spacing: 6; Repeater { model: root.selectedDecisionInputSnapshot; delegate: Label { required property var modelData; text: "• " + modelData.label + ": " + modelData.value; color: root.safeColor("textSecondary", "#c5cad3"); wrapMode: Text.WordWrap; Layout.fillWidth: true } } }
                }
                Components.PreviewCard {
                    objectName: "decisionExplainAlternatives"
                    designSystem: root.designSystem
                    title: qsTr("Alternatywy")
                    description: qsTr("Top 3 alternatywne pary z local scanner/decision preview i dlaczego nie wygrały.")
                    ColumnLayout { Layout.fillWidth: true; spacing: 6; Repeater { model: root.selectedDecisionAlternatives; delegate: Label { required property var modelData; text: "• " + modelData.pair + " — score " + modelData.score + " — " + modelData.reason; color: root.safeColor("textSecondary", "#c5cad3"); wrapMode: Text.WordWrap; Layout.fillWidth: true } } }
                }
                Components.PreviewCard {
                    objectName: "decisionExplainLineageLinks"
                    designSystem: root.designSystem
                    title: qsTr("Lineage links")
                    description: qsTr("Decision source, paper event link and telemetry link remain local-only.")
                    ColumnLayout { Layout.fillWidth: true; spacing: 6; Repeater { model: root.selectedDecisionLineageLinks; delegate: Label { required property var modelData; text: "• " + modelData.label + ": " + modelData.value; color: root.safeColor("textSecondary", "#c5cad3"); wrapMode: Text.WordWrap; Layout.fillWidth: true } } }
                }
            }
        }
    }

    Dialog {
        id: startupDialog
        modal: true
        standardButtons: Dialog.Ok
        anchors.centerIn: parent
        title: qsTr("Stan backendu")
        property string body: ""
        onAccepted: visible = false

        contentItem: ColumnLayout {
            anchors.fill: parent
            anchors.margins: 16
            spacing: 12

            Label {
                id: statusBody
                text: startupDialog.body
                wrapMode: Text.WordWrap
                color: designSystem.color("textPrimary")
                Layout.preferredWidth: 420
            }

            Label {
                text: qsTr("Jeśli problem dotyczy konfiguracji, sprawdź plik runtime.yaml lub flagę cloud.")
                wrapMode: Text.WordWrap
                color: designSystem.color("textSecondary")
                visible: startupDialog.title.indexOf(qsTr("Błąd")) !== -1
            }
        }
    }

    Connections {
        target: runtimeService
        function onErrorMessageChanged() {
            if (!runtimeService)
                return
            if (runtimeService.errorMessage && runtimeService.errorMessage.length > 0) {
                startupDialog.title = qsTr("Błąd uruchomienia runtime")
                startupDialog.body = runtimeService.errorMessage
                startupDialog.open()
            }
        }
        function onCloudRuntimeStatusChanged() {
            if (!runtimeService)
                return
            const status = runtimeService.cloudRuntimeStatus || {}
            if (status.status === "ready") {
                const targetLabel = status.target || (cloudRuntimeEnabled ? qsTr("profil cloud") : qsTr("tryb lokalny"))
                startupDialog.title = qsTr("Runtime gotowy")
                startupDialog.body = qsTr("Połączenie z backendem %1 aktywne.").arg(targetLabel)
                startupDialog.open()
            }
        }
    }

    Rectangle {
        anchors.fill: parent
        gradient: Gradient {
            GradientStop { position: 0; color: designSystem.color("gradientHeroStart") }
            GradientStop { position: 1; color: designSystem.color("gradientHeroEnd") }
        }
        z: -2
    }

    header: ToolBar {
        id: toolbar
        implicitHeight: 64
        background: Item {
            anchors.fill: parent
            Rectangle {
                id: toolbarGradient
                anchors.fill: parent
                gradient: Gradient {
                    GradientStop { position: 0; color: designSystem.color("gradientHeroStart") }
                    GradientStop { position: 1; color: designSystem.color("gradientHeroEnd") }
                }
                opacity: 0.9
            }
            MultiEffect {
                anchors.fill: parent
                source: toolbarGradient
                blurEnabled: true
                blur: 1.0
                blurMax: 24
                saturation: 0.95
                brightness: 0.05
            }
        }
        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 16
            anchors.rightMargin: 16
            spacing: 12

            ColumnLayout {
                Layout.alignment: Qt.AlignVCenter
                spacing: 2
                Label {
                    text: root.trText("app.title")
                    font.bold: true
                    font.pixelSize: 16
                    color: designSystem.color("textPrimary")
                }
                Label {
                    text: root.trText("safety.summary")
                    color: designSystem.color("textSecondary")
                    font.pixelSize: 11
                }
            }

            Rectangle { width: 1; height: parent.height * 0.6; color: designSystem.color("border"); opacity: 0.45 }

            Flickable {
                id: tabOverflow
                objectName: "productPreviewTabBar"
                Layout.fillWidth: true
                Layout.preferredHeight: 46
                Layout.alignment: Qt.AlignVCenter
                clip: true
                contentWidth: browserTabBar.implicitWidth
                boundsBehavior: Flickable.StopAtBounds
                flickableDirection: Flickable.HorizontalFlick

                Row {
                    id: browserTabBar
                    height: parent.height
                    spacing: 6
                    Repeater {
                        model: root.productTabs
                        delegate: Rectangle {
                            required property var modelData
                            property bool hovered: false
                            readonly property bool active: root.currentPanelId === modelData.panelId
                            width: Math.max(120, tabLabel.implicitWidth + 34)
                            height: 42
                            radius: 14
                            color: active ? designSystem.color("surface") : (hovered ? Qt.rgba(0.33, 0.78, 1.0, 0.12) : Qt.rgba(0, 0, 0, 0.20))
                            border.color: active ? designSystem.color("accent") : (hovered ? designSystem.color("textSecondary") : designSystem.color("border"))
                            border.width: active ? 2 : 1
                            Rectangle {
                                anchors.left: parent.left
                                anchors.right: parent.right
                                anchors.bottom: parent.bottom
                                height: active ? 3 : 0
                                radius: 2
                                color: designSystem.color("accent")
                            }
                            Label {
                                id: tabLabel
                                anchors.centerIn: parent
                                text: root.trText(modelData.titleKey)
                                font.bold: active
                                color: active ? designSystem.color("textPrimary") : designSystem.color("textSecondary")
                            }
                            MouseArea {
                                anchors.fill: parent
                                hoverEnabled: true
                                cursorShape: Qt.PointingHandCursor
                                onEntered: parent.hovered = true
                                onExited: parent.hovered = false
                                onPressed: parent.opacity = 0.82
                                onReleased: parent.opacity = 1.0
                                onClicked: root.showPanel(modelData.panelId)
                            }
                        }
                    }
                }
            }

            ComboBox {
                id: languageSelector
                objectName: "languageSelector"
                Layout.preferredWidth: 112
                model: root.languageOptions
                textRole: "display"
                currentIndex: root.currentLanguage === "EN" ? 1 : 0
                ToolTip.delay: 800
                ToolTip.visible: hovered || activeFocus
                ToolTip.text: root.trText("language.label") + " — local-only"
                onActivated: function(index) { root.setLanguage(root.languageOptions[index].code) }
            }

            Components.IconButton {
                designSystem: rootDesignSystem
                text: root.trText("refresh.preview")
                helpText: root.tooltipText("Generate Next Tick")
                iconName: "refresh"
                backgroundColor: designSystem.color("accent")
                foregroundColor: designSystem.color("surface")
                onClicked: root.pingTelemetryFeed()
            }
        }
    }

    Rectangle {
        id: appStatusBar
        objectName: "globalAppStatusBar"
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.leftMargin: 16
        anchors.rightMargin: 16
        height: 46
        radius: 18
        color: Qt.rgba(0, 0, 0, 0.18)
        border.color: designSystem.color("border")
        border.width: 1
        Flickable {
            objectName: "globalSafetyBadges"
            anchors.fill: parent
            anchors.margins: 8
            contentWidth: appStatusBadges.implicitWidth
            boundsBehavior: Flickable.StopAtBounds
            flickableDirection: Flickable.HorizontalFlick
            clip: true
            Row {
                id: appStatusBadges
                spacing: 8
                Repeater {
                    model: [
                        root.trText("app.status"),
                        "Mode: " + root.appModePreview,
                        "Live trading: disabled",
                        "Exchange I/O: disabled",
                        "Order submission: disabled",
                        "API keys: not required",
                        "Runtime loop: not started",
                        "Safety: safe preview",
                        "Alerts: " + root.alertUnreadCount,
                        "Lang: " + root.currentLanguage,
                        "Base: " + root.baseCurrency,
                        "Risk: " + root.defaultRiskProfile,
                        "Simulation: " + root.simulationStatusLabel
                    ]
                    delegate: Rectangle {
                        required property string modelData
                        height: 28
                        width: Math.max(98, appStatusBadgeLabel.implicitWidth + 18)
                        radius: 14
                        color: modelData.indexOf("disabled") >= 0 || modelData.indexOf("safe preview") >= 0 ? Qt.rgba(0.35, 0.95, 0.70, 0.13) : Qt.rgba(0.33, 0.78, 1.0, 0.12)
                        border.color: modelData.indexOf("disabled") >= 0 || modelData.indexOf("safe preview") >= 0 ? designSystem.color("accent") : designSystem.color("accent")
                        Label { id: appStatusBadgeLabel; anchors.centerIn: parent; text: modelData; color: designSystem.color("textPrimary"); font.pixelSize: 11; font.bold: true }
                    }
                }
            }
        }
    }

    Rectangle {
        id: centralContentRoot
        objectName: "centralContentRoot"
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.top: appStatusBar.bottom
        anchors.margins: 16
        radius: 28
        color: Qt.rgba(0, 0, 0, 0.08)
        border.color: designSystem.color("border")
        border.width: 1

        Loader {
            id: centralContentLoader
            objectName: "centralContentLoader"
            anchors.fill: parent
            anchors.margins: 0
            property var actionDispatchContextBridge: root.actionDispatchContextBridge
            active: true
            sourceComponent: root.selectedPanelComponent()
        }
    }

    LayoutComponents.DockManager {
        id: dockManager
        anchors.fill: centralContentRoot
        layoutController: layoutController
        panelRegistry: panelRegistry
        designSystem: rootDesignSystem
        visible: false
    }

    Component {
        id: sidePanelComponent
        Views.OperatorDashboard {
            previewState: root
            actionDispatchContextBridge: centralContentLoader.actionDispatchContextBridge
            width: parent ? parent.width : implicitWidth
            height: parent ? parent.height : implicitHeight
            designSystem: rootDesignSystem
        }
    }

    Component {
        id: aiCenterPanelComponent
        Views.AiControlCenter {
            previewState: root
            width: parent ? parent.width : implicitWidth
            height: parent ? parent.height : implicitHeight
            designSystem: rootDesignSystem
        }
    }

    Component {
        id: tradingUniversePanelComponent
        Views.TradingUniverse {
            previewState: root
            width: parent ? parent.width : implicitWidth
            height: parent ? parent.height : implicitHeight
            designSystem: rootDesignSystem
        }
    }

    Component {
        id: marketScannerPanelComponent
        Views.MarketScanner {
            previewState: root
            width: parent ? parent.width : implicitWidth
            height: parent ? parent.height : implicitHeight
            designSystem: rootDesignSystem
        }
    }

    Component {
        id: portfolioPerformancePanelComponent
        Views.PortfolioPerformance {
            previewState: root
            width: parent ? parent.width : implicitWidth
            height: parent ? parent.height : implicitHeight
            designSystem: rootDesignSystem
        }
    }


    Component {
        id: runtimeSessionControlPanelComponent
        Components.StyledScrollView {
            designSystem: rootDesignSystem
            objectName: "runtimeSessionControlPanel"
            contentWidth: availableWidth
            clip: true
            ColumnLayout {
                width: parent.availableWidth
                spacing: 12
                Label { objectName: "runtimeSessionControlTitle"; text: qsTr("Runtime / Session / Control-plane health"); font.bold: true; font.pixelSize: 22; color: designSystem.color("textPrimary") }
                Components.PreviewCard { objectName: "runtimeSessionPanelLiveShapeCard"; designSystem: rootDesignSystem; title: qsTr("RUNTIME SESSION PANEL / PREVIEW LIVE-SHAPE ONLY"); description: qsTr("Runtime session panel visible. Current session state: stopped preview; session controls visible: start, stop, pause, resume; start/stop/pause/resume visible as disabled control shape. Live runtime disabled in preview. NO REAL LOOP START. Runtime preflight gate: closed. Runtime activation blocked reason: preview/local/paper typed bridge without secrets, adapters or exchange I/O."); Layout.fillWidth: true }
                Components.PreviewCard { objectName: "runtimeControlPlaneHealthCard"; designSystem: rootDesignSystem; title: qsTr("CONTROL-PLANE HEALTH / HEARTBEAT / SCHEDULER / WORKER STATUS"); description: qsTr("Control-plane health visible: healthy local preview. Scheduler status: stopped preview; worker status: idle preview. Heartbeat visible: mock heartbeat #0 / local timestamp placeholder. MOCK HEARTBEAT ONLY. NO LIVE SCHEDULER WORKER START. NO LIVE ADAPTER START."); Layout.fillWidth: true }
                Components.PreviewCard { objectName: "runtimeRecoveryFailoverDegradedCard"; designSystem: rootDesignSystem; title: qsTr("RECOVERY / FAILOVER / DEGRADED MODE / EMERGENCY STOP SHAPE"); description: qsTr("Recovery controls visible but disabled. Failover state visible: standby preview only. Degraded mode visible: off/preview-ready. Emergency stop shape visible: armed UI control, no backend side effect. Recovery actions disabled. NO LIVE RECONNECT. No real adapter reconnect or worker restart is available in preview."); Layout.fillWidth: true }
                Components.PreviewCard { objectName: "runtimeAuditLocalOnlyBoundaryCard"; designSystem: rootDesignSystem; title: qsTr("RUNTIME AUDIT LOCAL ONLY / NO CLOUD SINK / NO EXTERNAL EXPORT"); description: qsTr("Runtime audit local-only visible. Typed preview bridge only; no real secrets, no live exchange I/O, no real orders/fills, no real account balance fetch, no runtime loop start, no scheduler/live worker start, NO CLOUD SINK, NO EXTERNAL EXPORT."); Layout.fillWidth: true }
            }
        }
    }

    Component {
        id: settingsPanelComponent
        Components.StyledScrollView {
            designSystem: rootDesignSystem
            objectName: "settingsPreviewPanel"
            contentWidth: availableWidth
            clip: true
            ColumnLayout {
                width: parent.availableWidth
                spacing: 14
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Rectangle { objectName: "settingsTitleAccentBar"; Layout.preferredWidth: 4; Layout.fillHeight: true; radius: 2; color: designSystem.color("accent") }
                    ColumnLayout {
                        Layout.fillWidth: true
                        Label { objectName: "settingsPreviewTitle"; text: root.trText("settings.title") + " / Settings"; font.bold: true; font.pixelSize: 24; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
                        Label { text: root.trText("settings.description") + " " + qsTr("Safety badges below confirm local-only preview boundaries."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
                    }
                }
                Components.PreviewCard {
                    objectName: "settingsSafetyBoundaryCard"
                    designSystem: rootDesignSystem
                    title: qsTr("Safety boundary / Granice bezpieczeństwa")
                    description: root.settingsSafetySummary
                    Flow {
                        objectName: "settingsGlobalSafetyBadges"
                        Layout.fillWidth: true
                        spacing: 8
                        Repeater {
                            model: ["Live trading: disabled", "Exchange I/O: disabled", "Order submission: disabled", "API keys: not required", "Runtime loop: not started", "Safety: safe preview"]
                            delegate: Rectangle { required property string modelData; width: Math.max(150, safetyBadgeText.implicitWidth + 20); height: 30; radius: 15; color: Qt.rgba(0.35, 0.95, 0.70, 0.12); border.color: designSystem.color("accent"); Label { id: safetyBadgeText; anchors.centerIn: parent; text: modelData; color: designSystem.color("textPrimary"); font.bold: true; font.pixelSize: 11 } }
                        }
                    }
                }
                GridLayout {
                    objectName: "settingsStateGrid"
                    Layout.fillWidth: true
                    columns: width > 980 ? 2 : 1
                    rowSpacing: 10
                    columnSpacing: 10
                    Components.PreviewCard {
                        designSystem: rootDesignSystem
                        title: qsTr("Język / Language")
                        description: qsTr("PL / EN • local-only")
                        Flow { Layout.fillWidth: true; spacing: 8; Repeater { model: root.languageOptions; delegate: Components.IconButton { required property var modelData; designSystem: rootDesignSystem; text: modelData.display; subtle: root.currentLanguage !== modelData.code; helpText: root.tooltipText("Settings"); onClicked: root.setLanguage(modelData.code) } } }
                    }
                    Components.PreviewCard {
                        objectName: "baseCurrencySelector"
                        designSystem: rootDesignSystem
                        title: qsTr("Waluta bazowa / Base currency")
                        description: root.baseCurrency + " • Base currency selector • " + root.tooltipText("Base currency selector")
                        Flow { Layout.fillWidth: true; spacing: 8; Repeater { model: root.baseCurrencyOptions; delegate: Components.IconButton { required property string modelData; designSystem: rootDesignSystem; text: modelData; subtle: root.baseCurrency !== modelData; helpText: root.tooltipText("Base currency selector"); onClicked: root.setBaseCurrency(modelData) } } }
                    }
                    Components.PreviewCard {
                        objectName: "themePreviewSelector"
                        designSystem: rootDesignSystem
                        title: qsTr("Motyw / Theme preview")
                        description: root.themeModePreview + " • Dark preview / Light planned"
                        Flow { Layout.fillWidth: true; spacing: 8; Repeater { model: root.themeModePreviewOptions; delegate: Components.IconButton { required property string modelData; designSystem: rootDesignSystem; text: modelData; subtle: root.themeModePreview !== modelData; helpText: root.tooltipText("Theme preview"); onClicked: root.setThemeModePreview(modelData) } } }
                    }
                    Components.PreviewCard {
                        objectName: "uiDensitySelector"
                        designSystem: rootDesignSystem
                        title: qsTr("Rozmiar UI / UI density")
                        description: root.uiDensity + " • Compact / Comfortable / Large • UI density selector"
                        Flow { Layout.fillWidth: true; spacing: 8; Repeater { model: root.uiDensityOptions; delegate: Components.IconButton { required property string modelData; designSystem: rootDesignSystem; text: modelData; subtle: root.uiDensity !== modelData; helpText: root.tooltipText("UI density selector"); onClicked: root.setUiDensity(modelData) } } }
                    }
                    Components.PreviewCard {
                        objectName: "appModePreviewSelector"
                        designSystem: rootDesignSystem
                        title: qsTr("Tryb aplikacji / App mode")
                        description: root.appModePreview + " • App mode selector • Demo Preview / Paper Preview / Sandbox planned / Live disabled"
                        Flow { Layout.fillWidth: true; spacing: 8; Repeater { model: root.appModePreviewOptions; delegate: Components.IconButton { required property string modelData; designSystem: rootDesignSystem; text: modelData; subtle: root.appModePreview !== modelData; helpText: root.tooltipText("App mode selector"); onClicked: root.setAppModePreview(modelData) } } }
                    }

                    Components.PreviewCard {
                        objectName: "settingsModeControlsLiveShapeCard"
                        designSystem: rootDesignSystem
                        title: qsTr("Mode controls live shape")
                        description: qsTr("Current mode: %1 • LOCAL PREVIEW • PAPER ONLY • LIVE DISABLED • preview_mode_boundary • live_mode_locked").arg(root.appModePreview)
                        Flow { Layout.fillWidth: true; spacing: 8; Repeater { model: ["LOCAL PREVIEW", "PAPER ONLY", "Paper mode shape", "Live mode: disabled/locked", "No real live execution"]; delegate: Rectangle { required property string modelData; objectName: modelData.indexOf("Live mode") === 0 ? "settingsLiveModeLockedBadge" : "settingsModeBoundaryBadge"; width: Math.max(150, modeBadgeText.implicitWidth + 20); height: 30; radius: 15; color: Qt.rgba(0.95, 0.65, 0.20, 0.13); border.color: designSystem.color("warning"); Label { id: modeBadgeText; anchors.centerIn: parent; text: modelData; color: designSystem.color("textPrimary"); font.bold: true; font.pixelSize: 11 } } } }
                    }
                    Components.PreviewCard {
                        objectName: "settingsApiKeyCredentialsLiveShapeCard"
                        designSystem: rootDesignSystem
                        title: qsTr("API keys / credentials shape")
                        description: qsTr("Exchange API key status: missing/not configured/read-only preview • key=•••••••• • secret=•••••••• • SECRETS MASKED • NO SECRET MATERIAL • READ-ONLY PREVIEW • no secret read attempts")
                    }
                    Components.PreviewCard {
                        objectName: "settingsExchangeAccountLiveShapeCard"
                        designSystem: rootDesignSystem
                        title: qsTr("Exchange / account configuration shape")
                        description: qsTr("Exchange profile: Paper Preview Catalog • Account profile/status: local placeholder only • Market data source: local preview catalog • Execution venue: paper-only bridge • NO EXCHANGE I/O • NO ACCOUNT BALANCE FETCH • NO LIVE ORDER ROUTE")
                    }
                    Components.PreviewCard {
                        objectName: "settingsConfigValidationLiveShapeCard"
                        designSystem: rootDesignSystem
                        title: qsTr("Safety / config validation shape")
                        description: qsTr("Config validation: preview valid • readiness checklist visible • risk/safety guardrails active • live activation blocked reason: preview build cannot enable live execution • emergency stop / kill-switch config shape armed")
                    }
                    Components.PreviewCard {
                        objectName: "settingsAuditTelemetryBoundaryCard"
                        designSystem: rootDesignSystem
                        title: qsTr("Audit / telemetry boundary")
                        description: qsTr("Config/settings changes are local only or read-only • no secrets in logs • no live adapter side effect • NO CLOUD SINK • NO EXTERNAL EXPORT")
                    }
                    Components.PreviewCard {
                        objectName: "defaultPreviewExchangeSelector"
                        designSystem: rootDesignSystem
                        title: qsTr("Domyślna giełda preview")
                        description: root.defaultPreviewExchange + " • preview catalog only"
                        Flow { Layout.fillWidth: true; spacing: 8; Repeater { model: root.defaultPreviewExchangeOptions; delegate: Components.IconButton { required property string modelData; designSystem: rootDesignSystem; text: modelData; subtle: root.defaultPreviewExchange !== modelData; helpText: root.tooltipText("Local preview state"); onClicked: root.setDefaultPreviewExchange(modelData) } } }
                    }
                    Components.PreviewCard {
                        objectName: "defaultTerminalPairSelector"
                        designSystem: rootDesignSystem
                        title: qsTr("Domyślna para terminala")
                        description: root.defaultTerminalPair
                        Flow { Layout.fillWidth: true; spacing: 8; Repeater { model: root.defaultTerminalPairOptions; delegate: Components.IconButton { required property string modelData; designSystem: rootDesignSystem; text: modelData; subtle: root.defaultTerminalPair !== modelData; helpText: root.tooltipText("Local preview state"); onClicked: root.setDefaultTerminalPair(modelData) } } }
                    }
                    Components.PreviewCard {
                        objectName: "defaultRiskProfileSelector"
                        designSystem: rootDesignSystem
                        title: qsTr("Domyślny profil ryzyka")
                        description: root.defaultRiskProfile
                        Flow { Layout.fillWidth: true; spacing: 8; Repeater { model: root.defaultRiskProfileOptions; delegate: Components.IconButton { required property string modelData; designSystem: rootDesignSystem; text: modelData; subtle: root.defaultRiskProfile !== modelData; helpText: root.tooltipText("AI recommended risk"); onClicked: root.setDefaultRiskProfile(modelData) } } }
                    }
                    Components.PreviewCard {
                        objectName: "simulationSettingsCard"
                        designSystem: rootDesignSystem
                        title: qsTr("Simulation speed / Market scenario")
                        description: qsTr("speed x%1 • scenario %2").arg(root.simulationSpeed).arg(root.simulationScenario)
                        Flow { Layout.fillWidth: true; spacing: 8; Components.IconButton { designSystem: rootDesignSystem; text: qsTr("Simulation speed x1"); helpText: root.tooltipText("Simulation speed"); onClicked: root.setSimulationSpeed(1) } Components.IconButton { designSystem: rootDesignSystem; text: qsTr("Simulation speed x3"); helpText: root.tooltipText("Simulation speed"); onClicked: root.setSimulationSpeed(3) } Repeater { model: root.simulationScenarios; delegate: Components.IconButton { required property string modelData; designSystem: rootDesignSystem; text: modelData; subtle: root.simulationScenario !== modelData; helpText: root.tooltipText("Market scenario"); onClicked: root.setSimulationScenario(modelData) } } }
                    }
                    Components.PreviewCard {
                        objectName: "alertPreviewTogglesSettings"
                        designSystem: rootDesignSystem
                        title: qsTr("Alert toggles preview")
                        description: qsTr("mute/sound/desktop preview toggles are UI-only")
                        Flow { Layout.fillWidth: true; spacing: 8; Components.IconButton { designSystem: rootDesignSystem; text: qsTr("Mute alerts"); helpText: root.tooltipText("Mute alerts"); subtle: !root.alertsMuted; onClicked: root.toggleAlertsMuted() } Components.IconButton { designSystem: rootDesignSystem; text: qsTr("Sound preview"); helpText: root.tooltipText("Sound preview"); subtle: !root.alertSoundPreviewEnabled; onClicked: root.toggleAlertSoundPreview() } Components.IconButton { designSystem: rootDesignSystem; text: qsTr("Desktop notification preview"); helpText: root.tooltipText("Desktop notification preview"); subtle: !root.alertDesktopPreviewEnabled; onClicked: root.toggleAlertDesktopPreview() } }
                    }
                }
                Components.PreviewCard {
                    objectName: "settingsActionsCard"
                    designSystem: rootDesignSystem
                    title: qsTr("Apply / reset local preview settings")
                    description: qsTr("dirty=%1 • last updated: %2 • settings_no_runtime_config_write • settings_no_secret_reads").arg(root.settingsDirty).arg(root.settingsLastUpdatedAt)
                    Flow {
                        Layout.fillWidth: true
                        spacing: 8
                        Components.IconButton { designSystem: rootDesignSystem; text: qsTr("Apply preview settings"); helpText: root.tooltipText("Apply preview settings"); backgroundColor: designSystem.color("accent"); foregroundColor: designSystem.color("surface"); onClicked: root.applyPreviewSettings() }
                        Components.IconButton { designSystem: rootDesignSystem; text: qsTr("Reset preview settings"); helpText: root.tooltipText("Reset local preview state"); subtle: true; onClicked: root.resetPreviewSettings() }
                        Components.IconButton { designSystem: rootDesignSystem; text: qsTr("Reset local preview state"); helpText: root.tooltipText("Reset local preview state"); subtle: true; onClicked: root.resetLocalPreviewState() }
                        Components.IconButton { designSystem: rootDesignSystem; text: qsTr("Start onboarding preview"); helpText: root.tooltipText("Start onboarding"); onClicked: root.startOnboardingPreview() }
                        Components.IconButton { designSystem: rootDesignSystem; text: qsTr("Export preview config planned"); helpText: root.tooltipText("Settings"); enabled: false }
                        Components.IconButton { designSystem: rootDesignSystem; text: qsTr("Import preview config planned"); helpText: root.tooltipText("Settings"); enabled: false }
                    }
                }
                Components.PreviewCard {
                    objectName: "firstRunOnboardingPreviewWizard"
                    visible: root.firstRunWizardVisible
                    designSystem: rootDesignSystem
                    title: root.trText("onboarding.title") + " — step " + root.onboardingStep + "/" + root.onboardingSteps.length
                    description: root.onboardingSteps[root.onboardingStep - 1] + " • local-only • No runtime config is written • No secrets are read • No exchange/API calls • No order submission • Live trading remains disabled"
                    ColumnLayout {
                        Layout.fillWidth: true
                        Label { text: qsTr("Step 1: Wybierz język • Step 2: Wybierz walutę bazową • Step 3: Demo Preview / Paper Preview / Sandbox planned / Live disabled • Step 4: Wybierz giełdę preview • Step 5: Wybierz profil ryzyka • Step 6: Uruchom Paper Preview / przejdź do Dashboard"); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
                        Flow { Layout.fillWidth: true; spacing: 8; Components.IconButton { designSystem: rootDesignSystem; text: qsTr("Previous"); helpText: root.tooltipText("Onboarding"); enabled: root.onboardingStep > 1; onClicked: root.previousOnboardingStep() } Components.IconButton { designSystem: rootDesignSystem; text: qsTr("Next"); helpText: root.tooltipText("Onboarding"); enabled: root.onboardingStep < root.onboardingSteps.length; onClicked: root.nextOnboardingStep() } Components.IconButton { designSystem: rootDesignSystem; text: qsTr("Complete onboarding"); helpText: root.tooltipText("Complete onboarding"); backgroundColor: designSystem.color("accent"); foregroundColor: designSystem.color("surface"); onClicked: root.completeOnboardingPreview() } Components.IconButton { designSystem: rootDesignSystem; text: qsTr("Skip onboarding preview"); helpText: root.tooltipText("Onboarding"); subtle: true; onClicked: root.skipOnboardingPreview() } }
                    }
                }
            }
        }
    }

    Component {
        id: helpGlossaryPanelComponent
        Components.StyledScrollView {
            objectName: "helpGlossaryRoot"
            designSystem: rootDesignSystem
            contentWidth: availableWidth
            clip: true
            ColumnLayout {
                width: parent.availableWidth
                spacing: 14
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Rectangle { objectName: "helpGlossaryTitleAccentBar"; width: 4; Layout.fillHeight: true; radius: 2; color: designSystem.color("accent") }
                    ColumnLayout {
                        Layout.fillWidth: true
                        Label { objectName: "helpGlossaryTitle"; text: root.trText("help.title"); font.bold: true; font.pixelSize: 24; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
                        Label { text: root.trText("help.description") + " • Live trading disabled • Exchange I/O disabled • Order submission disabled • API keys not required • No real orders • Runtime loop not started."; wrapMode: Text.WordWrap; color: designSystem.color("textSecondary"); Layout.fillWidth: true }
                    }
                }
                Repeater {
                    model: root.glossaryCategories
                    delegate: Components.PreviewCard {
                        required property var modelData
                        designSystem: rootDesignSystem
                        title: root.trText(modelData.key)
                        description: qsTr("Kliknij/hover na przyciskach w aplikacji, aby zobaczyć krótkie podpowiedzi.")
                        Flow {
                            Layout.fillWidth: true
                            spacing: 10
                            Repeater {
                                model: modelData.terms
                                delegate: Rectangle {
                                    required property string modelData
                                    width: Math.max(240, glossaryTermColumn.implicitWidth + 24)
                                    height: glossaryTermColumn.implicitHeight + 20
                                    radius: 14
                                    color: designSystem.color("surfaceMuted")
                                    border.color: designSystem.color("border")
                                    ColumnLayout {
                                        id: glossaryTermColumn
                                        anchors.fill: parent
                                        anchors.margins: 10
                                        spacing: 4
                                        Label { text: modelData; color: designSystem.color("textPrimary"); font.bold: true; Layout.fillWidth: true; wrapMode: Text.WordWrap }
                                        Label { text: root.glossaryDescription(modelData); color: designSystem.color("textSecondary"); Layout.fillWidth: true; wrapMode: Text.WordWrap }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Component {
        id: alertsPanelComponent
        Components.StyledScrollView {
            objectName: "alertCenterRoot"
            designSystem: rootDesignSystem
            contentWidth: availableWidth
            clip: true
            ColumnLayout {
                width: parent.availableWidth
                spacing: 14
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Rectangle { width: 4; Layout.fillHeight: true; radius: 2; color: designSystem.color("accent") }
                    ColumnLayout {
                        Layout.fillWidth: true
                        Label { objectName: "alertCenterTitle"; text: root.trText("nav.alerts"); font.bold: true; font.pixelSize: 24; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
                        Label { text: root.alertSafetyBoundaryCopy; color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
                    }
                }
                GridLayout {
                    objectName: "alertCenterSummaryCards"
                    Layout.fillWidth: true
                    columns: width > 980 ? 5 : 2
                    rowSpacing: 10
                    columnSpacing: 10
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Unread alerts"); description: String(root.alertUnreadCount); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Critical count"); description: String(root.alertCriticalCount); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Warning count"); description: String(root.alertWarningCount); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Info count"); description: String(root.alertInfoCount); Layout.fillWidth: true }
                    Components.PreviewCard { descriptionObjectName: "previewAlertsLatestMessageLabel"; designSystem: rootDesignSystem; title: qsTr("Last event"); description: root.alertRows.length > 0 ? root.alertRows[0].message : root.alertLastEventAt; Layout.fillWidth: true }
                }
                Components.PreviewCard {
                    objectName: "alertCenterFilters"
                    designSystem: rootDesignSystem
                    title: qsTr("Severity filter / Category filter")
                    description: qsTr("All • Critical • Warning • Info | Trading • Risk • AI • Scanner • Paper • Portfolio • Telemetry • Diagnostics • Safety")
                    Flow {
                        Layout.fillWidth: true
                        spacing: 8
                        Repeater {
                            model: root.alertSeverityFilters
                            delegate: Components.IconButton { required property string modelData; designSystem: rootDesignSystem; text: modelData; helpText: root.tooltipText("Severity filter"); subtle: root.alertSelectedSeverity !== modelData; onClicked: root.setAlertSeverityFilter(modelData) }
                        }
                    }
                    Flow {
                        Layout.fillWidth: true
                        spacing: 8
                        Repeater {
                            model: root.alertCategoryFilters
                            delegate: Components.IconButton { required property string modelData; designSystem: rootDesignSystem; text: modelData; helpText: root.tooltipText("Category filter"); subtle: root.alertSelectedCategory !== modelData; onClicked: root.setAlertCategoryFilter(modelData) }
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        Components.IconButton { designSystem: rootDesignSystem; text: qsTr("Mark all read"); helpText: root.tooltipText("Mark all read"); onClicked: root.markAllAlertsRead() }
                        Components.IconButton { designSystem: rootDesignSystem; text: qsTr("Clear alerts"); helpText: root.tooltipText("Clear alerts"); onClicked: root.clearPreviewAlerts() }
                        Components.IconButton { designSystem: rootDesignSystem; text: qsTr("Mute alerts"); helpText: root.tooltipText("Mute alerts"); subtle: !root.alertMutedPreview; onClicked: root.toggleAlertMutePreview() }
                        Components.IconButton { designSystem: rootDesignSystem; text: qsTr("Sound preview"); helpText: root.tooltipText("Sound preview"); subtle: !root.alertSoundEnabledPreview; onClicked: root.toggleAlertSoundPreview() }
                        Components.IconButton { designSystem: rootDesignSystem; text: qsTr("Desktop notification preview"); helpText: root.tooltipText("Desktop notification preview"); subtle: !root.alertDesktopNotificationsPreview; onClicked: root.toggleDesktopNotificationsPreview() }
                    }
                }
                Components.PreviewCard {
                    objectName: "alertCenterLiveShapeBoundary"
                    designSystem: rootDesignSystem
                    title: qsTr("LOCAL ALERT FEED ONLY")
                    description: qsTr("LOCAL ALERT FEED ONLY • alert feed/list visible • severity • source • category • message • timestamp/FRESHNESS • acknowledged/unresolved placeholder: unread/unresolved • risk blocked alert • order blocked alert • scanner candidate alert • NO CLOUD SINK • NO EXTERNAL EXPORT • PAPER / MOCK EVENTS ONLY • local-only guarantee")
                    Layout.fillWidth: true
                }
                Components.PreviewCard {
                    objectName: "alertCenterTimeline"
                    designSystem: rootDesignSystem
                    title: qsTr("Event timeline")
                    description: qsTr("time • severity • category • source • pair • title • message • action • status/read • acknowledged/unresolved placeholder")
                    Rectangle {
                        Layout.fillWidth: true
                        implicitHeight: 34
                        radius: 10
                        color: designSystem.color("surfaceMuted")
                        RowLayout { anchors.fill: parent; anchors.margins: 8; Label { text: qsTr("time"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 80 } Label { text: qsTr("severity"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 80 } Label { text: qsTr("category"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 90 } Label { text: qsTr("source"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 120 } Label { text: qsTr("pair"); color: designSystem.color("textSecondary"); Layout.preferredWidth: 90 } Label { text: qsTr("title / message / action / status/read"); color: designSystem.color("textSecondary"); Layout.fillWidth: true } }
                    }
                    ListView {
                        objectName: "alertCenterEventList"
                        Layout.fillWidth: true
                        Layout.preferredHeight: 340
                        clip: true
                        spacing: 8
                        model: root.visibleAlertRows()
                        delegate: Rectangle {
                            required property var modelData
                            required property int index
                            width: ListView.view ? ListView.view.width : 1000
                            height: alertRowLayout.implicitHeight + 18
                            radius: 12
                            color: modelData.read ? designSystem.color("surfaceMuted") : Qt.rgba(0.33, 0.78, 1, 0.10)
                            border.color: modelData.severity === "Critical" ? designSystem.color("critical") : (modelData.severity === "Warning" ? designSystem.color("warning") : designSystem.color("border"))
                            MouseArea { anchors.fill: parent; onClicked: root.selectAlertEvent(index) }
                            RowLayout {
                                id: alertRowLayout
                                anchors.fill: parent
                                anchors.margins: 9
                                Label { text: modelData.time; color: designSystem.color("textPrimary"); Layout.preferredWidth: 80 }
                                Label { text: modelData.severity; color: modelData.severity === "Critical" ? designSystem.color("critical") : (modelData.severity === "Warning" ? designSystem.color("warning") : designSystem.color("accent")); font.bold: true; Layout.preferredWidth: 80 }
                                Label { text: modelData.category; color: designSystem.color("textPrimary"); Layout.preferredWidth: 90 }
                                Label { text: modelData.source; color: designSystem.color("textSecondary"); Layout.preferredWidth: 120 }
                                Label { text: modelData.pair; color: designSystem.color("textPrimary"); Layout.preferredWidth: 90 }
                                Label { text: modelData.title + " — " + modelData.message + " • " + modelData.action + " • " + modelData.status; color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
                                Components.IconButton { designSystem: rootDesignSystem; text: qsTr("read"); helpText: root.tooltipText("Mark all read"); onClicked: root.markAlertRead(index) }
                                Components.IconButton { designSystem: rootDesignSystem; text: qsTr("Explain event"); helpText: root.tooltipText("Explain event"); onClicked: root.explainAlertEvent(index) }
                            }
                        }
                    }
                }
                Components.PreviewCard {
                    objectName: "alertCenterDetailPanel"
                    designSystem: rootDesignSystem
                    title: qsTr("Alert detail")
                    description: qsTr("Wyjaśnij zdarzenie / Explain event")
                    ColumnLayout {
                        Layout.fillWidth: true
                        Label { text: (root.alertSelectedEvent.title || qsTr("No selected event")) + " • " + (root.alertSelectedEvent.severity || "—") + " • " + (root.alertSelectedEvent.category || "—"); color: designSystem.color("textPrimary"); font.bold: true; Layout.fillWidth: true; wrapMode: Text.WordWrap }
                        Label { text: (root.alertSelectedEvent.time || "—") + " • " + (root.alertSelectedEvent.source || "—") + " • " + (root.alertSelectedEvent.pair || "—") + " • " + (root.alertSelectedEvent.status || "—"); color: designSystem.color("textSecondary"); Layout.fillWidth: true; wrapMode: Text.WordWrap }
                        Label { text: root.alertSelectedEvent.message || root.alertSafetyBoundaryCopy; color: designSystem.color("textSecondary"); Layout.fillWidth: true; wrapMode: Text.WordWrap }
                        Components.IconButton { designSystem: rootDesignSystem; text: qsTr("Wyjaśnij zdarzenie"); helpText: root.tooltipText("Explain event"); onClicked: root.explainAlertEvent(0) }
                        Label { text: root.alertEventExplanation; color: designSystem.color("textSecondary"); Layout.fillWidth: true; wrapMode: Text.WordWrap }
                    }
                }
            }
        }
    }


    Component {
        id: terminalPanelComponent
        Views.PaperTerminal {
            previewState: root
            width: parent ? parent.width : implicitWidth
            height: parent ? parent.height : implicitHeight
            designSystem: rootDesignSystem
        }
    }

    Component {
        id: telemetryPanelComponent
        Components.StyledScrollView {
            designSystem: rootDesignSystem
            objectName: "telemetryFeedPreviewPanel"
            contentWidth: availableWidth
            clip: true
            ColumnLayout {
                width: parent.availableWidth
                spacing: 14
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Rectangle { objectName: "telemetryTitleAccentBar"; width: 4; Layout.fillHeight: true; radius: 2; color: designSystem.color("accent") }
                    ColumnLayout {
                        Layout.fillWidth: true
                        Label { objectName: "telemetryFeedPreviewTitle"; text: qsTr("Telemetria"); font.bold: true; font.pixelSize: 22; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
                        Label { text: qsTr("Lokalna telemetria preview: heartbeat/tick source state, freshness status i limitowana lista 8–12 wierszy. Zero real network/API calls."); wrapMode: Text.WordWrap; color: designSystem.color("textSecondary"); Layout.fillWidth: true }
                    }
                }
                GridLayout {
                    Layout.fillWidth: true
                    columns: width > 900 ? 4 : 2
                    rowSpacing: 10
                    columnSpacing: 10
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Feed status"); description: qsTr("feed: safe preview • zero real network/API calls"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("runtime loop"); description: qsTr("not started"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("exchange route"); description: qsTr("disabled"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("paper bridge"); description: qsTr("local-only paper bridge/state • no real orders"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("session ticks"); description: String(root.paperSessionTicks); Layout.fillWidth: true }
                    Components.PreviewCard { descriptionObjectName: "previewTelemetryLatestMessageLabel"; designSystem: rootDesignSystem; title: qsTr("last paper event"); description: root.paperTelemetryRows.length > 0 ? root.paperTelemetryRows[0].message : qsTr("no paper telemetry yet"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Last heartbeat"); description: root.telemetryHeartbeat; Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Data freshness"); description: root.telemetryFreshness; Layout.fillWidth: true }
                }
                Components.PreviewCard {
                    objectName: "telemetryLiveShapeSummary"
                    designSystem: rootDesignSystem
                    title: qsTr("LOCAL PREVIEW TELEMETRY")
                    description: qsTr("LOCAL PREVIEW TELEMETRY • runtime mode: local preview / paper simulation • local preview source marker • event type • component • message • timestamp/FRESHNESS • NO CLOUD SINK • NO EXTERNAL EXPORT • NO LIVE EXCHANGE STREAM • NO REAL ORDER EVENT STREAM • SECRETS NOT LOGGED • PAPER / MOCK EVENTS ONLY")
                    Layout.fillWidth: true
                }
                Components.PreviewCard {
                    objectName: "telemetryAuditLogCard"
                    designSystem: rootDesignSystem
                    title: qsTr("LOCAL AUDIT LOG ONLY")
                    description: qsTr("LOCAL AUDIT LOG ONLY • audit/event log sequence • decision event • order event • risk event • scanner event • TRACE / CORRELATION marker • decision/order/risk/scanner event correlation • local-only guarantee • no external telemetry export")
                    Layout.fillWidth: true
                }
                Components.PreviewCard {
                    designSystem: rootDesignSystem
                    title: qsTr("Telemetry heartbeat feed")
                    description: qsTr("event type heartbeat/paper action • component previewState/paper bridge • message rows append to shared paperTelemetryRows with timestamp/FRESHNESS: last heartbeat, session ticks, last paper event, bounded list 8–12 rows, exchange I/O disabled, runtime loop not started.")
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8
                        Components.IconButton { designSystem: rootDesignSystem; text: qsTr("Ping feed"); iconName: "refresh"; backgroundColor: designSystem.color("accent"); foregroundColor: designSystem.color("surface"); onClicked: root.pingTelemetryFeed() }
                        Label { text: qsTr("heartbeat/tick source state: %1 • session ticks %2 • last paper event: %3").arg(root.telemetryTick).arg(root.paperSessionTicks).arg(root.paperTelemetryRows.length > 0 ? root.paperTelemetryRows[0].message : "none"); color: designSystem.color("textSecondary"); Layout.fillWidth: true; wrapMode: Text.WordWrap }
                    }
                    ListView {
                        objectName: "telemetryFeedList"
                        Layout.fillWidth: true
                        Layout.preferredHeight: 330
                        clip: true
                        spacing: 8
                        model: root.paperTelemetryRows
                        delegate: Rectangle {
                            required property var modelData
                            width: ListView.view ? ListView.view.width : 900
                            height: telemetryRow.implicitHeight + 16
                            radius: 12
                            color: designSystem.color("surfaceMuted")
                            border.color: designSystem.color("border")
                            ColumnLayout {
                                id: telemetryRow
                                anchors.fill: parent
                                anchors.margins: 10
                                Label { text: modelData.timestamp; color: designSystem.color("textPrimary"); font.bold: true }
                                Label { text: modelData.message; color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
                            }
                        }
                    }
                }
            }
        }
    }

    Component {
        id: aiDecisionsPanelComponent
        Views.AiDecisionsView {
            previewState: root
            Layout.fillWidth: true
            Layout.fillHeight: true
            runtimeService: runtimeService
            designSystem: rootDesignSystem
        }
    }

    Component {
        id: chartViewComponent
        Components.StyledScrollView {
            designSystem: rootDesignSystem
            objectName: "decisionStreamPreviewPanel"
            contentWidth: availableWidth
            clip: true
            ColumnLayout {
                width: parent.availableWidth
                spacing: 16
                Label {
                    objectName: "decisionStreamPreviewTitle"
                    text: qsTr("Strumień decyzji i dziennik governora")
                    font.bold: true
                    font.pixelSize: 22
                    color: designSystem.color("textPrimary")
                    Layout.fillWidth: true
                }
                Label {
                    text: qsTr("Wykres confidence oraz dziennik zdarzeń w trybie demo/offline. Order execution disabled.")
                    wrapMode: Text.WordWrap
                    color: designSystem.color("textSecondary")
                    Layout.fillWidth: true
                }
                Components.PreviewCard {
                    designSystem: rootDesignSystem
                    title: qsTr("Confidence preview")
                    description: qsTr("Canvas zachowany, z poprawionym paddingiem i pustym stanem.")
                    Canvas {
                        id: chartCanvas
                        Layout.fillWidth: true
                        Layout.preferredHeight: 180
                        onPaint: {
                            var ctx = getContext("2d")
                            ctx.reset()
                            ctx.fillStyle = designSystem.color("surfaceMuted")
                            ctx.fillRect(0, 0, width, height)
                            var data = runtimeService ? runtimeService.decisions || [] : []
                            if (data.length === 0) {
                                ctx.strokeStyle = designSystem.color("border")
                                ctx.lineWidth = 1
                                for (var g = 1; g < 4; ++g) {
                                    ctx.beginPath(); ctx.moveTo(0, height * g / 4); ctx.lineTo(width, height * g / 4); ctx.stroke()
                                }
                                return
                            }
                            var windowSize = Math.min(40, data.length)
                            var step = width / Math.max(windowSize - 1, 1)
                            ctx.strokeStyle = designSystem.color("accent")
                            ctx.lineWidth = 2
                            ctx.beginPath()
                            for (var i = 0; i < windowSize; ++i) {
                                var entry = data[data.length - windowSize + i]
                                var confidence = entry.decision && entry.decision.confidence !== undefined ? Number(entry.decision.confidence) : 0.35
                                confidence = Math.max(0.05, Math.min(confidence, 1.0))
                                var x = i * step
                                var y = height - (confidence * height)
                                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y)
                            }
                            ctx.stroke()
                        }
                    }
                }
                Connections {
                    target: runtimeService
                    function onDecisionsChanged() { chartCanvas.requestPaint() }
                }
                Components.PreviewCard {
                    designSystem: rootDesignSystem
                    title: qsTr("Zdarzenia governora")
                    description: (!runtimeService || !runtimeService.decisions || runtimeService.decisions.length === 0)
                                 ? qsTr("Brak danych live — pokazuję pusty stan demo/offline.")
                                 : qsTr("Ostatnie decyzje z lokalnego preview bridge.")
                    ListView {
                        id: decisionList
                        Layout.fillWidth: true
                        Layout.preferredHeight: 260
                        model: runtimeService ? runtimeService.decisions : []
                        clip: true
                        spacing: 8
                        ScrollBar.vertical: ScrollBar {
                            policy: ScrollBar.AsNeeded
                            width: 10
                            background: Rectangle { radius: 5; color: Qt.rgba(1, 1, 1, 0.04) }
                            contentItem: Rectangle { radius: 4; color: designSystem.color("surfaceElevated"); border.color: designSystem.color("border"); border.width: 1 }
                        }
                        delegate: Rectangle {
                            width: ListView.view.width
                            color: designSystem.color("surfaceMuted")
                            height: column.implicitHeight + 18
                            radius: 12
                            border.color: designSystem.color("border")
                            border.width: 1
                            Column {
                                id: column
                                anchors.fill: parent
                                anchors.margins: 12
                                spacing: 4
                                Label {
                                    text: qsTr("%1 • %2 • %3").arg(modelData.timestamp || "-").arg(modelData.portfolio || "-").arg(modelData.marketRegime && modelData.marketRegime.label ? modelData.marketRegime.label : "")
                                    font.bold: true
                                    color: designSystem.color("textPrimary")
                                    wrapMode: Text.Wrap
                                }
                                Label {
                                    text: modelData.decision && modelData.decision.shouldTrade ? qsTr("Decyzja: %1 %2 @ %3").arg(modelData.symbol || "-").arg(modelData.side || "").arg(modelData.price || "") : qsTr("Decyzja: brak transakcji")
                                    wrapMode: Text.Wrap
                                    color: designSystem.color("textPrimary")
                                }
                                Label {
                                    text: modelData.ai && modelData.ai.strategy ? qsTr("Governor: %1").arg(modelData.ai.strategy) : ""
                                    color: designSystem.color("textSecondary")
                                    visible: text.length > 0
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Component {
        id: strategyWorkbenchComponent
        Components.StyledScrollView {
            designSystem: rootDesignSystem
            objectName: "strategyWorkbenchPreviewPanel"
            contentWidth: availableWidth
            clip: true
            property var strategies: []
            property var marketplacePresets: strategyManagementController ? strategyManagementController.presets : []
            function rebuild() {
                var data = runtimeService ? runtimeService.decisions || [] : []
                var stats = {}
                for (var i = 0; i < data.length; ++i) {
                    var entry = data[i]
                    var strategy = entry.ai && entry.ai.strategy ? entry.ai.strategy : qsTr("Nieznana strategia")
                    if (!stats[strategy]) stats[strategy] = { count: 0, lastSymbol: entry.symbol || "-" }
                    stats[strategy].count += 1
                    stats[strategy].lastSymbol = entry.symbol || stats[strategy].lastSymbol
                }
                var collection = []
                for (var key in stats) collection.push({ name: key, count: stats[key].count, symbol: stats[key].lastSymbol })
                collection.sort(function(a, b) { return b.count - a.count })
                strategies = collection
            }
            Component.onCompleted: rebuild()
            Connections { target: runtimeService; function onDecisionsChanged() { rebuild() } }
            ColumnLayout {
                width: parent.availableWidth
                spacing: 16
                Label { objectName: "strategyWorkbenchPreviewTitle"; text: qsTr("Warsztat strategii"); font.bold: true; font.pixelSize: 22; color: designSystem.color("textPrimary") }
                Label { text: qsTr("Demo/offline workspace do analizy strategii bez uruchamiania live tradingu ani order execution."); wrapMode: Text.WordWrap; color: designSystem.color("textSecondary"); Layout.fillWidth: true }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Sygnały"); description: strategies.length > 0 ? qsTr("%1 strategii z decyzji preview").arg(strategies.length) : qsTr("Brak live danych — statyczny empty state demo/offline"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Marketplace"); description: marketplacePresets.length > 0 ? qsTr("Presety dostępne lokalnie") : qsTr("Marketplace unavailable w tym preview"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Safety"); description: qsTr("Runtime loop not started, API keys not required"); Layout.fillWidth: true }
                }
                GridLayout {
                    objectName: "strategyModelReplayLiveShapeGrid"
                    Layout.fillWidth: true
                    columns: width > 980 ? 2 : 1
                    rowSpacing: 10
                    columnSpacing: 10
                    Components.PreviewCard { objectName: "strategyRegistryLiveShapeCard"; designSystem: rootDesignSystem; title: qsTr("LOCAL STRATEGY CATALOG / PREVIEW STRATEGY STATE"); description: qsTr("Strategy registry visible: Momentum Guard enabled, Range Guard enabled, Volatility Breakout Preview disabled. Active strategy / selected strategy: Momentum Guard. Strategy health: healthy preview. Strategy risk profile: moderate; capital allocation shape 35% max preview."); Layout.fillWidth: true }
                    Components.PreviewCard { objectName: "modelArtifactLiveShapeCard"; designSystem: rootDesignSystem; title: qsTr("MOCK MODEL ARTIFACT / LOCAL INFERENCE PREVIEW / NO MODEL PROMOTION"); description: qsTr("Model artifact status: loaded mock artifact. Version v-preview-10, hash local-mock-7f3a, lineage scanner->governor->paper. Inference readiness: ready local only. Confidence score 0.81, calibration preview only; NO MODEL PROMOTION."); Layout.fillWidth: true }
                    Components.PreviewCard { objectName: "backtestReplayLiveShapeCard"; designSystem: rootDesignSystem; title: qsTr("Backtest / Replay controls — LOCAL REPLAY ONLY / NO LIVE MARKET DATA FETCH"); description: qsTr("Dataset: bundled preview candles; window 2026-06-01..2026-06-02; timeframe 5m. Start disabled, replay disabled in preview. Result summary: +126.75 USD paper PnL. Metrics: PnL +126.75, win rate 62%, drawdown 3.4%, trades 24. LOCAL REPLAY ONLY. NO LIVE MARKET DATA FETCH."); Layout.fillWidth: true }
                    Components.PreviewCard { objectName: "strategyReadinessDeploymentGateCard"; designSystem: rootDesignSystem; title: qsTr("Readiness checklist / LIVE PROMOTION DISABLED / PAPER ONLY"); description: qsTr("Readiness checklist visible: registry loaded, mock model ready, replay local, audit boundary green. Paper / sandbox / live promotion shape visible but locked in preview. Blocked live promotion reason: preview cannot train, promote, deploy or execute. no live deployment side effect."); Layout.fillWidth: true }
                    Components.PreviewCard { objectName: "strategyAuditTelemetryBoundaryCard"; designSystem: rootDesignSystem; title: qsTr("Strategy audit boundary — local-only mock actions"); description: qsTr("strategy/model/backtest actions are local-only, read-only or mock. No real model training, no model artifact promotion, no live execution, no exchange/account/order side effects, no secrets in logs, NO CLOUD SINK, NO EXTERNAL EXPORT."); Layout.fillWidth: true }
                }
            }
        }
    }

    Component {
        id: strategiesPanelComponent
        Views.Strategies { previewState: root; Layout.fillWidth: true; Layout.fillHeight: true; runtimeService: runtimeService; designSystem: rootDesignSystem }
    }

    Component {
        id: riskControlsPanelComponent
        Views.RiskControls { previewState: root; Layout.fillWidth: true; Layout.fillHeight: true; runtimeService: runtimeService; designSystem: rootDesignSystem }
    }

    Component {
        id: modeWizardPanelComponent
        Views.ModeWizard { width: parent ? parent.width : 900; height: parent ? parent.height : 620; designSystem: rootDesignSystem; modeWizardController: modeWizardController; compact: true; onLaunchWizardRequested: modeWizardDialog.open(); layoutController: layoutController; strategyManagementController: strategyManagementController }
    }

    Component {
        id: strategyManagerPanelComponent
        Views.StrategyManager { width: parent ? parent.width : 900; height: parent ? parent.height : 620; designSystem: rootDesignSystem; strategyManagementController: strategyManagementController; layoutController: layoutController }
    }

    Component {
        id: diagnosticsPanelComponent
        Components.StyledScrollView {
            designSystem: rootDesignSystem
            objectName: "diagnosticsPreviewPanel"
            contentWidth: availableWidth
            clip: true
            ColumnLayout {
                width: parent.availableWidth
                spacing: 14
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Rectangle { objectName: "diagnosticsTitleAccentBar"; Layout.preferredWidth: 4; Layout.fillHeight: true; radius: 2; color: designSystem.color("accent") }
                    ColumnLayout {
                        Layout.fillWidth: true
                        Label { objectName: "diagnosticsPreviewTitle"; text: qsTr("Diagnostyka"); font.bold: true; font.pixelSize: 22; color: designSystem.color("textPrimary"); Layout.fillWidth: true }
                        Label { text: qsTr("Preview diagnostics readiness panel. Generate local diagnostic status updates UI text only and does not read secrets, env files, keychain or real environment values."); color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
                    }
                }
                GridLayout {
                    Layout.fillWidth: true
                    columns: width > 900 ? 3 : 1
                    rowSpacing: 10
                    columnSpacing: 10
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Preview diagnostics readiness"); description: qsTr("ready for safe dry-run audit • UI-only bundle status"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Last bundle path/status"); description: root.diagnosticsBundleStatus; Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Safety boundary"); description: qsTr("Live trading disabled • Exchange route disabled • Order submission disabled • API keys not required"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Included • included"); description: qsTr("UI state • telemetry snapshot • governor rows • config preview metadata"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("Excluded • excluded"); description: qsTr("secrets • env files • keychain • real environment values • exchange state"); Layout.fillWidth: true }
                    Components.PreviewCard { designSystem: rootDesignSystem; title: qsTr("generate local diagnostic status"); description: qsTr("local UI status only; no filesystem secret scan and no exchange connection"); Layout.fillWidth: true }
                }
                Components.PreviewCard {
                    designSystem: rootDesignSystem
                    title: qsTr("Generate diagnostic bundle")
                    description: qsTr("Generate diagnostic bundle records a local status line for preview diagnostics; it never starts live/cloud/runtime services.")
                    RowLayout {
                        Layout.fillWidth: true
                        Components.IconButton { designSystem: rootDesignSystem; iconName: "diagnostics"; text: qsTr("Generate diagnostic bundle"); backgroundColor: designSystem.color("accent"); foregroundColor: designSystem.color("surface"); onClicked: root.generateDiagnosticBundle() }
                        Label { text: root.diagnosticsBundleStatus; color: designSystem.color("textSecondary"); wrapMode: Text.WordWrap; Layout.fillWidth: true }
                    }
                }
            }
        }
    }

    Component.onCompleted: {
        if (runtimeService && runtimeService.loadRecentDecisions)
            runtimeService.loadRecentDecisions(uiConfig ? uiConfig.decision_limit : 25)
        if (licensingController && licensingController.refreshFingerprint)
            licensingController.refreshFingerprint()
        refreshRiskActiveLimits(riskProfile)
        recomputeAlertCounts()
        if (layoutController && layoutController.registerPanels) {
            layoutController.registerPanels(panelMetadata)
            showOperatorDashboard()
        }
    }

    Timer {
        id: simulationTimer
        interval: root.simulationTickIntervalMs
        repeat: true
        running: root.simulationRunning && !root.simulationPaused
        onTriggered: root.runSimulationTick()
    }

    Timer {
        interval: 15000
        repeat: true
        running: true
        onTriggered: runtimeService && runtimeService.loadRecentDecisions(0)
    }

    Dialog {
        id: modeWizardDialog
        modal: true
        anchors.centerIn: parent
        width: Math.min(parent.width * 0.8, 1100)
        height: Math.min(parent.height * 0.85, 780)
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
        Overlay.modal: Rectangle {
            color: Qt.rgba(0, 0, 0, 0.62)
        }
        background: Rectangle {
            anchors.fill: parent
            radius: 24
            color: designSystem.color("surface")
            border.color: designSystem.color("border")
            border.width: 1
        }
        contentItem: Views.ModeWizard {
            anchors.fill: parent
            anchors.margins: 16
            designSystem: rootDesignSystem
            modeWizardController: modeWizardController
            compact: false
            onLaunchWizardRequested: modeWizardDialog.close()
        }
    }
}
