#include <QtTest/QtTest>

#include <QDateTime>
#include <QQmlComponent>
#include <QQmlContext>
#include <QQmlEngine>
#include <QPointer>
#include <QScopedPointer>

#include "models/RiskHistoryModel.hpp"
#include "models/RiskStateModel.hpp"
#include "models/RiskTypes.hpp"

class MockAppController : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString connectionStatus READ connectionStatus NOTIFY connectionStatusChanged)
    Q_PROPERTY(bool reduceMotionActive READ reduceMotionActive NOTIFY reduceMotionActiveChanged)
    Q_PROPERTY(bool offlineMode READ offlineMode NOTIFY offlineModeChanged)
    Q_PROPERTY(QString offlineDaemonStatus READ offlineDaemonStatus NOTIFY offlineDaemonStatusChanged)
    Q_PROPERTY(bool offlineAutomationRunning READ offlineAutomationRunning NOTIFY offlineAutomationRunningChanged)
    Q_PROPERTY(QVariantList activityLog READ activityLog NOTIFY activityLogChanged)

public:
    explicit MockAppController(QObject* parent = nullptr)
        : QObject(parent)
    {
    }

    QString connectionStatus() const { return m_connectionStatus; }
    void setConnectionStatus(const QString& status)
    {
        if (m_connectionStatus == status)
            return;
        m_connectionStatus = status;
        Q_EMIT connectionStatusChanged();
    }

    bool reduceMotionActive() const { return m_reduceMotion; }
    void setReduceMotionActive(bool active)
    {
        if (m_reduceMotion == active)
            return;
        m_reduceMotion = active;
        Q_EMIT reduceMotionActiveChanged();
    }

    bool offlineMode() const { return m_offlineMode; }
    void setOfflineMode(bool offline)
    {
        if (m_offlineMode == offline)
            return;
        m_offlineMode = offline;
        Q_EMIT offlineModeChanged();
    }

    QString offlineDaemonStatus() const { return m_offlineStatus; }
    void setOfflineDaemonStatus(const QString& status)
    {
        if (m_offlineStatus == status)
            return;
        m_offlineStatus = status;
        Q_EMIT offlineDaemonStatusChanged();
    }

    bool offlineAutomationRunning() const { return m_automationRunning; }
    void setOfflineAutomationRunning(bool running)
    {
        if (m_automationRunning == running)
            return;
        m_automationRunning = running;
        Q_EMIT offlineAutomationRunningChanged();
    }

    QVariantList activityLog() const { return m_activityLog; }

    Q_INVOKABLE QVariantMap instrumentConfigSnapshot() const { return m_instrument; }
    void setInstrumentConfig(const QVariantMap& map)
    {
        m_instrument = map;
        Q_EMIT instrumentChanged();
    }

    Q_INVOKABLE QVariantMap performanceGuardSnapshot() const { return m_guard; }
    void setPerformanceGuard(const QVariantMap& guard)
    {
        m_guard = guard;
        Q_EMIT performanceGuardChanged();
    }

    Q_INVOKABLE QVariantMap riskRefreshSnapshot() const { return m_riskRefresh; }
    void setRiskRefresh(const QVariantMap& refresh)
    {
        m_riskRefresh = refresh;
        Q_EMIT riskRefreshScheduleChanged();
    }

    Q_INVOKABLE QVariantList activityLogSnapshot() const { return m_activityLog; }
    void setActivityLog(const QVariantList& log)
    {
        m_activityLog = log;
        Q_EMIT activityLogChanged();
    }

    void appendActivityEvent(const QVariantMap& event)
    {
        QVariantList copy = m_activityLog;
        copy.prepend(event);
        while (copy.size() > 50)
            copy.removeLast();
        m_activityLog = copy;
        Q_EMIT activityLogChanged();
    }

    Q_INVOKABLE bool triggerRiskRefresh()
    {
        ++m_refreshRequests;
        if (m_refreshSuccess) {
            m_riskRefresh.insert(QStringLiteral("lastRequestAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODate));
            Q_EMIT riskRefreshScheduleChanged();
            QVariantMap event;
            event.insert(QStringLiteral("timestamp"), QDateTime::currentDateTimeUtc().toString(Qt::ISODate));
            event.insert(QStringLiteral("type"), QStringLiteral("risk:refresh"));
            event.insert(QStringLiteral("message"), QStringLiteral("Live refresh requested"));
            event.insert(QStringLiteral("success"), true);
            event.insert(QStringLiteral("source"), QStringLiteral("runtime"));
            appendActivityEvent(event);
        }
        return m_refreshSuccess;
    }

    Q_INVOKABLE bool requestRiskRefresh() { return triggerRiskRefresh(); }

    void setRiskRefreshSuccess(bool success) { m_refreshSuccess = success; }

    int refreshRequestCount() const { return m_refreshRequests; }

Q_SIGNALS:
    void connectionStatusChanged();
    void reduceMotionActiveChanged();
    void offlineModeChanged();
    void offlineDaemonStatusChanged();
    void offlineAutomationRunningChanged();
    void instrumentChanged();
    void performanceGuardChanged();
    void riskRefreshScheduleChanged();
    void activityLogChanged();

private:
    QString    m_connectionStatus;
    bool       m_reduceMotion = false;
    bool       m_offlineMode = false;
    QString    m_offlineStatus;
    bool       m_automationRunning = false;
    QVariantMap m_instrument;
    QVariantMap m_guard;
    QVariantMap m_riskRefresh;
    QVariantList m_activityLog;
    int         m_refreshRequests = 0;
    bool        m_refreshSuccess = true;
};

class MockStrategyController : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool schedulerRunning READ schedulerRunning NOTIFY schedulerStateChanged)

public:
    explicit MockStrategyController(QObject* parent = nullptr)
        : QObject(parent)
    {
    }

    void setSchedulerList(const QVariantList& list)
    {
        if (m_schedulerList == list)
            return;
        m_schedulerList = list;
        Q_EMIT schedulerListChanged();
    }

    void setDecisionConfig(const QVariantMap& config)
    {
        if (m_decisionConfig == config)
            return;
        m_decisionConfig = config;
        Q_EMIT decisionConfigChanged();
    }

    void setOpenPositions(const QVariantList& positions)
    {
        if (m_openPositions == positions)
            return;
        m_openPositions = positions;
        Q_EMIT positionsChanged();
    }

    void setPendingOrders(const QVariantList& orders)
    {
        if (m_pendingOrders == orders)
            return;
        m_pendingOrders = orders;
        Q_EMIT pendingOrdersChanged();
    }

    void setTradeHistory(const QVariantList& trades)
    {
        if (m_tradeHistory == trades)
            return;
        m_tradeHistory = trades;
        Q_EMIT tradeHistoryChanged();
    }

    void setSignalAlerts(const QVariantList& alerts)
    {
        if (m_signalAlerts == alerts)
            return;
        m_signalAlerts = alerts;
        Q_EMIT signalFeedChanged();
    }

    void setSchedulerRunning(bool running)
    {
        if (m_schedulerRunning == running)
            return;
        m_schedulerRunning = running;
        Q_EMIT schedulerStateChanged();
    }

    bool schedulerRunning() const { return m_schedulerRunning; }

    Q_INVOKABLE bool startScheduler()
    {
        setSchedulerRunning(true);
        return true;
    }

    Q_INVOKABLE bool stopScheduler()
    {
        setSchedulerRunning(false);
        return true;
    }

    Q_INVOKABLE QVariantList schedulerList() const { return m_schedulerList; }
    Q_INVOKABLE QVariantMap decisionConfigSnapshot() const { return m_decisionConfig; }
    Q_INVOKABLE QVariantList openPositionsSnapshot() const { return m_openPositions; }
    Q_INVOKABLE QVariantList pendingOrdersSnapshot() const { return m_pendingOrders; }
    Q_INVOKABLE QVariantList tradeHistorySnapshot() const { return m_tradeHistory; }
    Q_INVOKABLE QVariantList signalFeedSnapshot() const { return m_signalAlerts; }

Q_SIGNALS:
    void schedulerListChanged();
    void decisionConfigChanged();
    void schedulerStateChanged();
    void positionsChanged();
    void pendingOrdersChanged();
    void tradeHistoryChanged();
    void signalFeedChanged();

private:
    QVariantList m_schedulerList;
    QVariantMap  m_decisionConfig;
    QVariantList m_openPositions;
    QVariantList m_pendingOrders;
    QVariantList m_tradeHistory;
    QVariantList m_signalAlerts;
    bool         m_schedulerRunning = false;
};

class MockLicenseController : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool licenseActive READ licenseActive NOTIFY licenseActiveChanged)
    Q_PROPERTY(QString licenseEdition READ licenseEdition NOTIFY licenseDataChanged)
    Q_PROPERTY(QString licenseLicenseId READ licenseLicenseId NOTIFY licenseDataChanged)
    Q_PROPERTY(QString licenseHolderName READ licenseHolderName NOTIFY licenseDataChanged)
    Q_PROPERTY(int licenseSeats READ licenseSeats NOTIFY licenseDataChanged)
    Q_PROPERTY(QStringList licenseModules READ licenseModules NOTIFY licenseDataChanged)
    Q_PROPERTY(QStringList licenseRuntime READ licenseRuntime NOTIFY licenseDataChanged)

public:
    explicit MockLicenseController(QObject* parent = nullptr)
        : QObject(parent)
    {
    }

    bool licenseActive() const { return m_active; }
    void setLicenseActive(bool active)
    {
        if (m_active == active)
            return;
        m_active = active;
        Q_EMIT licenseActiveChanged();
    }

    QString licenseEdition() const { return m_edition; }
    void setLicenseEdition(const QString& edition)
    {
        if (m_edition == edition)
            return;
        m_edition = edition;
        Q_EMIT licenseDataChanged();
    }

    QString licenseLicenseId() const { return m_id; }
    void setLicenseId(const QString& id)
    {
        if (m_id == id)
            return;
        m_id = id;
        Q_EMIT licenseDataChanged();
    }

    QString licenseHolderName() const { return m_holder; }
    void setLicenseHolderName(const QString& holder)
    {
        if (m_holder == holder)
            return;
        m_holder = holder;
        Q_EMIT licenseDataChanged();
    }

    int licenseSeats() const { return m_seats; }
    void setLicenseSeats(int seats)
    {
        if (m_seats == seats)
            return;
        m_seats = seats;
        Q_EMIT licenseDataChanged();
    }

    QStringList licenseModules() const { return m_modules; }
    void setLicenseModules(const QStringList& modules)
    {
        if (m_modules == modules)
            return;
        m_modules = modules;
        Q_EMIT licenseDataChanged();
    }

    QStringList licenseRuntime() const { return m_runtime; }
    void setLicenseRuntime(const QStringList& runtime)
    {
        if (m_runtime == runtime)
            return;
        m_runtime = runtime;
        Q_EMIT licenseDataChanged();
    }

    QString licenseIssuedAt() const { return m_issuedAt; }
    void setLicenseIssuedAt(const QString& issued)
    {
        if (m_issuedAt == issued)
            return;
        m_issuedAt = issued;
        Q_EMIT licenseDataChanged();
    }

    QString licenseMaintenanceUntil() const { return m_maintenanceUntil; }
    void setLicenseMaintenanceUntil(const QString& until)
    {
        if (m_maintenanceUntil == until)
            return;
        m_maintenanceUntil = until;
        Q_EMIT licenseDataChanged();
    }

    bool licenseMaintenanceActive() const { return m_maintenanceActive; }
    void setLicenseMaintenanceActive(bool active)
    {
        if (m_maintenanceActive == active)
            return;
        m_maintenanceActive = active;
        Q_EMIT licenseDataChanged();
    }

    QString licenseHolderEmail() const { return m_holderEmail; }
    void setLicenseHolderEmail(const QString& email)
    {
        if (m_holderEmail == email)
            return;
        m_holderEmail = email;
        Q_EMIT licenseDataChanged();
    }

    bool licenseTrialActive() const { return m_trialActive; }
    void setLicenseTrialActive(bool active)
    {
        if (m_trialActive == active)
            return;
        m_trialActive = active;
        Q_EMIT licenseDataChanged();
    }

    QString licenseTrialExpiresAt() const { return m_trialExpires; }
    void setLicenseTrialExpiresAt(const QString& expires)
    {
        if (m_trialExpires == expires)
            return;
        m_trialExpires = expires;
        Q_EMIT licenseDataChanged();
    }

    QStringList licenseEnvironments() const { return m_environments; }
    void setLicenseEnvironments(const QStringList& envs)
    {
        if (m_environments == envs)
            return;
        m_environments = envs;
        Q_EMIT licenseDataChanged();
    }

Q_SIGNALS:
    void licenseActiveChanged();
    void licenseDataChanged();

private:
    bool        m_active = false;
    QString     m_edition;
    QString     m_id;
    QString     m_holder;
    QString     m_holderEmail;
    int         m_seats = 0;
    QStringList m_modules;
    QStringList m_runtime;
    QString     m_issuedAt;
    QString     m_maintenanceUntil;
    bool        m_maintenanceActive = false;
    bool        m_trialActive = false;
    QString     m_trialExpires;
    QStringList m_environments;
};

class StrategyWorkbenchE2ETest : public QObject {
    Q_OBJECT

private slots:
    void shouldExposeLiveDataFromControllers();
    void shouldSwitchDemoMode();
};

static QVariantMap makeInstrument(const QString& exchange,
                                  const QString& symbol,
                                  const QString& venue,
                                  const QString& quote,
                                  const QString& base,
                                  const QString& granularity)
{
    QVariantMap map;
    map.insert(QStringLiteral("exchange"), exchange);
    map.insert(QStringLiteral("symbol"), symbol);
    map.insert(QStringLiteral("venueSymbol"), venue);
    map.insert(QStringLiteral("quoteCurrency"), quote);
    map.insert(QStringLiteral("baseCurrency"), base);
    map.insert(QStringLiteral("granularity"), granularity);
    return map;
}

void StrategyWorkbenchE2ETest::shouldExposeLiveDataFromControllers()
{
    QQmlEngine engine;

    MockAppController appController;
    MockStrategyController strategyController;
    RiskStateModel riskModel;
    RiskHistoryModel riskHistoryModel;
    MockLicenseController licenseController;

    appController.setConnectionStatus(QStringLiteral("Połączono"));
    appController.setReduceMotionActive(false);
    appController.setOfflineMode(false);
    appController.setOfflineDaemonStatus(QStringLiteral("Online"));
    appController.setOfflineAutomationRunning(true);
    appController.setInstrumentConfig(makeInstrument(QStringLiteral("BINANCE"),
                                                     QStringLiteral("BTC/USDT"),
                                                     QStringLiteral("BTCUSDT"),
                                                     QStringLiteral("USDT"),
                                                     QStringLiteral("BTC"),
                                                     QStringLiteral("PT1M")));
    appController.setPerformanceGuard({{QStringLiteral("fpsTarget"), 120}, {QStringLiteral("jankThresholdMs"), 12.0}});
    appController.setRiskRefresh({{QStringLiteral("enabled"), true}, {QStringLiteral("intervalSeconds"), 30}});
    QVariantList initialLog;
    QVariantMap logEntry;
    logEntry.insert(QStringLiteral("timestamp"), QDateTime::currentDateTimeUtc().toString(Qt::ISODate));
    logEntry.insert(QStringLiteral("type"), QStringLiteral("system:init"));
    logEntry.insert(QStringLiteral("message"), QStringLiteral("System ready"));
    logEntry.insert(QStringLiteral("success"), true);
    logEntry.insert(QStringLiteral("source"), QStringLiteral("runtime"));
    initialLog.append(logEntry);
    appController.setActivityLog(initialLog);

    QVariantList schedulers;
    QVariantMap first;
    first.insert(QStringLiteral("name"), QStringLiteral("Momentum Alpha"));
    first.insert(QStringLiteral("enabled"), true);
    first.insert(QStringLiteral("timezone"), QStringLiteral("UTC"));
    first.insert(QStringLiteral("next_run_at"), QStringLiteral("2024-05-18T09:00:00Z"));
    first.insert(QStringLiteral("schedules"), QVariantList{QVariantMap{{QStringLiteral("window"), QStringLiteral("PT15M")}}});
    schedulers.append(first);

    QVariantMap second;
    second.insert(QStringLiteral("name"), QStringLiteral("Mean Reversion"));
    second.insert(QStringLiteral("enabled"), false);
    second.insert(QStringLiteral("timezone"), QStringLiteral("UTC"));
    second.insert(QStringLiteral("schedules"), QVariantList{});
    schedulers.append(second);
    strategyController.setSchedulerList(schedulers);
    strategyController.setDecisionConfig({{QStringLiteral("policy"), QStringLiteral("momentum")}});
    strategyController.setSchedulerRunning(true);

    QVariantList openPositions;
    QVariantMap firstPosition;
    firstPosition.insert(QStringLiteral("symbol"), QStringLiteral("BTC/USDT"));
    firstPosition.insert(QStringLiteral("exchange"), QStringLiteral("BINANCE"));
    firstPosition.insert(QStringLiteral("side"), QStringLiteral("Long"));
    firstPosition.insert(QStringLiteral("quantity"), 0.75);
    firstPosition.insert(QStringLiteral("entryPrice"), 58200.0);
    firstPosition.insert(QStringLiteral("markPrice"), 58400.0);
    firstPosition.insert(QStringLiteral("unrealizedPnl"), 150.0);
    openPositions.append(firstPosition);

    QVariantMap secondPosition;
    secondPosition.insert(QStringLiteral("symbol"), QStringLiteral("ETH/USD"));
    secondPosition.insert(QStringLiteral("exchange"), QStringLiteral("COINBASE"));
    secondPosition.insert(QStringLiteral("side"), QStringLiteral("Long"));
    secondPosition.insert(QStringLiteral("quantity"), 10.0);
    secondPosition.insert(QStringLiteral("entryPrice"), 3010.0);
    secondPosition.insert(QStringLiteral("markPrice"), 3032.0);
    secondPosition.insert(QStringLiteral("unrealizedPnl"), 220.0);
    openPositions.append(secondPosition);
    strategyController.setOpenPositions(openPositions);

    QVariantList pendingOrders;
    QVariantMap firstOrder;
    firstOrder.insert(QStringLiteral("id"), QStringLiteral("ORD-1"));
    firstOrder.insert(QStringLiteral("clientOrderId"), QStringLiteral("CLI-1"));
    firstOrder.insert(QStringLiteral("symbol"), QStringLiteral("BTC/USDT"));
    firstOrder.insert(QStringLiteral("exchange"), QStringLiteral("BINANCE"));
    firstOrder.insert(QStringLiteral("side"), QStringLiteral("Kupno"));
    firstOrder.insert(QStringLiteral("type"), QStringLiteral("Limit"));
    firstOrder.insert(QStringLiteral("quantity"), 0.25);
    firstOrder.insert(QStringLiteral("filled"), 0.1);
    firstOrder.insert(QStringLiteral("remaining"), 0.15);
    firstOrder.insert(QStringLiteral("price"), 58950.0);
    firstOrder.insert(QStringLiteral("averagePrice"), 58890.0);
    firstOrder.insert(QStringLiteral("status"), QStringLiteral("PartiallyFilled"));
    firstOrder.insert(QStringLiteral("timeInForce"), QStringLiteral("GTC"));
    firstOrder.insert(QStringLiteral("createdAt"), QStringLiteral("2024-05-18T08:58:10Z"));
    firstOrder.insert(QStringLiteral("updatedAt"), QStringLiteral("2024-05-18T08:59:30Z"));
    pendingOrders.append(firstOrder);

    QVariantMap secondOrder;
    secondOrder.insert(QStringLiteral("id"), QStringLiteral("ORD-2"));
    secondOrder.insert(QStringLiteral("symbol"), QStringLiteral("ETH/USD"));
    secondOrder.insert(QStringLiteral("exchange"), QStringLiteral("COINBASE"));
    secondOrder.insert(QStringLiteral("side"), QStringLiteral("Sprzedaż"));
    secondOrder.insert(QStringLiteral("type"), QStringLiteral("Stop"));
    secondOrder.insert(QStringLiteral("quantity"), 4.0);
    secondOrder.insert(QStringLiteral("remaining"), 4.0);
    secondOrder.insert(QStringLiteral("price"), 3080.0);
    secondOrder.insert(QStringLiteral("status"), QStringLiteral("Working"));
    secondOrder.insert(QStringLiteral("timeInForce"), QStringLiteral("GTC"));
    secondOrder.insert(QStringLiteral("reduceOnly"), true);
    pendingOrders.append(secondOrder);
    strategyController.setPendingOrders(pendingOrders);

    QVariantList tradeHistory;
    QVariantMap firstTrade;
    firstTrade.insert(QStringLiteral("symbol"), QStringLiteral("BTC/USDT"));
    firstTrade.insert(QStringLiteral("exchange"), QStringLiteral("BINANCE"));
    firstTrade.insert(QStringLiteral("side"), QStringLiteral("Kupno"));
    firstTrade.insert(QStringLiteral("quantity"), 0.3);
    firstTrade.insert(QStringLiteral("price"), 58120.0);
    firstTrade.insert(QStringLiteral("status"), QStringLiteral("Filled"));
    tradeHistory.append(firstTrade);

    QVariantMap secondTrade;
    secondTrade.insert(QStringLiteral("symbol"), QStringLiteral("BTC/USDT"));
    secondTrade.insert(QStringLiteral("exchange"), QStringLiteral("BINANCE"));
    secondTrade.insert(QStringLiteral("side"), QStringLiteral("Sprzedaż"));
    secondTrade.insert(QStringLiteral("quantity"), 0.2);
    secondTrade.insert(QStringLiteral("price"), 58300.0);
    secondTrade.insert(QStringLiteral("status"), QStringLiteral("Filled"));
    tradeHistory.append(secondTrade);
    strategyController.setTradeHistory(tradeHistory);

    QVariantList signalAlerts;
    QVariantMap firstAlert;
    firstAlert.insert(QStringLiteral("id"), QStringLiteral("SIG-1"));
    firstAlert.insert(QStringLiteral("category"), QStringLiteral("momentum"));
    firstAlert.insert(QStringLiteral("symbol"), QStringLiteral("BTC/USDT"));
    firstAlert.insert(QStringLiteral("direction"), QStringLiteral("Long"));
    firstAlert.insert(QStringLiteral("confidence"), 0.82);
    firstAlert.insert(QStringLiteral("impact"), 0.64);
    firstAlert.insert(QStringLiteral("message"), QStringLiteral("Momentum breakout"));
    firstAlert.insert(QStringLiteral("generatedAt"), QStringLiteral("2024-05-18T08:58:45Z"));
    signalAlerts.append(firstAlert);

    QVariantMap secondAlert;
    secondAlert.insert(QStringLiteral("id"), QStringLiteral("SIG-2"));
    secondAlert.insert(QStringLiteral("category"), QStringLiteral("volatility"));
    secondAlert.insert(QStringLiteral("symbol"), QStringLiteral("ETH/USD"));
    secondAlert.insert(QStringLiteral("direction"), QStringLiteral("Long"));
    secondAlert.insert(QStringLiteral("confidence"), 0.65);
    secondAlert.insert(QStringLiteral("impact"), 0.5);
    secondAlert.insert(QStringLiteral("message"), QStringLiteral("Volatility rising"));
    secondAlert.insert(QStringLiteral("generatedAt"), QStringLiteral("2024-05-18T08:53:12Z"));
    signalAlerts.append(secondAlert);
    strategyController.setSignalAlerts(signalAlerts);

    RiskSnapshotData snapshot;
    snapshot.hasData = true;
    snapshot.profileLabel = QStringLiteral("Live");
    snapshot.portfolioValue = 120000.0;
    snapshot.currentDrawdown = 0.015;
    snapshot.maxDailyLoss = 0.05;
    snapshot.usedLeverage = 1.4;
    snapshot.generatedAt = QDateTime::currentDateTimeUtc();
    snapshot.exposures.append(RiskExposureData{QStringLiteral("XBT"), 0.8, 0.6, 0.9});
    snapshot.exposures.append(RiskExposureData{QStringLiteral("ETH"), 0.5, 0.45, 0.7});

    RiskSnapshotData earlierSnapshot = snapshot;
    earlierSnapshot.generatedAt = snapshot.generatedAt.addSecs(-60);
    earlierSnapshot.portfolioValue = 118000.0;
    earlierSnapshot.currentDrawdown = 0.02;

    riskModel.updateFromSnapshot(snapshot);
    riskHistoryModel.recordSnapshot(earlierSnapshot);
    riskHistoryModel.recordSnapshot(snapshot);

    licenseController.setLicenseActive(true);
    licenseController.setLicenseEdition(QStringLiteral("OEM Demo"));
    licenseController.setLicenseId(QStringLiteral("LIVE-123"));
    licenseController.setLicenseHolderName(QStringLiteral("QA"));
    licenseController.setLicenseSeats(8);
    licenseController.setLicenseModules(QStringList{QStringLiteral("momentum"), QStringLiteral("portfolio")});
    licenseController.setLicenseRuntime(QStringList{QStringLiteral("desktop-shell")});

    engine.rootContext()->setContextProperty(QStringLiteral("appController"), &appController);
    engine.rootContext()->setContextProperty(QStringLiteral("strategyController"), &strategyController);
    engine.rootContext()->setContextProperty(QStringLiteral("riskModel"), &riskModel);
    engine.rootContext()->setContextProperty(QStringLiteral("riskHistoryModel"), &riskHistoryModel);
    engine.rootContext()->setContextProperty(QStringLiteral("licenseController"), &licenseController);

    QQmlComponent component(&engine, QUrl(QStringLiteral("qrc:/qml/components/workbench/StrategyWorkbench.qml")));
    QObject* object = component.create(engine.rootContext());
    QVERIFY2(object, qPrintable(component.errorString()));
    QScopedPointer<QObject> guard(object);

    QObject* viewModel = object->findChild<QObject*>(QStringLiteral("strategyWorkbenchViewModel"));
    QVERIFY(viewModel);

    const QVariantList schedulerList = viewModel->property("schedulerEntries").toList();
    QCOMPARE(schedulerList.size(), 2);
    QCOMPARE(schedulerList.first().toMap().value(QStringLiteral("name")).toString(), QStringLiteral("Momentum Alpha"));

    const QVariantList exchanges = viewModel->property("exchangeConnections").toList();
    QCOMPARE(exchanges.size(), 1);
    QCOMPARE(exchanges.first().toMap().value(QStringLiteral("exchange")).toString(), QStringLiteral("BINANCE"));

    const QVariantMap instrument = viewModel->property("instrumentDetails").toMap();
    QCOMPARE(instrument.value(QStringLiteral("exchange")).toString(), QStringLiteral("BINANCE"));
    QCOMPARE(instrument.value(QStringLiteral("quoteCurrency")).toString(), QStringLiteral("USDT"));

    const QVariantMap portfolio = viewModel->property("portfolioSummary").toMap();
    QVERIFY(portfolio.value(QStringLiteral("latestValue")).toDouble() > 0.0);
    QCOMPARE(portfolio.value(QStringLiteral("profileLabel")).toString(), QStringLiteral("Live"));
    QVERIFY(portfolio.contains(QStringLiteral("maxExposureUtilization")));

    const QVariantList positions = viewModel->property("openPositions").toList();
    QCOMPARE(positions.size(), 2);
    QCOMPARE(positions.first().toMap().value(QStringLiteral("symbol")).toString(), QStringLiteral("BTC/USDT"));

    const QVariantList orders = viewModel->property("pendingOrders").toList();
    QCOMPARE(orders.size(), 2);
    QCOMPARE(orders.first().toMap().value(QStringLiteral("status")).toString(), QStringLiteral("PartiallyFilled"));
    QCOMPARE(orders.last().toMap().value(QStringLiteral("timeInForce")).toString(), QStringLiteral("GTC"));

    const QVariantList trades = viewModel->property("tradeHistory").toList();
    QCOMPARE(trades.size(), 2);
    const QString firstStatus = trades.first().toMap().value(QStringLiteral("status")).toString();
    QCOMPARE(firstStatus.compare(QStringLiteral("filled"), Qt::CaseInsensitive), 0);

    const QVariantList alerts = viewModel->property("signalAlerts").toList();
    QCOMPARE(alerts.size(), 2);
    QCOMPARE(alerts.first().toMap().value(QStringLiteral("direction")).toString(), QStringLiteral("Long"));

    const QVariantList riskTimeline = viewModel->property("riskTimeline").toList();
    QVERIFY(!riskTimeline.isEmpty());
    const QVariantMap latestSample = riskTimeline.first().toMap();
    QVERIFY(!latestSample.value(QStringLiteral("timestamp")).toString().isEmpty());
    QVERIFY(latestSample.value(QStringLiteral("portfolioValue")).toDouble() > 0.0);
    QCOMPARE(latestSample.value(QStringLiteral("source")).toString(), QStringLiteral("live"));

    const QVariantMap license = viewModel->property("licenseStatus").toMap();
    QCOMPARE(license.value(QStringLiteral("active")).toBool(), true);
    QCOMPARE(license.value(QStringLiteral("licenseId")).toString(), QStringLiteral("LIVE-123"));

    QVariantList log = viewModel->property("activityLog").toList();
    QCOMPARE(log.size(), 1);
    QCOMPARE(log.first().toMap().value(QStringLiteral("type")).toString(), QStringLiteral("system:init"));

    QVariantMap control = viewModel->property("controlState").toMap();
    QCOMPARE(control.value(QStringLiteral("schedulerRunning")).toBool(), true);
    QCOMPARE(control.value(QStringLiteral("manualRefreshCount")).toInt(), 0);

    QVariant stopResult;
    QVERIFY(QMetaObject::invokeMethod(viewModel, "stopScheduler",
                                      Q_RETURN_ARG(QVariant, stopResult)));
    QVERIFY(stopResult.toBool());
    QVERIFY(!strategyController.schedulerRunning());

    QVariant startResult;
    QVERIFY(QMetaObject::invokeMethod(viewModel, "startScheduler",
                                      Q_RETURN_ARG(QVariant, startResult)));
    QVERIFY(startResult.toBool());
    QVERIFY(strategyController.schedulerRunning());

    QVariant refreshResult;
    QVERIFY(QMetaObject::invokeMethod(viewModel, "triggerRiskRefresh",
                                      Q_RETURN_ARG(QVariant, refreshResult)));
    QVERIFY(refreshResult.toBool());
    QCOMPARE(appController.refreshRequestCount(), 1);

    control = viewModel->property("controlState").toMap();
    QCOMPARE(control.value(QStringLiteral("manualRefreshCount")).toInt(), 1);

    log = viewModel->property("activityLog").toList();
    QVERIFY(!log.isEmpty());
    QCOMPARE(log.first().toMap().value(QStringLiteral("type")).toString(), QStringLiteral("risk:refresh"));
}

void StrategyWorkbenchE2ETest::shouldSwitchDemoMode()
{
    QQmlEngine engine;

    MockAppController appController;
    MockStrategyController strategyController;
    RiskStateModel riskModel;
    RiskHistoryModel riskHistoryModel;
    MockLicenseController licenseController;

    engine.rootContext()->setContextProperty(QStringLiteral("appController"), &appController);
    engine.rootContext()->setContextProperty(QStringLiteral("strategyController"), &strategyController);
    engine.rootContext()->setContextProperty(QStringLiteral("riskModel"), &riskModel);
    engine.rootContext()->setContextProperty(QStringLiteral("riskHistoryModel"), &riskHistoryModel);
    engine.rootContext()->setContextProperty(QStringLiteral("licenseController"), &licenseController);

    QQmlComponent component(&engine, QUrl(QStringLiteral("qrc:/qml/components/workbench/StrategyWorkbench.qml")));
    QObject* object = component.create(engine.rootContext());
    QVERIFY2(object, qPrintable(component.errorString()));
    QScopedPointer<QObject> guard(object);

    QObject* viewModel = object->findChild<QObject*>(QStringLiteral("strategyWorkbenchViewModel"));
    QVERIFY(viewModel);

    QVariant result;
    QVERIFY(QMetaObject::invokeMethod(viewModel, "activateDemoMode",
                                      Q_RETURN_ARG(QVariant, result),
                                      Q_ARG(QVariant, QStringLiteral("momentum"))));
    QVERIFY(result.toBool());
    QVERIFY(viewModel->property("demoModeActive").toBool());
    QCOMPARE(viewModel->property("demoModeTitle").toString(), QStringLiteral("Momentum Pro"));

    QVariantMap runtimeStatus = viewModel->property("runtimeStatus").toMap();
    QCOMPARE(runtimeStatus.value(QStringLiteral("connection")).toString(), QStringLiteral("Demo: połączenie symulacyjne"));

    const QVariantMap demoInstrument = viewModel->property("instrumentDetails").toMap();
    QCOMPARE(demoInstrument.value(QStringLiteral("exchange")).toString(), QStringLiteral("BINANCE"));
    QCOMPARE(demoInstrument.value(QStringLiteral("venueSymbol")).toString(), QStringLiteral("BTCUSDT"));

    const QVariantMap demoPortfolio = viewModel->property("portfolioSummary").toMap();
    QVERIFY(demoPortfolio.value(QStringLiteral("maxExposureUtilization")).toDouble() > 0.0);

    QVariantList demoTimeline = viewModel->property("riskTimeline").toList();
    QVERIFY(demoTimeline.size() >= 4);
    const QVariantMap demoFirstSample = demoTimeline.first().toMap();
    QCOMPARE(demoFirstSample.value(QStringLiteral("portfolioValue")).toDouble(), 129800.0);
    const QString initialTimelineTimestamp = demoFirstSample.value(QStringLiteral("timestamp")).toString();
    const int initialTimelineSize = demoTimeline.size();

    QVariantList demoLog = viewModel->property("activityLog").toList();
    QVERIFY(!demoLog.isEmpty());
    QCOMPARE(demoLog.first().toMap().value(QStringLiteral("type")).toString(), QStringLiteral("scheduler:start"));

    QVariantList demoPositions = viewModel->property("openPositions").toList();
    QVERIFY(demoPositions.size() >= 2);
    QCOMPARE(demoPositions.first().toMap().value(QStringLiteral("symbol")).toString(), QStringLiteral("BTC/USDT"));

    QVariantList demoOrders = viewModel->property("pendingOrders").toList();
    QVERIFY(demoOrders.size() >= 2);
    QCOMPARE(demoOrders.first().toMap().value(QStringLiteral("status")).toString(), QStringLiteral("PartiallyFilled"));

    QVariantList demoTrades = viewModel->property("tradeHistory").toList();
    QVERIFY(demoTrades.size() >= 3);
    const QString demoStatus = demoTrades.first().toMap().value(QStringLiteral("status")).toString();
    QCOMPARE(demoStatus.compare(QStringLiteral("filled"), Qt::CaseInsensitive), 0);

    QVariantList demoAlerts = viewModel->property("signalAlerts").toList();
    QVERIFY(demoAlerts.size() >= 2);
    QCOMPARE(demoAlerts.first().toMap().value(QStringLiteral("direction")).toString(), QStringLiteral("Long"));

    QVariant startResult;
    QVERIFY(QMetaObject::invokeMethod(viewModel, "startScheduler",
                                      Q_RETURN_ARG(QVariant, startResult)));
    QVERIFY(startResult.toBool());

    QVariantMap control = viewModel->property("controlState").toMap();
    QCOMPARE(control.value(QStringLiteral("schedulerRunning")).toBool(), true);

    QVariant refreshResult;
    QVERIFY(QMetaObject::invokeMethod(viewModel, "triggerRiskRefresh",
                                      Q_RETURN_ARG(QVariant, refreshResult)));
    QVERIFY(refreshResult.toBool());

    control = viewModel->property("controlState").toMap();
    QCOMPARE(control.value(QStringLiteral("manualRefreshCount")).toInt(), 3);

    demoTimeline = viewModel->property("riskTimeline").toList();
    QCOMPARE(demoTimeline.size(), initialTimelineSize + 1);
    const QVariantMap refreshedSample = demoTimeline.first().toMap();
    QVERIFY(refreshedSample.value(QStringLiteral("timestamp")).toString() != initialTimelineTimestamp);
    QCOMPARE(refreshedSample.value(QStringLiteral("source")).toString(), QStringLiteral("demo"));

    QVariant stopResult;
    QVERIFY(QMetaObject::invokeMethod(viewModel, "stopScheduler",
                                      Q_RETURN_ARG(QVariant, stopResult)));
    QVERIFY(stopResult.toBool());

    control = viewModel->property("controlState").toMap();
    QCOMPARE(control.value(QStringLiteral("schedulerRunning")).toBool(), false);

    demoLog = viewModel->property("activityLog").toList();
    QVERIFY(!demoLog.isEmpty());
    QCOMPARE(demoLog.first().toMap().value(QStringLiteral("type")).toString(), QStringLiteral("scheduler:stop"));

    QVERIFY(QMetaObject::invokeMethod(viewModel, "disableDemoMode"));
    QVERIFY(!viewModel->property("demoModeActive").toBool());
}

QTEST_MAIN(StrategyWorkbenchE2ETest)
#include "StrategyWorkbenchE2ETest.moc"
