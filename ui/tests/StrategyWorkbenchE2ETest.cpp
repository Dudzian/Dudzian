#include <QtTest/QtTest>

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

    Q_INVOKABLE bool triggerRiskRefresh()
    {
        ++m_refreshRequests;
        if (m_refreshSuccess) {
            m_riskRefresh.insert(QStringLiteral("lastRequestAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODate));
            Q_EMIT riskRefreshScheduleChanged();
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

private:
    QString    m_connectionStatus;
    bool       m_reduceMotion = false;
    bool       m_offlineMode = false;
    QString    m_offlineStatus;
    bool       m_automationRunning = false;
    QVariantMap m_instrument;
    QVariantMap m_guard;
    QVariantMap m_riskRefresh;
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

Q_SIGNALS:
    void schedulerListChanged();
    void decisionConfigChanged();
    void schedulerStateChanged();

private:
    QVariantList m_schedulerList;
    QVariantMap  m_decisionConfig;
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

    riskModel.updateFromSnapshot(snapshot);
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

    const QVariantMap license = viewModel->property("licenseStatus").toMap();
    QCOMPARE(license.value(QStringLiteral("active")).toBool(), true);
    QCOMPARE(license.value(QStringLiteral("licenseId")).toString(), QStringLiteral("LIVE-123"));

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

    QVariant stopResult;
    QVERIFY(QMetaObject::invokeMethod(viewModel, "stopScheduler",
                                      Q_RETURN_ARG(QVariant, stopResult)));
    QVERIFY(stopResult.toBool());

    control = viewModel->property("controlState").toMap();
    QCOMPARE(control.value(QStringLiteral("schedulerRunning")).toBool(), false);

    QVERIFY(QMetaObject::invokeMethod(viewModel, "disableDemoMode"));
    QVERIFY(!viewModel->property("demoModeActive").toBool());
}

QTEST_MAIN(StrategyWorkbenchE2ETest)
#include "StrategyWorkbenchE2ETest.moc"
