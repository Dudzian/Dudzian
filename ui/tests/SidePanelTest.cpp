#include <QtTest/QtTest>
#include <QQmlComponent>
#include <QQmlContext>
#include <QQmlEngine>
#include <QQuickItem>
#include <QVariant>
#include <memory>
#include <QCoreApplication>
#include <QDateTime>

#include "models/RiskStateModel.hpp"
#include "models/RiskTypes.hpp"
#include "utils/PerformanceGuard.hpp"

class StubAppController final : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString connectionStatus READ connectionStatus WRITE setConnectionStatus NOTIFY connectionStatusChanged)
    Q_PROPERTY(QString instrumentLabel READ instrumentLabel WRITE setInstrumentLabel NOTIFY instrumentLabelChanged)

public:
    explicit StubAppController(QObject* parent = nullptr)
        : QObject(parent) {}

    QString connectionStatus() const { return m_connectionStatus; }
    QString instrumentLabel() const { return m_instrumentLabel; }

    void setConnectionStatus(const QString& status) {
        if (status == m_connectionStatus)
            return;
        m_connectionStatus = status;
        Q_EMIT connectionStatusChanged();
    }

    void setInstrumentLabel(const QString& label) {
        if (label == m_instrumentLabel)
            return;
        m_instrumentLabel = label;
        Q_EMIT instrumentLabelChanged();
    }

signals:
    void connectionStatusChanged();
    void instrumentLabelChanged();

private:
    QString m_connectionStatus = QStringLiteral("connected");
    QString m_instrumentLabel = QStringLiteral("BTC/USDT");
};

class StubOhlcvModel final : public QObject {
    Q_OBJECT

public:
    explicit StubOhlcvModel(QObject* parent = nullptr)
        : QObject(parent) {}

    Q_INVOKABLE double latestClose() const { return m_latestClose; }
    void setLatestClose(double value) { m_latestClose = value; }

private:
    double m_latestClose = 0.0;
};

class SidePanelTest final : public QObject {
    Q_OBJECT

private slots:
    static void initTestCase();
    void init();
    void showsPlaceholdersWithoutRiskData();
    void rendersRiskSnapshotAndExposures();
    void formatsPerformanceGuardValues();

private:
    QObject* createPanel();

    QQmlEngine m_engine;
    RiskStateModel m_riskModel;
    StubAppController m_appController;
    StubOhlcvModel m_ohlcvModel;
};

void SidePanelTest::initTestCase() {
    Q_INIT_RESOURCE(qml);
    qmlRegisterUncreatableType<PerformanceGuard>(
        "BotCore", 1, 0, "PerformanceGuard", QStringLiteral("PerformanceGuard is provided by the controller"));
}

void SidePanelTest::init() {
    // Odśwież kontekst QML przed każdym testem, aby uniknąć wycieków bindingów.
    m_engine.rootContext()->setContextProperty(QStringLiteral("appController"), &m_appController);
    m_engine.rootContext()->setContextProperty(QStringLiteral("riskModel"), &m_riskModel);
    m_engine.rootContext()->setContextProperty(QStringLiteral("ohlcvModel"), &m_ohlcvModel);
}

QObject* SidePanelTest::createPanel() {
    QQmlComponent component(&m_engine, QUrl(QStringLiteral("qrc:/qml/components/SidePanel.qml")));
    if (component.status() != QQmlComponent::Ready) {
        qWarning() << component.errorString();
    }
    QObject* object = component.create();
    Q_ASSERT(object);
    return object;
}

void SidePanelTest::showsPlaceholdersWithoutRiskData() {
    std::unique_ptr<QObject> panel(createPanel());
    QVERIFY(panel != nullptr);

    auto* profileLabel = panel->findChild<QObject*>(QStringLiteral("riskProfileLabel"));
    QVERIFY(profileLabel != nullptr);
    QCOMPARE(profileLabel->property("text").toString(), QStringLiteral("Profil: —"));

    auto* portfolioLabel = panel->findChild<QObject*>(QStringLiteral("riskPortfolioLabel"));
    QVERIFY(portfolioLabel != nullptr);
    const QString portfolioText = portfolioLabel->property("text").toString();
    QVERIFY(portfolioText.startsWith(QStringLiteral("Wartość portfela:")));
    QVERIFY(portfolioText.endsWith(QStringLiteral("—")));

    auto* exposureList = panel->findChild<QObject*>(QStringLiteral("riskExposureList"));
    QVERIFY(exposureList != nullptr);
    QCOMPARE(exposureList->property("visible").toBool(), false);

    auto* latestCloseLabel = panel->findChild<QObject*>(QStringLiteral("latestCloseLabel"));
    QVERIFY(latestCloseLabel != nullptr);
    QCOMPARE(latestCloseLabel->property("text").toString(), QStringLiteral("Latest close: --"));
}

void SidePanelTest::rendersRiskSnapshotAndExposures() {
    std::unique_ptr<QObject> panel(createPanel());
    QVERIFY(panel != nullptr);

    RiskSnapshotData snapshot;
    snapshot.profileLabel = QStringLiteral("Zbalansowany");
    snapshot.portfolioValue = 1'250'000.0;
    snapshot.currentDrawdown = 0.0185;
    snapshot.usedLeverage = 1.37;
    snapshot.generatedAt = QDateTime::fromString(QStringLiteral("2024-04-03T09:30:00Z"), Qt::ISODate);
    snapshot.exposures.append({QStringLiteral("PORTFOLIO_VAR"), 600000.0, 450000.0, 550000.0});
    snapshot.exposures.append({QStringLiteral("MAX_POSITION"), 120000.0, 118000.0, 115000.0});

    m_riskModel.updateFromSnapshot(snapshot);
    QCoreApplication::processEvents();

    auto* profileLabel = panel->findChild<QObject*>(QStringLiteral("riskProfileLabel"));
    QVERIFY(profileLabel != nullptr);
    const QString profileText = profileLabel->property("text").toString();
    QVERIFY(profileText.contains(QStringLiteral("Zbalansowany")));

    auto* drawdownLabel = panel->findChild<QObject*>(QStringLiteral("riskDrawdownLabel"));
    QVERIFY(drawdownLabel != nullptr);
    QVERIFY(drawdownLabel->property("text").toString().contains(QStringLiteral("1.85")));

    auto* leverageLabel = panel->findChild<QObject*>(QStringLiteral("riskLeverageLabel"));
    QVERIFY(leverageLabel != nullptr);
    QVERIFY(leverageLabel->property("text").toString().contains(QStringLiteral("1.37")));

    auto* generatedAtLabel = panel->findChild<QObject*>(QStringLiteral("riskGeneratedAtLabel"));
    QVERIFY(generatedAtLabel != nullptr);
    QCOMPARE(generatedAtLabel->property("text").toString(),
             QStringLiteral("Aktualizacja: 2024-04-03T09:30:00Z"));

    auto* exposureList = panel->findChild<QObject*>(QStringLiteral("riskExposureList"));
    QVERIFY(exposureList != nullptr);
    QVERIFY(exposureList->property("visible").toBool());
    QCOMPARE(exposureList->property("count").toInt(), 2);
}

void SidePanelTest::formatsPerformanceGuardValues() {
    m_ohlcvModel.setLatestClose(101.234);

    std::unique_ptr<QObject> panel(createPanel());
    QVERIFY(panel != nullptr);

    PerformanceGuard guard;
    guard.fpsTarget = 120;
    guard.reduceMotionAfterSeconds = 0.45;
    guard.jankThresholdMs = 15.5;
    guard.maxOverlayCount = 4;
    guard.disableSecondaryWhenFpsBelow = 58;

    panel->setProperty("performanceGuard", QVariant::fromValue(guard));
    QCoreApplication::processEvents();

    auto* fpsValueLabel = panel->findChild<QObject*>(QStringLiteral("fpsTargetValueLabel"));
    QVERIFY(fpsValueLabel != nullptr);
    QCOMPARE(fpsValueLabel->property("text").toString(), QStringLiteral("120"));

    auto* reduceMotionValue = panel->findChild<QObject*>(QStringLiteral("reduceMotionValueLabel"));
    QVERIFY(reduceMotionValue != nullptr);
    QCOMPARE(reduceMotionValue->property("text").toString(), QStringLiteral("0.45 s"));

    auto* jankValueLabel = panel->findChild<QObject*>(QStringLiteral("jankBudgetValueLabel"));
    QVERIFY(jankValueLabel != nullptr);
    QCOMPARE(jankValueLabel->property("text").toString(), QStringLiteral("15.5 ms"));

    auto* overlayValueLabel = panel->findChild<QObject*>(QStringLiteral("overlayLimitValueLabel"));
    QVERIFY(overlayValueLabel != nullptr);
    QCOMPARE(overlayValueLabel->property("text").toString(), QStringLiteral("4"));

    auto* disableSecondaryLabel = panel->findChild<QObject*>(QStringLiteral("disableSecondaryValueLabel"));
    QVERIFY(disableSecondaryLabel != nullptr);
    QCOMPARE(disableSecondaryLabel->property("text").toString(), QStringLiteral("58"));

    auto* latestCloseLabel = panel->findChild<QObject*>(QStringLiteral("latestCloseLabel"));
    QVERIFY(latestCloseLabel != nullptr);
    QCOMPARE(latestCloseLabel->property("text").toString(), QStringLiteral("Latest close: 101.23"));
}

QTEST_MAIN(SidePanelTest)
#include "SidePanelTest.moc"
