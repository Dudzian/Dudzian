#include <QtTest/QtTest>

#include <QSignalSpy>
#include <QTemporaryDir>
#include <QTemporaryFile>

#include "runtime/RuntimeDecisionBridge.hpp"

class RuntimeDecisionBridgeTest : public QObject {
    Q_OBJECT

private slots:
    void loadsDecisionsFromJsonl();
    void reportsErrorsWhenFileMissing();
};

void RuntimeDecisionBridgeTest::loadsDecisionsFromJsonl()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());

    QFile file(dir.filePath("journal.jsonl"));
    QVERIFY(file.open(QIODevice::WriteOnly | QIODevice::Text));

    const QByteArray sample = R"({
        "event": "order_submitted",
        "timestamp": "2025-01-01T12:00:00+00:00",
        "environment": "prod",
        "portfolio": "alpha",
        "risk_profile": "balanced",
        "strategy": "mean_reversion",
        "schedule": "auto",
        "symbol": "BTC/USDT",
        "side": "buy",
        "status": "submitted",
        "decision_state": "trade",
        "decision_should_trade": "true",
        "decision_signal": "long",
        "market_regime": "bull",
        "market_regime_risk_level": "elevated",
        "ai_probability": 0.84,
        "strategy_recommendation": "momentum_v2"
    })";
    file.write(sample);
    file.write("\n");
    file.close();

    RuntimeDecisionBridge bridge;
    bridge.setLogPath(file.fileName());

    QSignalSpy decisionsSpy(&bridge, &RuntimeDecisionBridge::decisionsChanged);
    const QVariantList result = bridge.loadRecentDecisions(10);

    QCOMPARE(decisionsSpy.count(), 1);
    QCOMPARE(result.size(), 1);
    const QVariantMap entry = result.first().toMap();
    QCOMPARE(entry.value(QStringLiteral("event")).toString(), QStringLiteral("order_submitted"));
    QCOMPARE(entry.value(QStringLiteral("portfolio")).toString(), QStringLiteral("alpha"));
    const QVariantMap decision = entry.value(QStringLiteral("decision")).toMap();
    QVERIFY(decision.value(QStringLiteral("shouldTrade")).toBool());
    QCOMPARE(decision.value(QStringLiteral("state")).toString(), QStringLiteral("trade"));
    const QVariantMap regime = entry.value(QStringLiteral("marketRegime")).toMap();
    QCOMPARE(regime.value(QStringLiteral("regime")).toString(), QStringLiteral("bull"));
    const QVariantMap metadata = entry.value(QStringLiteral("metadata")).toMap();
    QCOMPARE(metadata.value(QStringLiteral("strategy_recommendation")).toString(), QStringLiteral("momentum_v2"));
}

void RuntimeDecisionBridgeTest::reportsErrorsWhenFileMissing()
{
    RuntimeDecisionBridge bridge;
    bridge.setLogPath(QStringLiteral("/nonexistent/journal.jsonl"));

    QSignalSpy errorSpy(&bridge, &RuntimeDecisionBridge::errorMessageChanged);
    const QVariantList result = bridge.loadRecentDecisions(5);

    QVERIFY(result.isEmpty());
    QVERIFY(errorSpy.count() >= 1);
    QVERIFY(!bridge.errorMessage().isEmpty());
}

QTEST_MAIN(RuntimeDecisionBridgeTest)
#include "RuntimeDecisionBridgeTest.moc"

