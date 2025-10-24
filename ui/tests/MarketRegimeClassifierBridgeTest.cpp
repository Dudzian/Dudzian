#include <QtTest>
#include <QTemporaryDir>
#include <QFile>
#include <QTextStream>
#include <algorithm>

#include "models/MarketRegimeClassifierBridge.hpp"

class MarketRegimeClassifierBridgeTest : public QObject {
    Q_OBJECT
private slots:
    void classifiesTrendForUpwardSeries();
    void loadsThresholdOverridesFromFile();
};

static QVector<OhlcvPoint> buildHistory(double startPrice, double step, int count)
{
    QVector<OhlcvPoint> history;
    history.reserve(count);
    qint64 timestamp = 1700000000000;
    for (int i = 0; i < count; ++i) {
        OhlcvPoint point;
        point.timestampMs = timestamp;
        point.open = startPrice + step * i;
        point.close = startPrice + step * (i + 1);
        point.high = std::max(point.open, point.close) + 1.0;
        point.low = std::min(point.open, point.close) - 1.0;
        point.volume = 10.0 + i;
        point.closed = true;
        history.append(point);
        timestamp += 60000;
    }
    return history;
}

void MarketRegimeClassifierBridgeTest::classifiesTrendForUpwardSeries()
{
    MarketRegimeClassifierBridge bridge;
    MarketRegimeClassifierBridge::Thresholds thresholds;
    thresholds.minHistory = 20;
    thresholds.trendStrengthThreshold = 0.01;
    thresholds.momentumThreshold = 0.001;
    thresholds.volumeTrendThreshold = 0.05;
    bridge.setThresholds(thresholds);

    const QVector<OhlcvPoint> history = buildHistory(100.0, 0.8, 40);
    const auto snapshot = bridge.classify(history);

    QVERIFY(snapshot.has_value());
    QCOMPARE(snapshot->regime, QStringLiteral("trend"));
    QVERIFY(snapshot->trendConfidence > snapshot->meanReversionConfidence);
}

void MarketRegimeClassifierBridgeTest::loadsThresholdOverridesFromFile()
{
    MarketRegimeClassifierBridge bridge;
    MarketRegimeClassifierBridge::Thresholds thresholds;
    thresholds.minHistory = 10;
    thresholds.trendStrengthThreshold = 0.005;
    bridge.setThresholds(thresholds);

    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString filePath = dir.filePath(QStringLiteral("regime_thresholds.yaml"));
    QFile file(filePath);
    QVERIFY(file.open(QIODevice::WriteOnly | QIODevice::Text));
    QTextStream stream(&file);
    stream << "min_history: 5\n";
    stream << "trend_strength_threshold: 1.0\n";
    stream << "momentum_threshold: 0.5\n";
    stream.flush();
    file.close();

    QVERIFY(bridge.loadThresholdsFromFile(filePath));

    const QVector<OhlcvPoint> history = buildHistory(100.0, 0.1, 30);
    const auto snapshot = bridge.classify(history);

    QVERIFY(snapshot.has_value());
    QVERIFY2(snapshot->regime != QStringLiteral("trend"),
             "Zaktualizowane progi powinny uniemożliwić klasyfikację trendu");
}

QTEST_MAIN(MarketRegimeClassifierBridgeTest)
#include "MarketRegimeClassifierBridgeTest.moc"
