#include <QtTest>
#include <QQmlComponent>
#include <QQmlEngine>
#include <QQuickItem>
#include <QColor>
#include <QCoreApplication>
#include <QVariant>

#include "models/IndicatorSeriesModel.hpp"
#include "models/MarketDataStreams.hpp"
#include "models/MarketRegimeTimelineModel.hpp"
#include "models/OhlcvListModel.hpp"
#include "models/SignalListModel.hpp"
#include "utils/PerformanceGuard.hpp"

class MarketMultiStreamViewTest : public QObject {
    Q_OBJECT
private slots:
    static void initTestCase()
    {
        Q_INIT_RESOURCE(qml);
        Q_INIT_RESOURCE(market_multi_stream_view_qml);
        qmlRegisterUncreatableType<PerformanceGuard>(
            "BotCore", 1, 0, "PerformanceGuard", QStringLiteral("PerformanceGuard is injected by the application"));
    }

    void createsWithSampleData();
};

void MarketMultiStreamViewTest::createsWithSampleData()
{
    OhlcvListModel priceModel;
    QList<OhlcvPoint> candles;
    OhlcvPoint point;
    point.timestampMs = 1700000000000;
    point.open = 100.0;
    point.high = 110.0;
    point.low = 95.0;
    point.close = 105.0;
    point.volume = 5.0;
    point.closed = true;
    point.sequence = 1;
    candles.append(point);
    point.timestampMs += 60000;
    point.open = 105.0;
    point.high = 112.0;
    point.low = 102.0;
    point.close = 108.0;
    point.volume = 4.0;
    point.sequence = 2;
    candles.append(point);
    priceModel.resetWithHistory(candles);

    IndicatorSeriesModel indicatorModel;
    QVector<IndicatorSeriesDefinition> defs = {
        {QStringLiteral("ema_fast"), QStringLiteral("EMA 12"), QColor(Qt::yellow), false},
        {QStringLiteral("ema_slow"), QStringLiteral("EMA 26"), QColor(Qt::cyan), true},
        {QStringLiteral("vwap"), QStringLiteral("VWAP"), QColor(Qt::magenta), true},
    };
    indicatorModel.setSeriesDefinitions(defs);
    QVector<IndicatorSample> emaFast = {
        {QStringLiteral("ema_fast"), candles.at(0).timestampMs, 101.0},
        {QStringLiteral("ema_fast"), candles.at(1).timestampMs, 106.0},
    };
    indicatorModel.replaceSamples(QStringLiteral("ema_fast"), emaFast);
    indicatorModel.replaceSamples(QStringLiteral("ema_slow"), emaFast);
    indicatorModel.replaceSamples(QStringLiteral("vwap"), emaFast);

    SignalListModel signalModel;
    QVector<SignalEventEntry> signals = {
        {candles.at(1).timestampMs, QStringLiteral("ema_bullish_cross"), QStringLiteral("Szybka EMA powyżej"), 0.8, QStringLiteral("trend")}
    };
    signalModel.resetWithSignals(signals);

    MarketRegimeTimelineModel regimeModel;
    MarketRegimeSnapshotEntry regimeSnapshot;
    regimeSnapshot.timestampMs = candles.last().timestampMs;
    regimeSnapshot.regime = QStringLiteral("trend");
    regimeSnapshot.trendConfidence = 1.2;
    regimeSnapshot.meanReversionConfidence = 0.4;
    regimeSnapshot.dailyConfidence = 0.6;
    regimeModel.resetWithSnapshots({regimeSnapshot});

    QQmlEngine engine;
    QQmlComponent component(&engine, QUrl(QStringLiteral("qrc:/qml/components/MarketMultiStreamView.qml")));
    QObject* object = component.create();
    QVERIFY2(object, qPrintable(component.errorString()));

    object->setProperty("priceModel", QVariant::fromValue(static_cast<QObject*>(&priceModel)));
    object->setProperty("indicatorModel", QVariant::fromValue(static_cast<QObject*>(&indicatorModel)));
    object->setProperty("signalModel", QVariant::fromValue(static_cast<QObject*>(&signalModel)));
    object->setProperty("regimeModel", QVariant::fromValue(static_cast<QObject*>(&regimeModel)));
    PerformanceGuard guard;
    guard.fpsTarget = 120;
    guard.maxOverlayCount = 3;
    object->setProperty("performanceGuard", QVariant::fromValue(guard));

    QCoreApplication::processEvents();

    auto* priceView = object->findChild<QObject*>("priceView");
    QVERIFY(priceView);
    auto* signalList = object->findChild<QObject*>("signalList");
    QVERIFY(signalList);

    delete object;
}

QTEST_MAIN(MarketMultiStreamViewTest)
#include "MarketMultiStreamViewTest.moc"
