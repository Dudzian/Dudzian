#include <QDateTime>
#include <QDebug>
#include <QQmlComponent>
#include <QQmlEngine>
#include <QQuickItem>
#include <QtTest/QTest>
#include <QVariant>
#include <QVector>
#include <QHash>

#include <memory>

#include "utils/PerformanceGuard.hpp"

class DummyOhlcvModel : public QObject {
    Q_OBJECT
    Q_PROPERTY(int count READ count CONSTANT)

public:
    explicit DummyOhlcvModel(QObject* parent = nullptr)
        : QObject(parent) {
        const qint64 baseTs = QDateTime(QDate(2023, 1, 1), QTime(12, 0), Qt::UTC).toMSecsSinceEpoch();
        m_candles.append(buildCandle(baseTs, 100.0, 102.0, 99.5, 101.0));
        m_candles.append(buildCandle(baseTs + 60'000, 101.0, 103.5, 100.5, 103.0));
        m_candles.append(buildCandle(baseTs + 120'000, 103.0, 104.0, 101.5, 102.2));

        QVariantList emaFast;
        QVariantList emaSlow;
        QVariantList vwap;
        for (const auto& candle : m_candles) {
            const qint64 ts = candle.value("timestamp").toLongLong();
            emaFast.append(buildOverlaySample(ts, candle.value("close").toDouble() - 0.4));
            emaSlow.append(buildOverlaySample(ts, candle.value("close").toDouble() - 0.9));
            vwap.append(buildOverlaySample(ts, candle.value("close").toDouble() - 0.2));
        }
        m_overlays.insert(QStringLiteral("ema_fast"), emaFast);
        m_overlays.insert(QStringLiteral("ema_slow"), emaSlow);
        m_overlays.insert(QStringLiteral("vwap"), vwap);
    }

    [[nodiscard]] int count() const { return m_candles.size(); }

    Q_INVOKABLE QVariantMap candleAt(int index) const {
        if (index < 0 || index >= m_candles.size()) {
            return {};
        }
        return m_candles.at(index);
    }

    Q_INVOKABLE QVariantList overlaySeries(const QString& key) const {
        return m_overlays.value(key);
    }

signals:
    void modelReset();
    void rowsInserted(const QVariant& parent, int first, int last);
    void dataChanged(const QVariantMap& topLeft, const QVariantMap& bottomRight, const QVariantList& roles);

private:
    static QVariantMap buildCandle(qint64 timestamp, double open, double high, double low, double close) {
        QVariantMap candle;
        candle.insert(QStringLiteral("timestamp"), timestamp);
        candle.insert(QStringLiteral("open"), open);
        candle.insert(QStringLiteral("high"), high);
        candle.insert(QStringLiteral("low"), low);
        candle.insert(QStringLiteral("close"), close);
        return candle;
    }

    static QVariantMap buildOverlaySample(qint64 timestamp, double value) {
        QVariantMap sample;
        sample.insert(QStringLiteral("timestamp"), timestamp);
        sample.insert(QStringLiteral("value"), value);
        return sample;
    }

    QVector<QVariantMap> m_candles;
    QHash<QString, QVariantList> m_overlays;
};

class CandlestickChartViewTest final : public QObject {
    Q_OBJECT

private slots:
    static void initTestCase() {
        Q_INIT_RESOURCE(qml);
        qmlRegisterUncreatableType<PerformanceGuard>(
            "BotCore", 1, 0, "PerformanceGuard", QStringLiteral("PerformanceGuard is provided by the controller"));
    }

    void testOverlayVisibilityPrimaryAndSecondary();
    void testReduceMotionDisablesSecondary();
    void testSecondaryThresholdDisablesOverlays();

private:
    [[nodiscard]] QObject* buildChart() {
        QQmlComponent component(&m_engine, QUrl(QStringLiteral("qrc:/qml/components/CandlestickChartView.qml")));
        if (component.status() != QQmlComponent::Ready) {
            qWarning() << component.errorString();
        }
        QObject* chart = component.create();
        Q_ASSERT(chart);

        auto* model = new DummyOhlcvModel(chart);
        chart->setProperty("model", QVariant::fromValue(static_cast<QObject*>(model)));
        PerformanceGuard guard;
        guard.maxOverlayCount = 3;
        guard.disableSecondaryWhenFpsBelow = 0;
        guard.fpsTarget = 60;
        chart->setProperty("performanceGuard", QVariant::fromValue(guard));
        QMetaObject::invokeMethod(chart, "refreshOverlayVisibility", Qt::DirectConnection);
        return chart;
    }

    [[nodiscard]] static QObject* findSeries(QObject* chart, const char* objectName) {
        auto* series = chart->findChild<QObject*>(objectName, Qt::FindChildrenRecursively);
        Q_ASSERT(series);
        return series;
    }

    QQmlEngine m_engine;
};

void CandlestickChartViewTest::testOverlayVisibilityPrimaryAndSecondary() {
    std::unique_ptr<QObject> chart(buildChart());
    auto* fast = findSeries(chart.get(), "emaFastSeries");
    auto* slow = findSeries(chart.get(), "emaSlowSeries");
    auto* vwap = findSeries(chart.get(), "vwapSeries");

    QVERIFY(fast->property("visible").toBool());
    QVERIFY(slow->property("visible").toBool());
    QVERIFY(vwap->property("visible").toBool());
}

void CandlestickChartViewTest::testReduceMotionDisablesSecondary() {
    std::unique_ptr<QObject> chart(buildChart());
    chart->setProperty("reduceMotion", true);
    QMetaObject::invokeMethod(chart.get(), "refreshOverlayVisibility", Qt::DirectConnection);

    auto* fast = findSeries(chart.get(), "emaFastSeries");
    auto* slow = findSeries(chart.get(), "emaSlowSeries");
    auto* vwap = findSeries(chart.get(), "vwapSeries");

    QVERIFY(fast->property("visible").toBool());
    QVERIFY(!slow->property("visible").toBool());
    QVERIFY(!vwap->property("visible").toBool());
}

void CandlestickChartViewTest::testSecondaryThresholdDisablesOverlays() {
    std::unique_ptr<QObject> chart(buildChart());
    PerformanceGuard guard;
    guard.maxOverlayCount = 3;
    guard.fpsTarget = 60;
    guard.disableSecondaryWhenFpsBelow = 90;
    chart->setProperty("performanceGuard", QVariant::fromValue(guard));
    chart->setProperty("reduceMotion", false);
    QMetaObject::invokeMethod(chart.get(), "refreshOverlayVisibility", Qt::DirectConnection);

    auto* fast = findSeries(chart.get(), "emaFastSeries");
    auto* slow = findSeries(chart.get(), "emaSlowSeries");
    auto* vwap = findSeries(chart.get(), "vwapSeries");

    QVERIFY(fast->property("visible").toBool());
    QVERIFY(!slow->property("visible").toBool());
    QVERIFY(!vwap->property("visible").toBool());
}

QTEST_MAIN(CandlestickChartViewTest)
#include "CandlestickChartViewTest.moc"
