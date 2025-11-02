#include "runtime/OfflineRuntimeService.hpp"

#include <QDate>
#include <QFile>
#include <QLoggingCategory>
#include <QtGlobal>
#include <QTextStream>
#include <QTime>
#include <QVariant>

#include <algorithm>
#include <cmath>
#include <random>

#include "utils/PathUtils.hpp"

Q_LOGGING_CATEGORY(lcOfflineService, "bot.shell.offline.service")

namespace {
constexpr auto kDefaultDatasetPath = "data/sample_ohlcv/trend.csv";

QVariantMap windowDescriptor(qint64 startMs, qint64 endMs)
{
    QVariantMap window;
    if (startMs <= 0 || endMs <= 0 || endMs < startMs)
        return window;
    const QDateTime start = QDateTime::fromMSecsSinceEpoch(startMs, Qt::UTC);
    const QDateTime end = QDateTime::fromMSecsSinceEpoch(endMs, Qt::UTC);
    window.insert(QStringLiteral("start"), start.toString(Qt::ISODate));
    window.insert(QStringLiteral("end"), end.toString(Qt::ISODate));
    window.insert(QStringLiteral("duration_s"), qMax<qint64>(0, (endMs - startMs) / 1000));
    return window;
}
} // namespace

OfflineRuntimeService::OfflineRuntimeService(QObject* parent)
    : QObject(parent)
{
}

void OfflineRuntimeService::setInstrument(const TradingClient::InstrumentConfig& config)
{
    m_instrument = config;
}

void OfflineRuntimeService::setHistoryLimit(int limit)
{
    const int clamped = qMax(1, limit);
    if (m_historyLimit == clamped)
        return;
    m_historyLimit = clamped;
    if (m_running)
        emit historyReady(limitedHistory());
}

void OfflineRuntimeService::setAutoRunEnabled(bool enabled)
{
    if (m_autoRunEnabled == enabled)
        return;
    m_autoRunEnabled = enabled;
    applyAutomationPreference();
    if (m_running)
        emit automationStateChanged(m_automationRunning);
}

void OfflineRuntimeService::setStrategyConfig(const QVariantMap& config)
{
    m_strategyConfig = config;
}

void OfflineRuntimeService::setDatasetPath(const QString& path)
{
    QString normalized = path.trimmed();
    if (!normalized.isEmpty())
        normalized = bot::shell::utils::expandPath(normalized);
    if (normalized == m_datasetPath)
        return;
    m_datasetPath = normalized;
    m_datasetDirty = true;
    if (m_running) {
        ensureDatasetLoaded();
        emit historyReady(limitedHistory());
        emit riskReady(buildRiskSnapshot());
        emit guardReady(buildPerformanceGuard());
    }
}

void OfflineRuntimeService::setAlertPreferences(const QVariantMap& preferences)
{
    if (m_alertPreferences == preferences)
        return;
    m_alertPreferences = preferences;
    if (m_running)
        emit alertPreferencesChanged(m_alertPreferences);
}

QVariantMap OfflineRuntimeService::buildAutoModeSnapshot() const
{
    QVariantMap snapshot;
    snapshot.insert(QStringLiteral("timestamp"), QDateTime::currentDateTimeUtc().toString(Qt::ISODate));
    QVariantMap automation;
    automation.insert(QStringLiteral("enabled"), m_autoRunEnabled);
    automation.insert(QStringLiteral("running"), m_automationRunning);
    snapshot.insert(QStringLiteral("automation"), automation);
    snapshot.insert(QStringLiteral("strategy"), m_strategyConfig);
    snapshot.insert(QStringLiteral("metrics"), buildAutomationMetrics());
    snapshot.insert(QStringLiteral("equity_curve"), buildEquityCurveSeries());
    snapshot.insert(QStringLiteral("risk_heatmap"), buildRiskHeatmapCells());
    snapshot.insert(QStringLiteral("performance"), buildPerformanceSummary());
    snapshot.insert(QStringLiteral("performance_window"), buildRecentPerformanceSummary());
    snapshot.insert(QStringLiteral("alerts"), m_alertPreferences);
    QVariantMap schedule;
    schedule.insert(QStringLiteral("mode"), m_autoRunEnabled ? QStringLiteral("auto") : QStringLiteral("manual"));
    schedule.insert(QStringLiteral("is_open"), true);
    snapshot.insert(QStringLiteral("schedule"), schedule);
    snapshot.insert(QStringLiteral("reasons"), QVariantList());
    snapshot.insert(QStringLiteral("controller_history"), QVariantList());
    snapshot.insert(QStringLiteral("decision_summary"), QVariantMap());
    snapshot.insert(QStringLiteral("guardrail_summary"), QVariantMap());
    return snapshot;
}

void OfflineRuntimeService::start()
{
    ensureDatasetLoaded();
    m_running = true;
    emit connectionStateChanged(tr("Offline runtime: aktywny (in-process)"));
    emit historyReady(limitedHistory());
    emit riskReady(buildRiskSnapshot());
    emit guardReady(buildPerformanceGuard());
    applyAutomationPreference();
    emit automationStateChanged(m_automationRunning);
    emit alertPreferencesChanged(m_alertPreferences);
}

void OfflineRuntimeService::stop()
{
    if (!m_running)
        return;
    m_running = false;
    m_automationRunning = false;
    emit automationStateChanged(false);
    emit connectionStateChanged(tr("Offline runtime: zatrzymany"));
}

void OfflineRuntimeService::refreshRisk()
{
    if (!m_running)
        ensureDatasetLoaded();
    emit riskReady(buildRiskSnapshot());
}

void OfflineRuntimeService::startAutomation()
{
    if (!m_autoRunEnabled)
        m_autoRunEnabled = true;
    applyAutomationPreference();
    if (m_running)
        emit automationStateChanged(m_automationRunning);
}

void OfflineRuntimeService::stopAutomation()
{
    if (m_autoRunEnabled)
        m_autoRunEnabled = false;
    applyAutomationPreference();
    if (m_running)
        emit automationStateChanged(m_automationRunning);
}

void OfflineRuntimeService::ensureDatasetLoaded()
{
    QString candidate = m_datasetPath;
    if (candidate.trimmed().isEmpty())
        candidate = QStringLiteral(kDefaultDatasetPath);
    candidate = bot::shell::utils::expandPath(candidate);

    if (!m_datasetDirty && candidate == m_loadedDatasetPath)
        return;

    QList<OhlcvPoint> loaded;
    QFile file(candidate);
    if (file.exists() && file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QTextStream stream(&file);
        if (!stream.atEnd())
            stream.readLine();
        quint64 sequence = 0;
        while (!stream.atEnd()) {
            const QString line = stream.readLine();
            const QStringList parts = line.split(QLatin1Char(','));
            if (parts.size() < 6)
                continue;
            const QString timestampRaw = parts.at(0).trimmed();
            QDateTime timestamp = QDateTime::fromString(timestampRaw, Qt::ISODate);
            if (!timestamp.isValid())
                timestamp = QDateTime::fromString(timestampRaw, QStringLiteral("yyyy-MM-dd HH:mm:ss"));
            if (!timestamp.isValid())
                continue;
            timestamp.setTimeSpec(Qt::UTC);

            bool ok = false;
            const double open = parts.at(1).toDouble(&ok);
            if (!ok)
                continue;
            const double high = parts.at(2).toDouble(&ok);
            if (!ok)
                continue;
            const double low = parts.at(3).toDouble(&ok);
            if (!ok)
                continue;
            const double close = parts.at(4).toDouble(&ok);
            if (!ok)
                continue;
            const double volume = parts.at(5).toDouble(&ok);
            if (!ok)
                continue;

            OhlcvPoint candle{};
            candle.timestampMs = timestamp.toMSecsSinceEpoch();
            candle.open = open;
            candle.high = high;
            candle.low = low;
            candle.close = close;
            candle.volume = volume;
            candle.closed = true;
            candle.sequence = sequence++;
            loaded.append(candle);
        }
    } else {
        if (!file.exists())
            qCWarning(lcOfflineService) << "Plik datasetu offline nie istnieje" << candidate;
        else
            qCWarning(lcOfflineService) << "Nie udało się otworzyć datasetu offline" << candidate << file.errorString();
    }

    if (loaded.isEmpty()) {
        qCInfo(lcOfflineService) << "Buduję syntetyczny dataset offline";
        buildSyntheticDataset(loaded);
    }

    m_history = loaded;
    m_loadedDatasetPath = candidate;
    m_datasetDirty = false;
}

QList<OhlcvPoint> OfflineRuntimeService::limitedHistory() const
{
    if (m_historyLimit <= 0 || m_historyLimit >= m_history.size())
        return m_history;
    const int start = m_history.size() - m_historyLimit;
    return m_history.mid(start);
}

RiskSnapshotData OfflineRuntimeService::buildRiskSnapshot() const
{
    RiskSnapshotData snapshot{};
    snapshot.profileLabel = tr("Profil offline");
    snapshot.profileEnum = 1;
    snapshot.generatedAt = QDateTime::currentDateTimeUtc();

    if (m_history.isEmpty()) {
        snapshot.hasData = false;
        return snapshot;
    }

    snapshot.hasData = true;

    double lastClose = m_history.constLast().close;
    double maxClose = lastClose;
    double minClose = lastClose;
    double sumClose = 0.0;
    QVector<double> returns;
    returns.reserve(qMax(0, m_history.size() - 1));
    for (int i = 0; i < m_history.size(); ++i) {
        const double close = m_history.at(i).close;
        sumClose += close;
        maxClose = std::max(maxClose, close);
        minClose = std::min(minClose, close);
        if (i > 0) {
            const double prev = m_history.at(i - 1).close;
            if (prev > 0.0)
                returns.append(close / prev - 1.0);
        }
    }
    const double avgClose = sumClose / static_cast<double>(m_history.size());
    const double drawdown = maxClose > 0.0 ? (maxClose - lastClose) / maxClose : 0.0;
    snapshot.portfolioValue = lastClose * 100.0;
    snapshot.currentDrawdown = drawdown;
    snapshot.maxDailyLoss = std::min(0.05, drawdown + 0.02);
    const double deviation = avgClose > 0.0 ? std::abs(lastClose - avgClose) / avgClose : 0.0;
    const double rangePct = avgClose > 0.0 ? (maxClose - minClose) / avgClose : 0.0;

    double variance = 0.0;
    if (!returns.isEmpty()) {
        double mean = 0.0;
        for (double value : returns)
            mean += value;
        mean /= static_cast<double>(returns.size());
        double accum = 0.0;
        for (double value : returns) {
            const double delta = value - mean;
            accum += delta * delta;
        }
        variance = accum / static_cast<double>(returns.size());
    }
    const double volatility = std::sqrt(std::max(0.0, variance));
    snapshot.usedLeverage = 1.0 + std::min(1.5, volatility * 10.0);

    RiskExposureData drawdownExposure;
    drawdownExposure.code = QStringLiteral("drawdown");
    drawdownExposure.maxValue = 0.10;
    drawdownExposure.thresholdValue = 0.07;
    drawdownExposure.currentValue = drawdown;

    RiskExposureData lossExposure;
    lossExposure.code = QStringLiteral("daily_loss");
    lossExposure.maxValue = snapshot.maxDailyLoss;
    lossExposure.thresholdValue = std::max(0.0, snapshot.maxDailyLoss - 0.01);
    lossExposure.currentValue = std::min(snapshot.maxDailyLoss, std::max(drawdown * 0.6, deviation * 0.5));

    RiskExposureData volatilityExposure;
    volatilityExposure.code = QStringLiteral("volatility");
    volatilityExposure.maxValue = 0.20;
    volatilityExposure.thresholdValue = std::min(0.20, std::max(0.08, rangePct * 0.25));
    volatilityExposure.currentValue = std::min(0.20, std::max(volatility * std::sqrt(60.0), rangePct / 12.0));

    snapshot.exposures = {drawdownExposure, lossExposure, volatilityExposure};

    QVariantMap limits;
    limits.insert(QStringLiteral("max_positions"), 12);
    limits.insert(QStringLiteral("max_leverage"), 3.5);
    limits.insert(QStringLiteral("max_position_pct"), 0.25);
    limits.insert(QStringLiteral("daily_loss_limit"), snapshot.maxDailyLoss);
    limits.insert(QStringLiteral("drawdown_limit"), drawdownExposure.maxValue);
    limits.insert(QStringLiteral("target_volatility"), std::min(0.25, std::max(0.08, volatility * 2.5)));
    limits.insert(QStringLiteral("stop_loss_atr_multiple"), 3.2);
    snapshot.limits = limits;

    QVariantMap statistics;
    statistics.insert(QStringLiteral("dailyRealizedPnl"), -snapshot.portfolioValue * std::min(0.02, lossExposure.currentValue));
    statistics.insert(QStringLiteral("grossNotional"), snapshot.portfolioValue * (1.0 + snapshot.usedLeverage * 0.35));
    statistics.insert(QStringLiteral("activePositions"), QVariant::fromValue<int>(std::max(1, static_cast<int>(std::round(snapshot.usedLeverage)))));
    statistics.insert(QStringLiteral("dailyLossPct"), lossExposure.currentValue);
    statistics.insert(QStringLiteral("drawdownPct"), snapshot.currentDrawdown);
    snapshot.statistics = statistics;

    QVariantMap costBreakdown;
    costBreakdown.insert(QStringLiteral("averageCostBps"), 8.75);
    costBreakdown.insert(QStringLiteral("totalCostBps"), 24.0);
    snapshot.costBreakdown = costBreakdown;

    snapshot.killSwitchEngaged = (snapshot.currentDrawdown >= limits.value(QStringLiteral("drawdown_limit")).toDouble())
        || (lossExposure.currentValue >= limits.value(QStringLiteral("daily_loss_limit")).toDouble());

    return snapshot;
}

PerformanceGuard OfflineRuntimeService::buildPerformanceGuard() const
{
    PerformanceGuard guard;
    const double drawdown = !m_history.isEmpty() ? buildRiskSnapshot().currentDrawdown : 0.0;
    guard.fpsTarget = 60;
    guard.reduceMotionAfterSeconds = drawdown > 0.05 ? 0.75 : 1.5;
    guard.jankThresholdMs = 16.0 + std::min(6.0, drawdown * 120.0);
    guard.maxOverlayCount = drawdown > 0.04 ? 2 : 4;
    guard.disableSecondaryWhenFpsBelow = 24;
    return guard;
}

void OfflineRuntimeService::applyAutomationPreference()
{
    if (!m_running) {
        m_automationRunning = false;
        return;
    }
    const bool shouldRun = m_autoRunEnabled;
    if (m_automationRunning == shouldRun)
        return;
    m_automationRunning = shouldRun;
}

void OfflineRuntimeService::buildSyntheticDataset(QList<OhlcvPoint>& out) const
{
    out.clear();
    out.reserve(64);
    const qint64 baseMs = QDateTime(QDate(2023, 1, 1), QTime(0, 0), Qt::UTC).toMSecsSinceEpoch();
    double price = 100.0;
    std::mt19937_64 rng{123456789ULL};
    std::normal_distribution<double> drift(0.05, 0.12);
    for (int i = 0; i < 64; ++i) {
        const double delta = drift(rng);
        const double open = price;
        const double close = std::max(1.0, open + delta);
        const double high = std::max(open, close) + std::abs(drift(rng));
        const double low = std::max(1.0, std::min(open, close) - std::abs(drift(rng)));
        const double volume = 800.0 + std::abs(drift(rng)) * 120.0;
        price = close;

        OhlcvPoint candle{};
        candle.timestampMs = baseMs + static_cast<qint64>(i) * 60 * 1000;
        candle.open = open;
        candle.high = high;
        candle.low = low;
        candle.close = close;
        candle.volume = volume;
        candle.closed = true;
        candle.sequence = static_cast<quint64>(i);
        out.append(candle);
    }
}

QVariantList OfflineRuntimeService::buildEquityCurveSeries() const
{
    QVariantList points;
    if (m_history.isEmpty())
        return points;

    double equity = 100000.0;
    double previous = m_history.first().close;
    for (int i = 0; i < m_history.size(); ++i) {
        const OhlcvPoint& candle = m_history.at(i);
        if (i == 0) {
            equity = 100000.0;
        } else if (previous > 0.0) {
            equity *= candle.close / previous;
        }
        previous = candle.close;

        QVariantMap entry;
        entry.insert(QStringLiteral("timestamp"), QDateTime::fromMSecsSinceEpoch(candle.timestampMs, Qt::UTC).toString(Qt::ISODate));
        entry.insert(QStringLiteral("value"), equity);
        entry.insert(QStringLiteral("source"), QStringLiteral("offline"));
        entry.insert(QStringLiteral("category"), QStringLiteral("auto-mode"));
        points.append(entry);
    }
    return points;
}

QVariantList OfflineRuntimeService::buildRiskHeatmapCells() const
{
    QVariantList cells;
    const RiskSnapshotData snapshot = buildRiskSnapshot();
    for (const RiskExposureData& exposure : snapshot.exposures) {
        QVariantMap cell;
        cell.insert(QStringLiteral("asset"), exposure.code);
        cell.insert(QStringLiteral("label"), exposure.code);
        cell.insert(QStringLiteral("value"), exposure.currentValue);
        QVariantList sources;
        QVariantMap limitEntry;
        limitEntry.insert(QStringLiteral("source"), tr("Limit"));
        limitEntry.insert(QStringLiteral("category"), QStringLiteral("limit"));
        limitEntry.insert(QStringLiteral("value"), exposure.maxValue);
        sources.append(limitEntry);
        QVariantMap thresholdEntry;
        thresholdEntry.insert(QStringLiteral("source"), tr("Próg"));
        thresholdEntry.insert(QStringLiteral("category"), QStringLiteral("threshold"));
        thresholdEntry.insert(QStringLiteral("value"), exposure.thresholdValue);
        sources.append(thresholdEntry);
        cell.insert(QStringLiteral("sources"), sources);
        cells.append(cell);
    }
    return cells;
}

QVariantMap OfflineRuntimeService::buildAutomationMetrics() const
{
    QVariantMap metrics;
    metrics.insert(QStringLiteral("history_size"), m_history.size());
    metrics.insert(QStringLiteral("auto_enabled"), m_autoRunEnabled);
    metrics.insert(QStringLiteral("automation_running"), m_automationRunning);
    if (!m_history.isEmpty())
        metrics.insert(QStringLiteral("last_close"), m_history.constLast().close);
    return metrics;
}

QVariantMap OfflineRuntimeService::buildPerformanceSummary() const
{
    return buildPerformanceSummaryFor(m_history);
}

QVariantMap OfflineRuntimeService::buildRecentPerformanceSummary(int hours) const
{
    if (m_history.isEmpty() || hours <= 0)
        return QVariantMap();

    const qint64 horizonMs = static_cast<qint64>(hours) * 3600 * 1000;
    const qint64 endMs = m_history.constLast().timestampMs;
    const qint64 startMs = endMs - horizonMs;

    QList<OhlcvPoint> subset;
    subset.reserve(m_history.size());
    for (const OhlcvPoint& candle : m_history) {
        if (candle.timestampMs >= startMs)
            subset.append(candle);
    }

    if (subset.size() < 2)
        return QVariantMap();

    QVariantMap summary = buildPerformanceSummaryFor(subset);
    QVariantMap window = summary.value(QStringLiteral("window")).toMap();
    if (window.isEmpty()) {
        window = windowDescriptor(subset.constFirst().timestampMs, subset.constLast().timestampMs);
        if (!window.isEmpty())
            summary.insert(QStringLiteral("window"), window);
    }
    summary.insert(QStringLiteral("hours"), hours);
    summary.insert(QStringLiteral("label"), tr("Ostatnie %1 h").arg(hours));
    return summary;
}

QVariantMap OfflineRuntimeService::buildPerformanceSummaryFor(const QList<OhlcvPoint>& candles) const
{
    QVariantMap summary;
    if (candles.isEmpty())
        return summary;

    summary.insert(QStringLiteral("total"), candles.size());
    summary.insert(QStringLiteral("cycle_count"), candles.size());

    const double firstClose = candles.constFirst().close;
    const double lastClose = candles.constLast().close;
    if (!qFuzzyIsNull(firstClose))
        summary.insert(QStringLiteral("net_return_pct"), (lastClose - firstClose) / firstClose);

    double peak = firstClose;
    double maxDrawdown = 0.0;
    for (const OhlcvPoint& candle : candles) {
        peak = std::max(peak, candle.close);
        if (peak > 0.0) {
            const double drawdown = (peak - candle.close) / peak;
            if (drawdown > maxDrawdown)
                maxDrawdown = drawdown;
        }
    }
    summary.insert(QStringLiteral("max_drawdown_pct"), maxDrawdown);

    double previousClose = firstClose;
    double sumReturns = 0.0;
    double sumSquares = 0.0;
    int sampleCount = 0;
    for (int index = 1; index < candles.size(); ++index) {
        const double close = candles.at(index).close;
        if (previousClose > 0.0) {
            const double ret = (close - previousClose) / previousClose;
            sumReturns += ret;
            sumSquares += ret * ret;
            ++sampleCount;
        }
        previousClose = close;
    }

    if (sampleCount > 0) {
        const double mean = sumReturns / static_cast<double>(sampleCount);
        const double variance = std::max(0.0, (sumSquares / static_cast<double>(sampleCount)) - (mean * mean));
        summary.insert(QStringLiteral("avg_return_pct"), mean);
        summary.insert(QStringLiteral("volatility_pct"), std::sqrt(variance));
    }

    const QVariantMap window = windowDescriptor(candles.constFirst().timestampMs, candles.constLast().timestampMs);
    if (!window.isEmpty())
        summary.insert(QStringLiteral("window"), window);

    return summary;
}
