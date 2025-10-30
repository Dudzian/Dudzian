#include "runtime/OfflineRuntimeService.hpp"

#include <QDate>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QLoggingCategory>
#include <QTextStream>
#include <QTime>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#include "utils/PathUtils.hpp"

Q_LOGGING_CATEGORY(lcOfflineService, "bot.shell.offline.service")

namespace {
constexpr auto kDefaultDatasetPath = "data/sample_ohlcv/trend.csv";
}

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

void OfflineRuntimeService::setStreamingEnabled(bool enabled)
{
    if (m_streamingEnabled == enabled)
        return;
    m_streamingEnabled = enabled;
    m_datasetDirty = true;
    if (m_running) {
        ensureDatasetLoaded();
        emit historyReady(limitedHistory());
        emit riskReady(buildRiskSnapshot());
        emit guardReady(buildPerformanceGuard());
    }
}

void OfflineRuntimeService::setStreamSnapshotPath(const QString& path)
{
    QString normalized = path.trimmed();
    if (!normalized.isEmpty())
        normalized = bot::shell::utils::expandPath(normalized);
    if (normalized == m_streamSnapshotPath)
        return;
    m_streamSnapshotPath = normalized;
    m_datasetDirty = true;
    if (m_running) {
        ensureDatasetLoaded();
        emit historyReady(limitedHistory());
        emit riskReady(buildRiskSnapshot());
        emit guardReady(buildPerformanceGuard());
    }
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
    if (m_streamingEnabled && !m_streamSnapshotPath.trimmed().isEmpty()) {
        const QString streamPath = bot::shell::utils::expandPath(m_streamSnapshotPath);
        if (!m_datasetDirty && streamPath == m_loadedStreamSnapshotPath)
            return;
        QList<OhlcvPoint> streamed;
        if (tryLoadStreamSnapshot(streamed)) {
            m_history = streamed;
            m_loadedStreamSnapshotPath = streamPath;
            m_loadedDatasetPath.clear();
            m_datasetDirty = false;
            return;
        }
        qCWarning(lcOfflineService) << "Nie udało się załadować snapshotu strumieniowego" << streamPath;
    }

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
    m_loadedStreamSnapshotPath.clear();
    m_datasetDirty = false;
}

bool OfflineRuntimeService::tryLoadStreamSnapshot(QList<OhlcvPoint>& out) const
{
    out.clear();
    if (m_streamSnapshotPath.trimmed().isEmpty())
        return false;

    const QString path = bot::shell::utils::expandPath(m_streamSnapshotPath);
    QFile file(path);
    if (!file.exists() || !file.open(QIODevice::ReadOnly | QIODevice::Text))
        return false;

    const QByteArray payload = file.readAll();
    QJsonParseError error{};
    const QJsonDocument doc = QJsonDocument::fromJson(payload, &error);
    if (error.error != QJsonParseError::NoError || !doc.isArray())
        return false;

    const QJsonArray array = doc.array();
    if (array.isEmpty())
        return false;

    QList<OhlcvPoint> parsed;
    parsed.reserve(array.size());
    quint64 sequence = 0;
    for (const QJsonValue& value : array) {
        if (!value.isObject())
            continue;
        const QJsonObject object = value.toObject();
        const QJsonValue timestampValue = object.value(QStringLiteral("timestamp_ms"));
        if (!timestampValue.isDouble() && !timestampValue.isString())
            continue;
        qint64 timestampMs = -1;
        if (timestampValue.isDouble())
            timestampMs = static_cast<qint64>(timestampValue.toDouble());
        else
            timestampMs = timestampValue.toString().toLongLong();
        if (timestampMs <= 0)
            continue;

        if (!object.value(QStringLiteral("open")).isDouble()
            || !object.value(QStringLiteral("high")).isDouble()
            || !object.value(QStringLiteral("low")).isDouble()
            || !object.value(QStringLiteral("close")).isDouble()) {
            continue;
        }

        const double volume = object.value(QStringLiteral("volume")).toDouble(0.0);

        OhlcvPoint candle{};
        candle.timestampMs = timestampMs;
        candle.open = object.value(QStringLiteral("open")).toDouble();
        candle.high = object.value(QStringLiteral("high")).toDouble();
        candle.low = object.value(QStringLiteral("low")).toDouble();
        candle.close = object.value(QStringLiteral("close")).toDouble();
        candle.volume = volume;
        candle.closed = object.value(QStringLiteral("closed")).toBool(true);
        candle.sequence = sequence++;
        parsed.append(candle);
    }

    if (parsed.isEmpty())
        return false;

    std::sort(parsed.begin(), parsed.end(), [](const OhlcvPoint& lhs, const OhlcvPoint& rhs) {
        if (lhs.timestampMs == rhs.timestampMs)
            return lhs.sequence < rhs.sequence;
        return lhs.timestampMs < rhs.timestampMs;
    });

    QList<OhlcvPoint> normalized;
    normalized.reserve(parsed.size());
    qint64 lastTimestamp = std::numeric_limits<qint64>::min();
    for (const OhlcvPoint& candle : parsed) {
        if (!normalized.isEmpty() && candle.timestampMs == lastTimestamp) {
            normalized.last() = candle;
        } else {
            normalized.append(candle);
            lastTimestamp = candle.timestampMs;
        }
    }

    for (int index = 0; index < normalized.size(); ++index)
        normalized[index].sequence = static_cast<quint64>(index);

    out = normalized;
    return true;
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
