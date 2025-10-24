#include "MarketRegimeClassifierBridge.hpp"

#include <QFile>
#include <QTextStream>
#include <QtMath>

#include <numeric>

namespace {

QString trimComment(QString line)
{
    const int hashPos = line.indexOf(QLatin1Char('#'));
    if (hashPos >= 0) {
        line.truncate(hashPos);
    }
    return line.trimmed();
}

} // namespace

MarketRegimeClassifierBridge::MarketRegimeClassifierBridge(QObject* parent)
    : QObject(parent)
{
}

void MarketRegimeClassifierBridge::setThresholds(const Thresholds& thresholds)
{
    m_thresholds = thresholds;
}

bool MarketRegimeClassifierBridge::loadThresholdsFromFile(const QString& path)
{
    QFile file(path);
    if (!file.exists() || !file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return false;
    }

    Thresholds loaded = m_thresholds;
    QTextStream stream(&file);
    while (!stream.atEnd()) {
        const QString rawLine = stream.readLine();
        const QString line = trimComment(rawLine);
        if (line.isEmpty()) {
            continue;
        }
        const int colon = line.indexOf(QLatin1Char(':'));
        if (colon <= 0) {
            continue;
        }
        const QString key = line.left(colon).trimmed();
        const QString value = line.mid(colon + 1).trimmed();
        bool ok = false;
        if (key == QLatin1String("min_history")) {
            const int parsed = value.toInt(&ok);
            if (ok) loaded.minHistory = parsed;
        } else if (key == QLatin1String("trend_window")) {
            const int parsed = value.toInt(&ok);
            if (ok) loaded.trendWindow = parsed;
        } else if (key == QLatin1String("daily_window")) {
            const int parsed = value.toInt(&ok);
            if (ok) loaded.dailyWindow = parsed;
        } else if (key == QLatin1String("trend_strength_threshold")) {
            const double parsed = value.toDouble(&ok);
            if (ok) loaded.trendStrengthThreshold = parsed;
        } else if (key == QLatin1String("momentum_threshold")) {
            const double parsed = value.toDouble(&ok);
            if (ok) loaded.momentumThreshold = parsed;
        } else if (key == QLatin1String("volatility_threshold")) {
            const double parsed = value.toDouble(&ok);
            if (ok) loaded.volatilityThreshold = parsed;
        } else if (key == QLatin1String("intraday_threshold")) {
            const double parsed = value.toDouble(&ok);
            if (ok) loaded.intradayThreshold = parsed;
        } else if (key == QLatin1String("autocorr_threshold")) {
            const double parsed = value.toDouble(&ok);
            if (ok) loaded.autocorrThreshold = parsed;
        } else if (key == QLatin1String("volume_trend_threshold")) {
            const double parsed = value.toDouble(&ok);
            if (ok) loaded.volumeTrendThreshold = parsed;
        }
    }
    setThresholds(loaded);
    return true;
}

std::optional<MarketRegimeSnapshotEntry> MarketRegimeClassifierBridge::classify(const QVector<OhlcvPoint>& history) const
{
    if (history.size() < m_thresholds.minHistory) {
        return std::nullopt;
    }

    const int trendWindow = qMin(m_thresholds.trendWindow, history.size());
    const int dailyWindow = qMin(m_thresholds.dailyWindow, history.size());

    const double trendStrength = computeReturn(history, trendWindow);
    const double momentum = computeMomentum(history, trendWindow);
    const double volatility = computeVolatility(history, trendWindow);
    const double intraday = computeIntradayRange(history, dailyWindow);
    const double autocorr = computeAutocorrelation(history, trendWindow);
    const double volumeTrend = computeVolumeTrend(history, trendWindow);

    QString regime = QStringLiteral("mean_reversion");
    double trendConfidence = qBound(0.0, trendStrength / m_thresholds.trendStrengthThreshold, 2.0);
    double momentumScore = qBound(0.0, momentum / m_thresholds.momentumThreshold, 2.0);
    double volatilityScore = 1.0 - qMin(1.0, volatility / (m_thresholds.volatilityThreshold * 2.0));

    if (trendStrength > m_thresholds.trendStrengthThreshold && momentum > m_thresholds.momentumThreshold && volumeTrend > m_thresholds.volumeTrendThreshold) {
        regime = QStringLiteral("trend");
    } else if (volatility < m_thresholds.volatilityThreshold && intraday < m_thresholds.intradayThreshold) {
        regime = QStringLiteral("daily");
    } else if (autocorr < m_thresholds.autocorrThreshold) {
        regime = QStringLiteral("mean_reversion");
    }

    MarketRegimeSnapshotEntry snapshot;
    snapshot.timestampMs = history.last().timestampMs;
    snapshot.regime = regime;
    snapshot.trendConfidence = qBound(0.0, (trendConfidence + momentumScore + qMax(0.0, volumeTrend)) / 3.0, 1.5);
    snapshot.meanReversionConfidence = qBound(0.0, qAbs(autocorr) + (1.0 - trendConfidence / 2.0), 2.0);
    snapshot.dailyConfidence = qBound(0.0, volatilityScore + (1.0 - intraday / m_thresholds.intradayThreshold), 2.0);
    return snapshot;
}

double MarketRegimeClassifierBridge::computeReturn(const QVector<OhlcvPoint>& history, int window) const
{
    if (history.size() < window || window <= 1) {
        return 0.0;
    }
    const double start = history[history.size() - window].close;
    const double end = history.last().close;
    if (qFuzzyIsNull(start)) {
        return 0.0;
    }
    return (end - start) / start;
}

double MarketRegimeClassifierBridge::computeMomentum(const QVector<OhlcvPoint>& history, int window) const
{
    if (history.size() < 2) {
        return 0.0;
    }
    const int sampleCount = qMin(window, history.size() - 1);
    double sum = 0.0;
    for (int i = history.size() - sampleCount; i < history.size(); ++i) {
        const double prev = history[i - 1].close;
        const double current = history[i].close;
        if (!qFuzzyIsNull(prev)) {
            sum += (current - prev) / prev;
        }
    }
    return sum / sampleCount;
}

double MarketRegimeClassifierBridge::computeVolatility(const QVector<OhlcvPoint>& history, int window) const
{
    if (history.size() < 2) {
        return 0.0;
    }
    const int startIndex = qMax(1, history.size() - window);
    QVector<double> returns;
    returns.reserve(history.size() - startIndex);
    for (int i = startIndex; i < history.size(); ++i) {
        const double prev = history[i - 1].close;
        const double current = history[i].close;
        if (!qFuzzyIsNull(prev)) {
            returns.append((current - prev) / prev);
        }
    }
    if (returns.isEmpty()) {
        return 0.0;
    }
    const double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double variance = 0.0;
    for (double value : returns) {
        const double diff = value - mean;
        variance += diff * diff;
    }
    variance /= returns.size();
    return std::sqrt(variance);
}

double MarketRegimeClassifierBridge::computeIntradayRange(const QVector<OhlcvPoint>& history, int window) const
{
    if (history.isEmpty()) {
        return 0.0;
    }
    const int startIndex = qMax(0, history.size() - window);
    double sum = 0.0;
    int samples = 0;
    for (int i = startIndex; i < history.size(); ++i) {
        const auto& candle = history[i];
        if (!qFuzzyIsNull(candle.open)) {
            sum += (candle.high - candle.low) / candle.open;
            ++samples;
        }
    }
    if (samples == 0) {
        return 0.0;
    }
    return sum / samples;
}

double MarketRegimeClassifierBridge::computeAutocorrelation(const QVector<OhlcvPoint>& history, int window) const
{
    if (history.size() < window || window < 3) {
        return 0.0;
    }
    QVector<double> returns;
    returns.reserve(window - 1);
    const int startIndex = history.size() - window;
    for (int i = startIndex + 1; i < history.size(); ++i) {
        const double prev = history[i - 1].close;
        const double current = history[i].close;
        if (!qFuzzyIsNull(prev)) {
            returns.append((current - prev) / prev);
        }
    }
    if (returns.size() < 2) {
        return 0.0;
    }
    const double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double numerator = 0.0;
    double denom = 0.0;
    for (int i = 1; i < returns.size(); ++i) {
        const double x1 = returns[i - 1] - mean;
        const double x2 = returns[i] - mean;
        numerator += x1 * x2;
        denom += x1 * x1;
    }
    if (qFuzzyIsNull(denom)) {
        return 0.0;
    }
    return numerator / denom;
}

double MarketRegimeClassifierBridge::computeVolumeTrend(const QVector<OhlcvPoint>& history, int window) const
{
    if (history.size() < 2) {
        return 0.0;
    }
    const int startIndex = qMax(1, history.size() - window);
    QVector<double> changes;
    changes.reserve(history.size() - startIndex);
    for (int i = startIndex; i < history.size(); ++i) {
        const double prev = history[i - 1].volume;
        const double current = history[i].volume;
        if (!qFuzzyIsNull(prev)) {
            changes.append((current - prev) / prev);
        }
    }
    if (changes.isEmpty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (double change : changes) {
        sum += change;
    }
    return sum / changes.size();
}

