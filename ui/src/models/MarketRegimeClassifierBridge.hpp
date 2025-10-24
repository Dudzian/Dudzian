#pragma once

#include <QObject>
#include <QString>
#include <optional>

#include "models/OhlcvListModel.hpp"
#include "models/MarketDataStreams.hpp"

class MarketRegimeClassifierBridge : public QObject {
    Q_OBJECT
public:
    struct Thresholds {
        int    minHistory = 30;
        int    trendWindow = 50;
        int    dailyWindow = 20;
        double trendStrengthThreshold = 0.01;
        double momentumThreshold = 0.0015;
        double volatilityThreshold = 0.015;
        double intradayThreshold = 0.02;
        double autocorrThreshold = -0.2;
        double volumeTrendThreshold = 0.15;
    };

    explicit MarketRegimeClassifierBridge(QObject* parent = nullptr);

    void setThresholds(const Thresholds& thresholds);
    bool loadThresholdsFromFile(const QString& path);

    std::optional<MarketRegimeSnapshotEntry> classify(const QVector<OhlcvPoint>& history) const;

private:
    Thresholds m_thresholds;

    double computeReturn(const QVector<OhlcvPoint>& history, int window) const;
    double computeMomentum(const QVector<OhlcvPoint>& history, int window) const;
    double computeVolatility(const QVector<OhlcvPoint>& history, int window) const;
    double computeIntradayRange(const QVector<OhlcvPoint>& history, int window) const;
    double computeAutocorrelation(const QVector<OhlcvPoint>& history, int window) const;
    double computeVolumeTrend(const QVector<OhlcvPoint>& history, int window) const;
};

