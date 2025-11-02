#pragma once

#include <QDateTime>
#include <QList>
#include <QMetaType>
#include <QString>
#include <QVariantMap>

struct RiskExposureData {
    QString code;
    double maxValue = 0.0;
    double currentValue = 0.0;
    double thresholdValue = 0.0;

    bool isBreached() const { return thresholdValue > 0.0 && currentValue >= thresholdValue; }
};

struct RiskSnapshotData {
    QString profileLabel;
    int profileEnum = 0;
    double portfolioValue = 0.0;
    double currentDrawdown = 0.0;
    double maxDailyLoss = 0.0;
    double usedLeverage = 0.0;
    QDateTime generatedAt;
    QList<RiskExposureData> exposures;
    QVariantMap limits;
    QVariantMap statistics;
    QVariantMap costBreakdown;
    bool killSwitchEngaged = false;
    bool hasData = false;
};

Q_DECLARE_METATYPE(RiskExposureData)
Q_DECLARE_METATYPE(RiskSnapshotData)
