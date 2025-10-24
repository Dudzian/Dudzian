#pragma once

#include <QColor>
#include <QMetaType>
#include <QString>
#include <QVector>

struct IndicatorSample {
    QString seriesId;
    qint64  timestampMs = 0;
    double  value = 0.0;
};
Q_DECLARE_METATYPE(IndicatorSample)

struct SignalEventEntry {
    qint64  timestampMs = 0;
    QString code;
    QString description;
    double  confidence = 0.0;
    QString regime;
};
Q_DECLARE_METATYPE(SignalEventEntry)

struct MarketRegimeSnapshotEntry {
    qint64  timestampMs = 0;
    QString regime;
    double  trendConfidence = 0.0;
    double  meanReversionConfidence = 0.0;
    double  dailyConfidence = 0.0;
};
Q_DECLARE_METATYPE(MarketRegimeSnapshotEntry)

struct IndicatorSeriesDefinition {
    QString id;
    QString label;
    QColor  color;
    bool    secondary = false;
};

