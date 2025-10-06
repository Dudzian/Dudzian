#pragma once

#include <QAbstractListModel>
#include <QDateTime>
#include <QVector>
#include <QString>
#include <QVariant>

struct OhlcvPoint {
    qint64 timestampMs = 0;
    double open = 0.0;
    double high = 0.0;
    double low = 0.0;
    double close = 0.0;
    double volume = 0.0;
    bool closed = false;
    quint64 sequence = 0;
};
Q_DECLARE_METATYPE(OhlcvPoint)

class OhlcvListModel : public QAbstractListModel {
    Q_OBJECT
    Q_PROPERTY(int maximumSamples READ maximumSamples WRITE setMaximumSamples NOTIFY maximumSamplesChanged)

public:
    enum Roles {
        TimestampRole = Qt::UserRole + 1,
        OpenRole,
        HighRole,
        LowRole,
        CloseRole,
        VolumeRole,
        ClosedRole,
        SequenceRole
    };

    explicit OhlcvListModel(QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QHash<int, QByteArray> roleNames() const override;

    void resetWithHistory(const QList<OhlcvPoint>& candles);
    void applyIncrement(const OhlcvPoint& candle);

    int maximumSamples() const { return m_maximumSamples; }
    void setMaximumSamples(int value);

    Q_INVOKABLE QVariant latestClose() const;
    Q_INVOKABLE QVariant timestampAt(int row) const;
    Q_INVOKABLE QVariantMap candleAt(int row) const;
    Q_INVOKABLE QVariantList overlaySeries(const QString& id) const;

signals:
    void maximumSamplesChanged();

private:
    void enforceMaximum();
    void recomputeIndicators();

    QVector<OhlcvPoint> m_candles;
    QVector<double> m_emaFast;
    QVector<double> m_emaSlow;
    QVector<double> m_vwap;
    int m_maximumSamples = 10240;
};
