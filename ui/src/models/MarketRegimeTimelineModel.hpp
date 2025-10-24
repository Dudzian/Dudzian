#pragma once

#include <QAbstractListModel>

#include "models/MarketDataStreams.hpp"

class MarketRegimeTimelineModel : public QAbstractListModel {
    Q_OBJECT
    Q_PROPERTY(int count READ rowCount NOTIFY countChanged)
    Q_PROPERTY(QString latestRegime READ latestRegime NOTIFY latestRegimeChanged)
    Q_PROPERTY(int maximumSnapshots READ maximumSnapshots WRITE setMaximumSnapshots NOTIFY maximumSnapshotsChanged)

public:
    enum Roles {
        TimestampRole = Qt::UserRole + 1,
        TimestampDisplayRole,
        RegimeRole,
        TrendConfidenceRole,
        MeanReversionConfidenceRole,
        DailyConfidenceRole,
    };

    explicit MarketRegimeTimelineModel(QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QHash<int, QByteArray> roleNames() const override;

    QString latestRegime() const;

    int maximumSnapshots() const { return m_maxSnapshots; }
    void setMaximumSnapshots(int value);

    void resetWithSnapshots(const QVector<MarketRegimeSnapshotEntry>& snapshots);
    void appendSnapshot(const MarketRegimeSnapshotEntry& snapshot);

signals:
    void countChanged();
    void latestRegimeChanged();
    void maximumSnapshotsChanged();

private:
    bool trimExcessSnapshots();

    QVector<MarketRegimeSnapshotEntry> m_snapshots;
    int m_maxSnapshots = 720;
};

