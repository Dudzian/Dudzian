#pragma once

#include <QAbstractListModel>
#include <QVariantMap>

#include "RiskTypes.hpp"

class RiskStateModel : public QAbstractListModel {
    Q_OBJECT
    Q_PROPERTY(bool hasData READ hasData NOTIFY riskStateChanged)
    Q_PROPERTY(QString profileLabel READ profileLabel NOTIFY riskStateChanged)
    Q_PROPERTY(double portfolioValue READ portfolioValue NOTIFY riskStateChanged)
    Q_PROPERTY(double currentDrawdown READ currentDrawdown NOTIFY riskStateChanged)
    Q_PROPERTY(double maxDailyLoss READ maxDailyLoss NOTIFY riskStateChanged)
    Q_PROPERTY(double usedLeverage READ usedLeverage NOTIFY riskStateChanged)
    Q_PROPERTY(QDateTime generatedAt READ generatedAt NOTIFY riskStateChanged)

public:
    enum Roles {
        CodeRole = Qt::UserRole + 1,
        CurrentValueRole,
        MaxValueRole,
        ThresholdValueRole,
        BreachRole
    };

    explicit RiskStateModel(QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QHash<int, QByteArray> roleNames() const override;

    bool hasData() const { return m_snapshot.hasData; }
    QString profileLabel() const { return m_snapshot.profileLabel; }
    double portfolioValue() const { return m_snapshot.portfolioValue; }
    double currentDrawdown() const { return m_snapshot.currentDrawdown; }
    double maxDailyLoss() const { return m_snapshot.maxDailyLoss; }
    double usedLeverage() const { return m_snapshot.usedLeverage; }
    QDateTime generatedAt() const { return m_snapshot.generatedAt; }
    QVariantMap currentSnapshot() const;

public slots:
    void updateFromSnapshot(const RiskSnapshotData& snapshot);
    void clear();

signals:
    void riskStateChanged();

private:
    RiskSnapshotData m_snapshot;
};
