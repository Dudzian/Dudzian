#pragma once

#include <QAbstractListModel>
#include <QVariantMap>

#include "RiskTypes.hpp"

class RiskCostModel : public QAbstractListModel {
    Q_OBJECT
    Q_PROPERTY(QVariantMap summary READ summary NOTIFY summaryChanged)

public:
    enum Roles {
        KeyRole = Qt::UserRole + 1,
        LabelRole,
        ValueRole,
        FormattedRole,
    };

    explicit RiskCostModel(QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QHash<int, QByteArray> roleNames() const override;

    QVariantMap summary() const { return m_summary; }

public slots:
    void updateFromSnapshot(const RiskSnapshotData& snapshot);
    void clear();

signals:
    void summaryChanged();

private:
    struct Entry {
        QString key;
        QString label;
        QVariant value;
        QString formatted;
    };

    struct MetricDefinition {
        QString key;
        QString label;
        bool percent = false;
        int decimals = 2;
    };

    QVector<Entry> m_entries;
    QVariantMap m_summary;

    static const QVector<MetricDefinition>& metricDefinitions();
    static QString formatValue(const MetricDefinition& def, const QVariant& value);
};

