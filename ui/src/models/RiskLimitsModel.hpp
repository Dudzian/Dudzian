#pragma once

#include <QAbstractListModel>
#include <QVariantMap>

#include "RiskTypes.hpp"

class RiskLimitsModel : public QAbstractListModel {
    Q_OBJECT
    Q_PROPERTY(QVariantMap limits READ limits NOTIFY limitsChanged)

public:
    enum Roles {
        KeyRole = Qt::UserRole + 1,
        LabelRole,
        ValueRole,
        MinimumRole,
        MaximumRole,
        StepRole,
        PercentRole,
        EditableRole,
    };

    explicit RiskLimitsModel(QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QHash<int, QByteArray> roleNames() const override;

    QVariantMap limits() const { return m_limits; }

public slots:
    void updateFromSnapshot(const RiskSnapshotData& snapshot);
    bool setLimitValue(const QString& key, double value);
    bool setLimitValueAt(int index, double value);
    void clear();

signals:
    void limitsChanged();
    void limitEdited(const QString& key, double value);

private:
    struct Entry {
        QString key;
        QString label;
        double value = 0.0;
        double minValue = 0.0;
        double maxValue = 0.0;
        double step = 0.1;
        bool isPercent = false;
        bool editable = true;
    };

    struct Definition {
        QString key;
        QString label;
        double defaultValue = 0.0;
        double minValue = 0.0;
        double maxValue = 0.0;
        double step = 0.1;
        bool isPercent = false;
        bool editable = true;
    };

    QVector<Entry> m_entries;
    QVariantMap m_limits;

    static const QVector<Definition>& knownDefinitions();
    int findEntryIndex(const QString& key) const;
    void rebuildFromMap(const QVariantMap& limits);
};

