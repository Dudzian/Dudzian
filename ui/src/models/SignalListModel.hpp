#pragma once

#include <QAbstractListModel>

#include "models/MarketDataStreams.hpp"

class SignalListModel : public QAbstractListModel {
    Q_OBJECT
    Q_PROPERTY(int count READ rowCount NOTIFY countChanged)

public:
    enum Roles {
        TimestampRole = Qt::UserRole + 1,
        TimestampDisplayRole,
        CodeRole,
        DescriptionRole,
        ConfidenceRole,
        RegimeRole,
    };

    explicit SignalListModel(QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QHash<int, QByteArray> roleNames() const override;

    void resetWithSignals(const QVector<SignalEventEntry>& events);
    void appendSignal(const SignalEventEntry& event);

signals:
    void countChanged();

private:
    QVector<SignalEventEntry> m_events;
};

