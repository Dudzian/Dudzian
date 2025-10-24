#pragma once

#include <QAbstractListModel>
#include <QHash>

#include "models/MarketDataStreams.hpp"

class IndicatorSeriesModel : public QAbstractListModel {
    Q_OBJECT
    Q_PROPERTY(int count READ rowCount NOTIFY countChanged)

public:
    enum Roles {
        IdRole = Qt::UserRole + 1,
        LabelRole,
        ColorRole,
        SecondaryRole,
        SamplesRole,
        LatestValueRole,
    };

    explicit IndicatorSeriesModel(QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QHash<int, QByteArray> roleNames() const override;

    void setSeriesDefinitions(const QVector<IndicatorSeriesDefinition>& definitions);
    void replaceSamples(const QString& id, const QVector<IndicatorSample>& samples);
    void appendSample(const IndicatorSample& sample);

signals:
    void countChanged();

private:
    struct SeriesData {
        IndicatorSeriesDefinition definition;
        QVector<IndicatorSample> samples;
    };

    QVector<SeriesData> m_series;

    int indexForId(const QString& id) const;
};

