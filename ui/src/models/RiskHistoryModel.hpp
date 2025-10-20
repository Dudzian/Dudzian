#pragma once

#include <QAbstractListModel>
#include <QDateTime>
#include <QJsonArray>
#include <QList>

#include "RiskTypes.hpp"

class RiskHistoryModel : public QAbstractListModel {
    Q_OBJECT
    Q_PROPERTY(bool hasSamples READ hasSamples NOTIFY historyChanged)
    Q_PROPERTY(int maximumEntries READ maximumEntries WRITE setMaximumEntries NOTIFY maximumEntriesChanged)
    Q_PROPERTY(int entryCount READ entryCount NOTIFY historyChanged)
    Q_PROPERTY(double maxDrawdown READ maxDrawdown NOTIFY summaryChanged)
    Q_PROPERTY(double minDrawdown READ minDrawdown NOTIFY summaryChanged)
    Q_PROPERTY(double averageDrawdown READ averageDrawdown NOTIFY summaryChanged)
    Q_PROPERTY(double maxLeverage READ maxLeverage NOTIFY summaryChanged)
    Q_PROPERTY(double averageLeverage READ averageLeverage NOTIFY summaryChanged)
    Q_PROPERTY(double minPortfolioValue READ minPortfolioValue NOTIFY summaryChanged)
    Q_PROPERTY(double maxPortfolioValue READ maxPortfolioValue NOTIFY summaryChanged)
    Q_PROPERTY(bool anyExposureBreached READ anyExposureBreached NOTIFY summaryChanged)
    Q_PROPERTY(int totalBreachCount READ totalBreachCount NOTIFY summaryChanged)
    Q_PROPERTY(double maxExposureUtilization READ maxExposureUtilization NOTIFY summaryChanged)

public:
    enum Roles {
        TimestampRole = Qt::UserRole + 1,
        DrawdownRole,
        LeverageRole,
        PortfolioValueRole,
        ProfileLabelRole,
        HasBreachRole,
        BreachCountRole,
        MaxExposureUtilizationRole,
        ExposuresRole,
    };

    explicit RiskHistoryModel(QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QHash<int, QByteArray> roleNames() const override;

    bool hasSamples() const { return !m_entries.isEmpty(); }

    int maximumEntries() const { return m_maxEntries; }
    void setMaximumEntries(int value);

    int entryCount() const { return m_entries.size(); }

    double maxDrawdown() const { return m_summary.maxDrawdown; }
    double minDrawdown() const { return m_summary.minDrawdown; }
    double averageDrawdown() const { return m_summary.averageDrawdown; }
    double maxLeverage() const { return m_summary.maxLeverage; }
    double averageLeverage() const { return m_summary.averageLeverage; }
    double minPortfolioValue() const { return m_summary.minPortfolioValue; }
    double maxPortfolioValue() const { return m_summary.maxPortfolioValue; }
    bool anyExposureBreached() const { return m_summary.anyBreach; }
    int totalBreachCount() const { return m_summary.totalBreaches; }
    double maxExposureUtilization() const { return m_summary.maxExposureUtilization; }

    QJsonArray toJson(int limit = -1) const;
    void restoreFromJson(const QJsonArray& array);
    bool exportToCsv(const QString& filePath, int limit = -1) const;

public slots:
    void recordSnapshot(const RiskSnapshotData& snapshot);
    void clear();

signals:
    void historyChanged();
    void maximumEntriesChanged();
    void summaryChanged();
    void snapshotRecorded(const QDateTime& timestamp);

private:
    struct Entry {
        QDateTime timestamp;
        QString profileLabel;
        double drawdown = 0.0;
        double leverage = 0.0;
        double portfolioValue = 0.0;
        QList<RiskExposureData> exposures;
        int breachCount = 0;
        bool hasBreach = false;
        double maxExposureUtilization = 0.0;
    };

    struct Summary {
        double maxDrawdown = 0.0;
        double minDrawdown = 0.0;
        double averageDrawdown = 0.0;
        double maxLeverage = 0.0;
        double averageLeverage = 0.0;
        double minPortfolioValue = 0.0;
        double maxPortfolioValue = 0.0;
        bool anyBreach = false;
        int totalBreaches = 0;
        double maxExposureUtilization = 0.0;
    };

    bool trimExcess();
    void recalculateSummary();
    void updateDerivedFields(Entry& entry) const;
    QVariantList exposuresToVariantList(const Entry& entry) const;

    QList<Entry> m_entries;
    int m_maxEntries = 50;
    Summary m_summary;
};

