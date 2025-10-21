#pragma once

#include <QAbstractListModel>
#include <QDateTime>
#include <QFileSystemWatcher>
#include <QTimer>
#include <QVariantMap>

class DecisionLogModel : public QAbstractListModel {
    Q_OBJECT
    Q_PROPERTY(QString logPath READ logPath NOTIFY logPathChanged)
    Q_PROPERTY(int maximumEntries READ maximumEntries WRITE setMaximumEntries NOTIFY maximumEntriesChanged)
    Q_PROPERTY(int count READ rowCount NOTIFY countChanged)

public:
    enum Roles {
        TimestampRole = Qt::UserRole + 1,
        TimestampDisplayRole,
        EventRole,
        EnvironmentRole,
        PortfolioRole,
        RiskProfileRole,
        ScheduleRole,
        StrategyRole,
        SymbolRole,
        SideRole,
        QuantityRole,
        PriceRole,
        ApprovedRole,
        DecisionStateRole,
        DecisionReasonRole,
        DecisionModeRole,
        TelemetryNamespaceRole,
        DetailsRole,
    };

    explicit DecisionLogModel(QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QHash<int, QByteArray> roleNames() const override;

    QString logPath() const { return m_logPath; }

    int maximumEntries() const { return m_maxEntries; }
    void setMaximumEntries(int value);

    Q_INVOKABLE bool reload();

    void setLogPath(const QString& path);

signals:
    void logPathChanged();
    void maximumEntriesChanged();
    void countChanged();

private slots:
    void handleWatchedPathChanged(const QString& path);
    void performReload();

private:
    struct Entry {
        QDateTime timestampUtc;
        QString event;
        QString environment;
        QString portfolio;
        QString riskProfile;
        QString schedule;
        QString strategy;
        QString symbol;
        QString side;
        QString quantity;
        QString price;
        bool approved = false;
        QString decisionState;
        QString decisionReason;
        QString decisionMode;
        QString telemetryNamespace;
        QVariantMap payload;
    };

    void scheduleReload();
    bool readEntries(QVector<Entry>& entries, QStringList& watchedFiles) const;
    Entry buildEntry(const QVariantMap& payload) const;
    QDateTime parseTimestamp(const QString& text) const;
    bool coerceBool(const QVariant& value) const;
    void updateWatchers(const QStringList& files);

    QVector<Entry> m_entries;
    QString m_logPath;
    int m_maxEntries = 250;
    QFileSystemWatcher m_watcher;
    QTimer m_reloadDebounce;
};

