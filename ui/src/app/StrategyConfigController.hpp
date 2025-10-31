#pragma once

#include <QObject>
#include <QVariant>
#include <QProcess>
#include <QHash>
#include <QStringList>

#include <memory>

class StrategyConfigController : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool busy READ busy NOTIFY busyChanged)
    Q_PROPERTY(QString lastError READ lastError NOTIFY lastErrorChanged)
    Q_PROPERTY(QStringList validationIssues READ validationIssues NOTIFY validationIssuesChanged)

public:
    explicit StrategyConfigController(QObject* parent = nullptr);
    ~StrategyConfigController() override;

    void setConfigPath(const QString& path);
    void setPythonExecutable(const QString& executable);
    void setScriptPath(const QString& path);

    bool busy() const { return m_busy; }
    QString lastError() const { return m_lastError; }
    QStringList validationIssues() const { return m_validationIssues; }

    Q_INVOKABLE bool refresh();
    Q_INVOKABLE QVariantMap decisionConfigSnapshot() const;
    Q_INVOKABLE QVariantList schedulerList() const;
    Q_INVOKABLE QVariantMap schedulerConfigSnapshot(const QString& name) const;
    Q_INVOKABLE bool saveDecisionConfig(const QVariantMap& config);
    Q_INVOKABLE bool saveSchedulerConfig(const QString& name, const QVariantMap& config);
    Q_INVOKABLE bool removeSchedulerConfig(const QString& name);
    Q_INVOKABLE bool runSchedulerNow(const QString& name);

signals:
    void busyChanged();
    void lastErrorChanged();
    void validationIssuesChanged();
    void decisionConfigChanged();
    void schedulerListChanged();

private:
    struct BridgeResult {
        bool ok = false;
        QByteArray stdoutData;
        QString errorMessage;
    };

    BridgeResult invokeBridge(const QStringList& args, const QByteArray& stdinData = QByteArray()) const;
    bool handleBridgeValidation(const BridgeResult& result);
    bool parseDump(const QByteArray& payload);
    bool ensureReady() const;

    QString m_configPath;
    QString m_pythonExecutable = QStringLiteral("python3");
    QString m_scriptPath;

    QVariantMap m_decisionConfig;
    QHash<QString, QVariantMap> m_schedulerConfigs;
    QVariantList m_schedulerList;

    bool m_busy = false;
    QString m_lastError;
    QStringList m_validationIssues;
};
