#pragma once

#include <QObject>
#include <QProcess>
#include <QString>
#include <QVariantMap>

class HypercareController final : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString pythonExecutable READ pythonExecutable WRITE setPythonExecutable NOTIFY pythonExecutableChanged)
    Q_PROPERTY(QString cycleScript READ cycleScript WRITE setCycleScript NOTIFY cycleScriptChanged)
    Q_PROPERTY(QString bundleSource READ bundleSource WRITE setBundleSource NOTIFY bundleSourceChanged)
    Q_PROPERTY(QString bundleOutputDir READ bundleOutputDir WRITE setBundleOutputDir NOTIFY bundleOutputDirChanged)
    Q_PROPERTY(QString failoverPlanPath READ failoverPlanPath WRITE setFailoverPlanPath NOTIFY failoverPlanPathChanged)
    Q_PROPERTY(QString selfHealRulesPath READ selfHealRulesPath WRITE setSelfHealRulesPath NOTIFY selfHealRulesPathChanged)
    Q_PROPERTY(QString selfHealOutputPath READ selfHealOutputPath WRITE setSelfHealOutputPath NOTIFY selfHealOutputPathChanged)
    Q_PROPERTY(QString signingKeyPath READ signingKeyPath WRITE setSigningKeyPath NOTIFY signingKeyPathChanged)
    Q_PROPERTY(QString signingKeyId READ signingKeyId WRITE setSigningKeyId NOTIFY signingKeyIdChanged)
    Q_PROPERTY(QString stage6SummaryPath READ stage6SummaryPath WRITE setStage6SummaryPath NOTIFY stage6SummaryPathChanged)
    Q_PROPERTY(bool running READ isRunning NOTIFY runningChanged)
    Q_PROPERTY(QString lastMessage READ lastMessage NOTIFY lastMessageChanged)
    Q_PROPERTY(QString lastError READ lastError NOTIFY lastErrorChanged)

public:
    explicit HypercareController(QObject* parent = nullptr);
    ~HypercareController() override;

    QString pythonExecutable() const { return m_pythonExecutable; }
    void setPythonExecutable(const QString& value);

    QString cycleScript() const { return m_cycleScript; }
    void setCycleScript(const QString& value);

    QString bundleSource() const { return m_bundleSource; }
    void setBundleSource(const QString& value);

    QString bundleOutputDir() const { return m_bundleOutputDir; }
    void setBundleOutputDir(const QString& value);

    QString failoverPlanPath() const { return m_failoverPlanPath; }
    void setFailoverPlanPath(const QString& value);

    QString selfHealRulesPath() const { return m_selfHealRulesPath; }
    void setSelfHealRulesPath(const QString& value);

    QString selfHealOutputPath() const { return m_selfHealOutputPath; }
    void setSelfHealOutputPath(const QString& value);

    QString signingKeyPath() const { return m_signingKeyPath; }
    void setSigningKeyPath(const QString& value);

    QString signingKeyId() const { return m_signingKeyId; }
    void setSigningKeyId(const QString& value);

    QString stage6SummaryPath() const { return m_stage6SummaryPath; }
    void setStage6SummaryPath(const QString& value);

    bool isRunning() const { return m_running; }
    QString lastMessage() const { return m_lastMessage; }
    QString lastError() const { return m_lastError; }

    Q_INVOKABLE QVariantMap loadSummary(const QString& path) const;
    Q_INVOKABLE bool triggerSelfHealing();

signals:
    void pythonExecutableChanged();
    void cycleScriptChanged();
    void bundleSourceChanged();
    void bundleOutputDirChanged();
    void failoverPlanPathChanged();
    void selfHealRulesPathChanged();
    void selfHealOutputPathChanged();
    void signingKeyPathChanged();
    void signingKeyIdChanged();
    void stage6SummaryPathChanged();
    void runningChanged();
    void lastMessageChanged();
    void lastErrorChanged();
    void selfHealingCompleted(int exitCode);

private slots:
    void handleProcessFinished(int exitCode, QProcess::ExitStatus status);
    void handleProcessError(QProcess::ProcessError error);
    void handleReadyRead();

private:
    void resetProcess();
    QString normalizePath(const QString& path) const;

    QString m_pythonExecutable = QStringLiteral("python3");
    QString m_cycleScript = QStringLiteral("scripts/run_stage6_resilience_cycle.py");
    QString m_bundleSource = QStringLiteral("var/audit/resilience");
    QString m_bundleOutputDir = QStringLiteral("var/resilience");
    QString m_failoverPlanPath = QStringLiteral("data/stage6/resilience/failover_plan.json");
    QString m_selfHealRulesPath = QStringLiteral("config/stage6/resilience_self_heal.json");
    QString m_selfHealOutputPath = QStringLiteral("var/audit/resilience/self_healing_report.json");
    QString m_signingKeyPath;
    QString m_signingKeyId;
    QString m_stage6SummaryPath = QStringLiteral("var/audit/stage6/stage6_hypercare_summary.json");

    bool m_running = false;
    QString m_lastMessage;
    QString m_lastError;
    QString m_outputBuffer;
    QString m_errorBuffer;
    QProcess* m_process = nullptr;
};
