#pragma once

#include <QObject>
#include <QProcess>
#include <QUrl>
#include <QVariantMap>
#include <QVector>
#include <QStringList>

#include <memory>

class SupportBundleController : public QObject {
    Q_OBJECT

    Q_PROPERTY(bool busy READ busy NOTIFY busyChanged)
    Q_PROPERTY(QString pythonExecutable READ pythonExecutable WRITE setPythonExecutable NOTIFY pythonExecutableChanged)
    Q_PROPERTY(QString scriptPath READ scriptPath WRITE setScriptPath NOTIFY scriptPathChanged)
    Q_PROPERTY(QString outputDirectory READ outputDirectory WRITE setOutputDirectory NOTIFY outputDirectoryChanged)
    Q_PROPERTY(QString defaultBasename READ defaultBasename WRITE setDefaultBasename NOTIFY defaultBasenameChanged)
    Q_PROPERTY(QString format READ format WRITE setFormat NOTIFY formatChanged)
    Q_PROPERTY(bool includeLogs READ includeLogs WRITE setIncludeLogs NOTIFY includeLogsChanged)
    Q_PROPERTY(bool includeReports READ includeReports WRITE setIncludeReports NOTIFY includeReportsChanged)
    Q_PROPERTY(bool includeLicenses READ includeLicenses WRITE setIncludeLicenses NOTIFY includeLicensesChanged)
    Q_PROPERTY(bool includeMetrics READ includeMetrics WRITE setIncludeMetrics NOTIFY includeMetricsChanged)
    Q_PROPERTY(bool includeAudit READ includeAudit WRITE setIncludeAudit NOTIFY includeAuditChanged)
    Q_PROPERTY(QString logsPath READ logsPath WRITE setLogsPath NOTIFY logsPathChanged)
    Q_PROPERTY(QString reportsPath READ reportsPath WRITE setReportsPath NOTIFY reportsPathChanged)
    Q_PROPERTY(QString licensesPath READ licensesPath WRITE setLicensesPath NOTIFY licensesPathChanged)
    Q_PROPERTY(QString metricsPath READ metricsPath WRITE setMetricsPath NOTIFY metricsPathChanged)
    Q_PROPERTY(QString auditPath READ auditPath WRITE setAuditPath NOTIFY auditPathChanged)
    Q_PROPERTY(QString lastBundlePath READ lastBundlePath NOTIFY lastBundlePathChanged)
    Q_PROPERTY(QString lastStatusMessage READ lastStatusMessage NOTIFY lastStatusMessageChanged)
    Q_PROPERTY(QString lastErrorMessage READ lastErrorMessage NOTIFY lastErrorMessageChanged)
    Q_PROPERTY(QVariantMap lastResult READ lastResult NOTIFY lastResultChanged)
    Q_PROPERTY(QVariantMap metadata READ metadata NOTIFY metadataChanged)

public:
    explicit SupportBundleController(QObject* parent = nullptr);
    ~SupportBundleController() override;

    bool busy() const { return m_busy; }

    QString pythonExecutable() const { return m_pythonExecutable; }
    void setPythonExecutable(const QString& executable);

    QString scriptPath() const { return m_scriptPath; }
    void setScriptPath(const QString& path);

    QString outputDirectory() const { return m_outputDirectory; }
    void setOutputDirectory(const QString& directory);

    QString defaultBasename() const { return m_defaultBasename; }
    void setDefaultBasename(const QString& basename);

    QString format() const { return m_format; }
    void setFormat(const QString& format);

    bool includeLogs() const { return m_includeLogs; }
    void setIncludeLogs(bool include);

    bool includeReports() const { return m_includeReports; }
    void setIncludeReports(bool include);

    bool includeLicenses() const { return m_includeLicenses; }
    void setIncludeLicenses(bool include);

    bool includeMetrics() const { return m_includeMetrics; }
    void setIncludeMetrics(bool include);

    bool includeAudit() const { return m_includeAudit; }
    void setIncludeAudit(bool include);

    QString logsPath() const { return m_logsPath; }
    void setLogsPath(const QString& path);

    QString reportsPath() const { return m_reportsPath; }
    void setReportsPath(const QString& path);

    QString licensesPath() const { return m_licensesPath; }
    void setLicensesPath(const QString& path);

    QString metricsPath() const { return m_metricsPath; }
    void setMetricsPath(const QString& path);

    QString auditPath() const { return m_auditPath; }
    void setAuditPath(const QString& path);

    QString lastBundlePath() const { return m_lastBundlePath; }
    QString lastStatusMessage() const { return m_lastStatusMessage; }
    QString lastErrorMessage() const { return m_lastErrorMessage; }
    QVariantMap lastResult() const { return m_lastResult; }
    QVariantMap metadata() const { return m_metadata; }

    Q_INVOKABLE bool exportBundle(const QUrl& destination = QUrl(), bool dryRun = false);
    Q_INVOKABLE QStringList buildCommandArguments(const QUrl& destination = QUrl(), bool dryRun = false) const;
    void setMetadata(const QVariantMap& metadata);
    void setExtraIncludeSpecs(const QStringList& specs);

signals:
    void busyChanged();
    void pythonExecutableChanged();
    void scriptPathChanged();
    void outputDirectoryChanged();
    void defaultBasenameChanged();
    void formatChanged();
    void includeLogsChanged();
    void includeReportsChanged();
    void includeLicensesChanged();
    void includeMetricsChanged();
    void includeAuditChanged();
    void logsPathChanged();
    void reportsPathChanged();
    void licensesPathChanged();
    void metricsPathChanged();
    void auditPathChanged();
    void lastBundlePathChanged();
    void lastStatusMessageChanged();
    void lastErrorMessageChanged();
    void lastResultChanged();
    void metadataChanged();
    void exportFinished(bool success, const QVariantMap& payload);

private:
    QString expandPath(const QString& path) const;
    QString sanitizeFormat(const QString& requested) const;
    QString buildDefaultDestination() const;
    QStringList buildIncludeArguments() const;
    QStringList buildDisableArguments() const;
    QStringList buildMetadataArguments() const;
    QStringList buildExtraIncludeArguments() const;
    void handleProcessFinished(int exitCode, QProcess::ExitStatus status);
    void resetActiveProcess();

    bool                 m_busy = false;
    QString              m_pythonExecutable = QStringLiteral("python3");
    QString              m_scriptPath;
    QString              m_outputDirectory;
    QString              m_defaultBasename = QStringLiteral("support-bundle");
    QString              m_format = QStringLiteral("tar.gz");
    bool                 m_includeLogs = true;
    bool                 m_includeReports = true;
    bool                 m_includeLicenses = true;
    bool                 m_includeMetrics = true;
    bool                 m_includeAudit = false;
    QString              m_logsPath;
    QString              m_reportsPath;
    QString              m_licensesPath;
    QString              m_metricsPath;
    QString              m_auditPath;
    QString              m_lastBundlePath;
    QString              m_lastStatusMessage;
    QString              m_lastErrorMessage;
    QVariantMap          m_lastResult;
    QVariantMap          m_metadata;
    QStringList          m_extraIncludeSpecs;
    std::unique_ptr<QProcess> m_process;
};
