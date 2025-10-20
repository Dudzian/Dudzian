#pragma once

#include <QObject>
#include <QProcess>
#include <QString>
#include <QStringList>

class BotCoreLocalService : public QObject {
    Q_OBJECT

public:
    explicit BotCoreLocalService(QObject* parent = nullptr);
    ~BotCoreLocalService() override;

    void setPythonExecutable(const QString& executable);
    void setModuleName(const QString& moduleName);
    void setDatasetPath(const QString& datasetPath);
    void setHost(const QString& host);
    void setPort(int port);
    void setStreamRepeat(bool enabled);
    void setStreamInterval(double seconds);
    void setRepoRoot(const QString& path);

    [[nodiscard]] bool start(int timeoutMs = 8000);
    void stop(int timeoutMs = 2000);

    [[nodiscard]] QString endpoint() const { return m_endpoint; }
    [[nodiscard]] QString lastError() const { return m_lastError; }
    [[nodiscard]] bool isRunning() const { return m_process.state() != QProcess::NotRunning; }

private:
    void resetState();
    bool waitForReady(int timeoutMs);
    void handleReadyRead();
    void handleReadyReadError();
    bool parseStdoutBuffer();
    void handleProcessFinished(int exitCode, QProcess::ExitStatus status);
    QString locateRepoRootFromCwd() const;
    void appendStderr(const QByteArray& chunk);
    QString lastStderrLines(int maxLines = 3) const;

    QProcess m_process;
    QString m_pythonExecutable = QStringLiteral("python3");
    QString m_moduleName = QStringLiteral("bot_core.runtime.local_service");
    QString m_datasetPath;
    QString m_host = QStringLiteral("127.0.0.1");
    int m_port = 0;
    bool m_streamRepeat = false;
    double m_streamInterval = 0.0;
    QString m_repoRoot;
    QString m_endpoint;
    QString m_lastError;
    QByteArray m_stdoutBuffer;
    QByteArray m_stderrBuffer;
};
