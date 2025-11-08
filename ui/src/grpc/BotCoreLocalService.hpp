#pragma once

#include <QObject>
#include <QHash>
#include <QJsonObject>
#include <QProcess>
#include <QString>
#include <QStringList>
#include <QTimer>
#include <QVariantList>
#include <QVariantMap>

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

    void startOhlcvStream(const QString& symbol, int maxUpdates = 1, int timeoutMs = 250);
    void stopOhlcvStream();

    [[nodiscard]] QVariantMap fetchAutoModeSnapshot(int timeoutMs = 800);
    [[nodiscard]] QVariantMap toggleAutoMode(bool enabled, int timeoutMs = 800);
    [[nodiscard]] QVariantMap updateAutoModeAlerts(const QVariantMap& preferences, int timeoutMs = 800);
    [[nodiscard]] QVariantMap saveStrategyPreset(const QVariantMap& preset, int timeoutMs = 800);
    [[nodiscard]] QVariantList listStrategyPresets(int timeoutMs = 800);
    [[nodiscard]] QVariantMap loadStrategyPreset(const QVariantMap& selector, int timeoutMs = 800);
    [[nodiscard]] QVariantMap deleteStrategyPreset(const QVariantMap& selector, int timeoutMs = 800);

signals:
    void ohlcvSnapshotReady(const QVariantList& snapshot, const QString& subscriptionId);
    void ohlcvUpdatesReady(const QVariantList& updates, const QString& subscriptionId);
    void ohlcvStreamClosed(const QString& subscriptionId);
    void ohlcvStreamError(const QString& message);
    void rpcRequestSucceeded(quint64 id, const QString& method, const QJsonObject& result);
    void rpcRequestFailed(quint64 id, const QString& method, const QString& errorMessage);

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
    quint64 nextRequestId();
    bool sendJsonRequest(const QJsonObject& object, quint64 id, const QString& method);
    void handleRpcMessage(const QJsonObject& object);
    void handleStreamResponse(const QJsonObject& result);
    void scheduleNextPoll(int delayMs = 0);
    void sendStreamPoll();
    QJsonObject invokeRpc(const QString& method, const QJsonObject& params, int timeoutMs, bool* ok = nullptr);

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
    quint64 m_nextRequestId = 1;
    QHash<quint64, QString> m_pendingMethods;
    QTimer m_streamPollTimer;
    QString m_streamSubscriptionId;
    QString m_streamSymbol;
    int m_streamMaxUpdates = 1;
    int m_streamTimeoutMs = 250;
    bool m_streamActive = false;
};
