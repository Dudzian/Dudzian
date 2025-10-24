#pragma once

#include <QObject>
#include <QByteArray>
#include <QFileSystemWatcher>
#include <QString>
#include <QStringList>
#include <QTimer>
#include <QVariant>

class StrategyWorkbenchController : public QObject
{
    Q_OBJECT
    Q_PROPERTY(bool busy READ busy NOTIFY busyChanged)
    Q_PROPERTY(bool ready READ ready NOTIFY readyChanged)
    Q_PROPERTY(QString lastError READ lastError NOTIFY lastErrorChanged)
    Q_PROPERTY(QVariantList catalogEngines READ catalogEngines NOTIFY catalogChanged)
    Q_PROPERTY(QVariantList catalogDefinitions READ catalogDefinitions NOTIFY catalogChanged)
    Q_PROPERTY(QVariantMap catalogMetadata READ catalogMetadata NOTIFY catalogChanged)
    Q_PROPERTY(QVariantMap catalogBlockedDefinitions READ catalogBlockedDefinitions NOTIFY catalogChanged)
    Q_PROPERTY(QVariantMap catalogRegimeTemplates READ catalogRegimeTemplates NOTIFY catalogChanged)
    Q_PROPERTY(QVariantMap lastValidation READ lastValidation NOTIFY lastValidationChanged)
    Q_PROPERTY(QVariantList presetHistory READ presetHistory NOTIFY presetHistoryChanged)

public:
    explicit StrategyWorkbenchController(QObject* parent = nullptr);
    ~StrategyWorkbenchController() override;

    void setConfigPath(const QString& path);
    void setPythonExecutable(const QString& executable);
    void setScriptPath(const QString& path);

    [[nodiscard]] bool busy() const { return m_busy; }
    [[nodiscard]] QString lastError() const { return m_lastError; }
    [[nodiscard]] bool ready() const { return m_ready; }

    [[nodiscard]] QVariantList catalogEngines() const { return m_catalogEngines; }
    [[nodiscard]] QVariantList catalogDefinitions() const { return m_catalogDefinitions; }
    [[nodiscard]] QVariantMap catalogMetadata() const { return m_catalogMetadata; }
    [[nodiscard]] QVariantMap catalogBlockedDefinitions() const { return m_catalogBlockedDefinitions; }
    [[nodiscard]] QVariantMap catalogRegimeTemplates() const { return m_catalogRegimeTemplates; }
    [[nodiscard]] QVariantMap lastValidation() const { return m_lastValidation; }
    [[nodiscard]] QVariantList presetHistory() const { return m_presetHistory; }

    Q_INVOKABLE bool refreshCatalog();
    Q_INVOKABLE QVariantMap validatePreset(const QVariantMap& presetPayload);
    Q_INVOKABLE void savePresetSnapshot(const QVariantMap& snapshot);
    Q_INVOKABLE QVariantMap restorePreviousPreset();
    Q_INVOKABLE void clearLastError();

Q_SIGNALS:
    void busyChanged();
    void lastErrorChanged();
    void readyChanged();
    void catalogChanged();
    void lastValidationChanged();
    void presetHistoryChanged();

private:
    struct BridgeResult {
        bool ok = false;
        QString errorMessage;
        QByteArray stdoutData;
    };

    [[nodiscard]] bool ensureReady(QString* errorMessage = nullptr) const;
    BridgeResult invokeBridge(const QStringList& args, const QByteArray& stdinData = QByteArray()) const;
    bool parseCatalogPayload(const QByteArray& payload);
    void setLastError(const QString& message);
    void setReady(bool ready);
    void updateWatcherPaths();
    void clearWatcherPaths();
    void handleWatchedPathChanged(const QString& path);
    void scheduleAutoRefresh();
    void handleDebouncedReload();
    void maybeTriggerInitialRefresh();
    void invalidateWorkbenchData();

    QString m_configPath;
    QString m_pythonExecutable = QStringLiteral("python3");
    QString m_scriptPath;

    bool m_busy = false;
    QString m_lastError;
    bool m_catalogInitialized = false;
    bool m_ready = false;

    QVariantList m_catalogEngines;
    QVariantList m_catalogDefinitions;
    QVariantMap m_catalogMetadata;
    QVariantMap m_catalogBlockedDefinitions;
    QVariantMap m_catalogRegimeTemplates;
    QVariantMap m_lastValidation;
    QVariantList m_presetHistory;

    QFileSystemWatcher m_configWatcher;
    QTimer m_reloadDebounce;
    bool m_pendingRefresh = false;
};

