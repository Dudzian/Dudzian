#pragma once

#include <QObject>
#include <QVariant>
#include <QStringList>

#include <memory>

class MarketplaceController : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool busy READ busy NOTIFY busyChanged)
    Q_PROPERTY(QVariantList presets READ presets NOTIFY presetsChanged)
    Q_PROPERTY(QStringList categories READ categories NOTIFY categoriesChanged)
    Q_PROPERTY(QString lastError READ lastError NOTIFY lastErrorChanged)

public:
    explicit MarketplaceController(QObject* parent = nullptr);
    ~MarketplaceController() override;

    bool busy() const { return m_busy; }
    QVariantList presets() const { return m_presets; }
    QStringList categories() const { return m_categories; }
    QString lastError() const { return m_lastError; }

    void setPythonExecutable(const QString& executable);
    void setBridgeScriptPath(const QString& path);
    void setPresetsDirectory(const QString& path);
    void setLicensesPath(const QString& path);
    void setSigningKeys(const QStringList& keys);
    void setSigningKeyFiles(const QStringList& paths);
    void setFingerprintOverride(const QString& fingerprint);

    QString pythonExecutable() const { return m_pythonExecutable; }
    QString bridgeScriptPath() const { return m_bridgeScriptPath; }
    QString presetsDirectory() const { return m_presetsDir; }
    QString licensesPath() const { return m_licensesPath; }
    QStringList signingKeys() const { return m_signingKeys; }
    QStringList signingKeyFiles() const { return m_signingKeyFiles; }
    QString fingerprintOverride() const { return m_fingerprintOverride; }

    Q_INVOKABLE bool refreshPresets();
    Q_INVOKABLE bool activatePreset(const QString& presetId, const QVariantMap& licensePayload);
    Q_INVOKABLE bool deactivatePreset(const QString& presetId);
    Q_INVOKABLE bool assignPresetToPortfolio(const QString& presetId, const QString& portfolioId);
    Q_INVOKABLE bool unassignPresetFromPortfolio(const QString& presetId, const QString& portfolioId);
    Q_INVOKABLE QVariantList presetsForCategory(const QString& category) const;
    Q_INVOKABLE QVariantMap presetDetails(const QString& presetId) const;

signals:
    void busyChanged();
    void presetsChanged();
    void categoriesChanged();
    void lastErrorChanged();

private:
    struct BridgeResult {
        bool ok = false;
        QByteArray stdoutData;
        QString errorMessage;
    };

    BridgeResult runBridge(const QStringList& arguments, const QByteArray& stdinData = QByteArray());
    bool ensureReady(QString* message = nullptr) const;
    QVariantList parsePresets(const QByteArray& payload) const;
    QStringList buildCommonArguments() const;
    bool applyPresetUpdate(const QVariantMap& presetPayload);
    void rebuildCategories();
    QVariantMap presetById(const QString& presetId) const;

    QString m_pythonExecutable = QStringLiteral("python3");
    QString m_bridgeScriptPath;
    QString m_presetsDir;
    QString m_licensesPath;
    QStringList m_signingKeys;
    QStringList m_signingKeyFiles;
    QString m_fingerprintOverride;

    bool m_busy = false;
    QVariantList m_presets;
    QStringList m_categories;
    QString m_lastError;
};
