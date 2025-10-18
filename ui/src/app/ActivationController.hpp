#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QVariantList>
#include <QVariantMap>

class ActivationController : public QObject {
    Q_OBJECT
    Q_PROPERTY(QVariantMap fingerprint READ fingerprint NOTIFY fingerprintChanged)
    Q_PROPERTY(QVariantList licenses READ licenses NOTIFY licensesChanged)
    Q_PROPERTY(QString lastError READ lastError NOTIFY errorChanged)

public:
    explicit ActivationController(QObject* parent = nullptr);

    QVariantMap  fingerprint() const { return m_fingerprint; }
    QVariantList licenses() const { return m_licenses; }
    QString      lastError() const { return m_lastError; }

    Q_INVOKABLE void refresh();
    Q_INVOKABLE void reloadRegistry();
    Q_INVOKABLE bool exportFingerprint(const QUrl& destination) const;

    void setPythonExecutable(const QString& value);
    void setKeysFile(const QString& value);
    void setRotationLog(const QString& value);
    void setRegistryPath(const QString& value);
    void setDongleHint(const QString& value);

signals:
    void fingerprintChanged();
    void licensesChanged();
    void errorChanged();

private:
    void updateFingerprint();
    void updateLicenses();
    void setError(const QString& message);
    void clearError();
    QStringList loadKeyArguments() const;
    QVariantMap enrichFingerprintPayload(const QVariantMap& payload) const;

    QString     m_pythonExecutable;
    QString     m_keysFile;
    QString     m_rotationLog;
    QString     m_registryPath;
    QString     m_dongleHint;
    QVariantMap m_fingerprint;
    QVariantList m_licenses;
    QString      m_lastError;
};

