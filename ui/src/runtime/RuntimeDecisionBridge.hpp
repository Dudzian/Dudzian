#pragma once

#include <QObject>
#include <QVariantList>

class RuntimeDecisionBridge : public QObject {
    Q_OBJECT
    Q_PROPERTY(QVariantList decisions READ decisions NOTIFY decisionsChanged)
    Q_PROPERTY(QString errorMessage READ errorMessage NOTIFY errorMessageChanged)
    Q_PROPERTY(QString logPath READ logPath WRITE setLogPath NOTIFY logPathChanged)

public:
    explicit RuntimeDecisionBridge(QObject* parent = nullptr);

    QVariantList decisions() const { return m_decisions; }
    QString errorMessage() const { return m_errorMessage; }

    QString logPath() const { return m_logPath; }
    void setLogPath(const QString& path);

    Q_INVOKABLE QVariantList loadRecentDecisions(int limit = 0);

signals:
    void decisionsChanged();
    void errorMessageChanged();
    void logPathChanged();

private:
    QVariantList readDecisions(int limit, QString* error) const;
    QVariantMap buildPayload(const QVariantMap& record) const;
    QStringList resolveCandidateFiles() const;
    static QVariant normalizeBoolean(const QVariant& value);
    static QString camelize(const QString& prefix, const QString& key);

    QVariantList m_decisions;
    QString m_errorMessage;
    QString m_logPath;
};

