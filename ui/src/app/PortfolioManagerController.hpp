#pragma once

#include <QObject>
#include <QVariant>
#include <QString>
#include <QStringList>

class PortfolioManagerController : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool busy READ busy NOTIFY busyChanged)
    Q_PROPERTY(QVariantList portfolios READ portfolios NOTIFY portfoliosChanged)
    Q_PROPERTY(QString lastError READ lastError NOTIFY lastErrorChanged)

public:
    explicit PortfolioManagerController(QObject* parent = nullptr);
    ~PortfolioManagerController() override;

    bool busy() const { return m_busy; }
    QVariantList portfolios() const { return m_portfolios; }
    QString lastError() const { return m_lastError; }

    void setPythonExecutable(const QString& executable);
    void setBridgeScriptPath(const QString& path);
    void setStorePath(const QString& path);

    QString pythonExecutable() const { return m_pythonExecutable; }
    QString bridgeScriptPath() const { return m_bridgeScriptPath; }
    QString storePath() const { return m_storePath; }

    Q_INVOKABLE bool refreshPortfolios();
    Q_INVOKABLE bool applyPortfolio(const QVariantMap& payload);
    Q_INVOKABLE bool removePortfolio(const QString& portfolioId);

signals:
    void busyChanged();
    void portfoliosChanged();
    void lastErrorChanged();

private:
    struct BridgeResult {
        bool ok = false;
        QByteArray stdoutData;
        QString errorMessage;
    };

    BridgeResult runBridge(const QStringList& arguments, const QByteArray& stdinData = QByteArray());
    bool ensureReady(QString* message = nullptr) const;
    QVariantList parsePortfolios(const QByteArray& payload) const;

    QString m_pythonExecutable = QStringLiteral("python3");
    QString m_bridgeScriptPath;
    QString m_storePath;
    bool m_busy = false;
    QVariantList m_portfolios;
    QString m_lastError;
};
