#pragma once

#include <QObject>
#include <QPointer>
#include <QVariant>
#include <QStringList>

class StrategyConfigController;
class MarketplaceController;
class RiskStateModel;
class AlertsModel;

class ConfigurationWizardController : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool busy READ busy NOTIFY busyChanged)
    Q_PROPERTY(QVariantList steps READ steps NOTIFY stepsChanged)
    Q_PROPERTY(int currentStepIndex READ currentStepIndex NOTIFY currentStepIndexChanged)
    Q_PROPERTY(bool completed READ completed NOTIFY completedChanged)

public:
    explicit ConfigurationWizardController(QObject* parent = nullptr);

    bool busy() const { return m_busy; }
    QVariantList steps() const { return m_steps; }
    int currentStepIndex() const { return m_currentStepIndex; }
    bool completed() const { return m_completed; }

    void setStrategyConfigController(StrategyConfigController* controller);
    void setMarketplaceController(MarketplaceController* controller);
    void setRiskModel(RiskStateModel* model);
    void setAlertsModel(AlertsModel* model);

    Q_INVOKABLE bool start(const QString& profileId);
    Q_INVOKABLE QVariantMap currentStep() const;
    Q_INVOKABLE bool commitStep(const QVariantMap& payload);
    Q_INVOKABLE void reset();
    Q_INVOKABLE bool finish();

signals:
    void busyChanged();
    void stepsChanged();
    void currentStepIndexChanged();
    void completedChanged();
    void wizardStarted(const QString& profileId);
    void wizardStepCompleted(const QString& profileId, const QString& stepId);
    void wizardCompleted(const QString& profileId);
    void wizardAborted(const QString& profileId, const QString& reason);

private:
    void setBusy(bool busy);
    void rebuildDefaultSteps();
    void raiseWizardAlert(const QString& message, int severity = 0);
    bool applyCollectedConfig();
    bool ensureControllersReady(QString* errorMessage = nullptr) const;

    QPointer<StrategyConfigController> m_strategyController;
    QPointer<MarketplaceController> m_marketplaceController;
    QPointer<RiskStateModel> m_riskModel;
    QPointer<AlertsModel> m_alertsModel;

    QString m_profileId;
    QVariantList m_steps;
    QVariantMap m_collectedConfig;
    bool m_busy = false;
    int m_currentStepIndex = -1;
    bool m_completed = false;
};
