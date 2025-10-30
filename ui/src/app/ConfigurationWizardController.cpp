#include "ConfigurationWizardController.hpp"

#include <QDateTime>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLoggingCategory>
#include <QtGlobal>

#include "MarketplaceController.hpp"
#include "StrategyConfigController.hpp"
#include "models/AlertsModel.hpp"
#include "models/RiskStateModel.hpp"

Q_LOGGING_CATEGORY(lcConfigWizard, "bot.shell.ui.config_wizard")

namespace {
QVariantMap buildStep(const QString& id,
                      const QString& title,
                      const QString& description,
                      const QString& category,
                      const QVariantMap& metadata = {})
{
    QVariantMap step;
    step.insert(QStringLiteral("id"), id);
    step.insert(QStringLiteral("title"), title);
    step.insert(QStringLiteral("description"), description);
    step.insert(QStringLiteral("category"), category);
    if (!metadata.isEmpty())
        step.insert(QStringLiteral("metadata"), metadata);
    return step;
}

QVariantMap mergeVariantMaps(QVariantMap base, const QVariantMap& overlay)
{
    for (auto it = overlay.begin(); it != overlay.end(); ++it) {
        if (it->typeId() == QMetaType::QVariantMap && base.value(it.key()).typeId() == QMetaType::QVariantMap) {
            base.insert(it.key(), mergeVariantMaps(base.value(it.key()).toMap(), it->toMap()));
        } else {
            base.insert(it.key(), it.value());
        }
    }
    return base;
}
} // namespace

ConfigurationWizardController::ConfigurationWizardController(QObject* parent)
    : QObject(parent)
{
}

void ConfigurationWizardController::setStrategyConfigController(StrategyConfigController* controller)
{
    if (m_strategyController == controller)
        return;
    m_strategyController = controller;
}

void ConfigurationWizardController::setMarketplaceController(MarketplaceController* controller)
{
    if (m_marketplaceController == controller)
        return;
    m_marketplaceController = controller;
}

void ConfigurationWizardController::setRiskModel(RiskStateModel* model)
{
    if (m_riskModel == model)
        return;
    m_riskModel = model;
}

void ConfigurationWizardController::setAlertsModel(AlertsModel* model)
{
    if (m_alertsModel == model)
        return;
    m_alertsModel = model;
}

bool ConfigurationWizardController::start(const QString& profileId)
{
    QString trimmed = profileId.trimmed();
    if (trimmed.isEmpty())
        trimmed = QStringLiteral("default");

    QString message;
    if (!ensureControllersReady(&message)) {
        raiseWizardAlert(message, AlertsModel::Critical);
        return false;
    }

    m_profileId = trimmed;
    m_collectedConfig.clear();
    rebuildDefaultSteps();
    if (m_steps.isEmpty()) {
        raiseWizardAlert(tr("Kreator nie ma żadnych kroków do wykonania."), AlertsModel::Warning);
        return false;
    }

    m_currentStepIndex = 0;
    m_completed = false;
    Q_EMIT currentStepIndexChanged();
    Q_EMIT completedChanged();

    raiseWizardAlert(tr("Uruchomiono kreator konfiguracji profilu %1.").arg(m_profileId), AlertsModel::Info);
    return true;
}

QVariantMap ConfigurationWizardController::currentStep() const
{
    if (m_currentStepIndex < 0 || m_currentStepIndex >= m_steps.size())
        return {};
    return m_steps.at(m_currentStepIndex).toMap();
}

bool ConfigurationWizardController::commitStep(const QVariantMap& payload)
{
    if (m_currentStepIndex < 0 || m_currentStepIndex >= m_steps.size()) {
        raiseWizardAlert(tr("Kreator nie jest aktywny."), AlertsModel::Warning);
        return false;
    }

    const QVariantMap step = m_steps.at(m_currentStepIndex).toMap();
    const QString stepId = step.value(QStringLiteral("id")).toString();
    QVariantMap normalizedPayload = payload;
    normalizedPayload.insert(QStringLiteral("completedAt"), QDateTime::currentDateTimeUtc());
    m_collectedConfig.insert(stepId, normalizedPayload);

    if (m_currentStepIndex + 1 < m_steps.size()) {
        m_currentStepIndex += 1;
        Q_EMIT currentStepIndexChanged();
        return true;
    }

    return finish();
}

void ConfigurationWizardController::reset()
{
    m_steps.clear();
    m_collectedConfig.clear();
    m_currentStepIndex = -1;
    m_completed = false;
    m_profileId.clear();
    Q_EMIT stepsChanged();
    Q_EMIT currentStepIndexChanged();
    Q_EMIT completedChanged();
}

bool ConfigurationWizardController::finish()
{
    if (m_completed)
        return true;

    if (!applyCollectedConfig()) {
        raiseWizardAlert(tr("Nie udało się zapisać konfiguracji profilu %1.").arg(m_profileId), AlertsModel::Critical);
        return false;
    }

    m_completed = true;
    Q_EMIT completedChanged();
    raiseWizardAlert(tr("Zakończono kreator profilu %1.").arg(m_profileId), AlertsModel::Info);
    Q_EMIT wizardCompleted(m_profileId);
    return true;
}

void ConfigurationWizardController::setBusy(bool busy)
{
    if (m_busy == busy)
        return;
    m_busy = busy;
    Q_EMIT busyChanged();
}

void ConfigurationWizardController::rebuildDefaultSteps()
{
    QVariantList steps;

    QVariantMap licenseMeta;
    licenseMeta.insert(QStringLiteral("requiresLicense"), true);
    steps.append(buildStep(QStringLiteral("license"),
                           tr("Aktywacja licencji"),
                           tr("Zweryfikuj fingerprint urządzenia i aktywuj licencję OEM."),
                           tr("Licencja"),
                           licenseMeta));

    QVariantMap connectivityMeta;
    connectivityMeta.insert(QStringLiteral("exchanges"), QStringList());
    steps.append(buildStep(QStringLiteral("connectivity"),
                           tr("Połączenia giełdowe"),
                           tr("Skonfiguruj klucze API i preferowane instrumenty."),
                           tr("Połączenia"),
                           connectivityMeta));

    QVariantMap automationMeta;
    automationMeta.insert(QStringLiteral("supportsSchedulers"), true);
    automationMeta.insert(QStringLiteral("supportsRisk"), true);
    steps.append(buildStep(QStringLiteral("automation"),
                           tr("Automatyzacja strategii"),
                           tr("Zdefiniuj harmonogramy, parametry DCA i market-making."),
                           tr("Strategie"),
                           automationMeta));

    QVariantMap marketplaceMeta;
    marketplaceMeta.insert(QStringLiteral("requiresSignature"), true);
    steps.append(buildStep(QStringLiteral("marketplace"),
                           tr("Marketplace presetów"),
                           tr("Wybierz podpisane presety strategii dla profilu."),
                           tr("Marketplace"),
                           marketplaceMeta));

    QVariantMap telemetryMeta;
    telemetryMeta.insert(QStringLiteral("riskSnapshot"), m_riskModel ? m_riskModel->currentSnapshot() : QVariantMap());
    steps.append(buildStep(QStringLiteral("telemetry"),
                           tr("Monitoring wyników"),
                           tr("Potwierdź wskaźniki ryzyka i alerty."),
                           tr("Monitoring"),
                           telemetryMeta));

    if (steps != m_steps) {
        m_steps = steps;
        Q_EMIT stepsChanged();
    }
}

void ConfigurationWizardController::raiseWizardAlert(const QString& message, int severity)
{
    if (!m_alertsModel)
        return;
    const QString alertId = QStringLiteral("wizard:%1:%2")
                                .arg(m_profileId.isEmpty() ? QStringLiteral("default") : m_profileId)
                                .arg(severity);
    m_alertsModel->raiseAlert(alertId,
                              tr("Kreator konfiguracji"),
                              message,
                              static_cast<AlertsModel::Severity>(qBound(0, severity, 2)),
                              true);
}

bool ConfigurationWizardController::applyCollectedConfig()
{
    if (!m_strategyController)
        return false;

    QVariantMap baseConfig = m_strategyController->decisionConfigSnapshot();
    QVariantMap overlay;

    const QVariant licenseVariant = m_collectedConfig.value(QStringLiteral("license"));
    if (licenseVariant.typeId() == QMetaType::QVariantMap)
        overlay.insert(QStringLiteral("license"), licenseVariant.toMap());

    const QVariant connectivity = m_collectedConfig.value(QStringLiteral("connectivity"));
    if (connectivity.typeId() == QMetaType::QVariantMap)
        overlay.insert(QStringLiteral("connectivity"), connectivity.toMap());

    const QVariant automation = m_collectedConfig.value(QStringLiteral("automation"));
    if (automation.typeId() == QMetaType::QVariantMap)
        overlay.insert(QStringLiteral("automation"), automation.toMap());

    const QVariant telemetry = m_collectedConfig.value(QStringLiteral("telemetry"));
    if (telemetry.typeId() == QMetaType::QVariantMap)
        overlay.insert(QStringLiteral("telemetry"), telemetry.toMap());

    const QVariant marketplace = m_collectedConfig.value(QStringLiteral("marketplace"));
    if (marketplace.typeId() == QMetaType::QVariantMap)
        overlay.insert(QStringLiteral("marketplace"), marketplace.toMap());

    baseConfig = mergeVariantMaps(baseConfig, overlay);

    setBusy(true);
    const bool ok = m_strategyController->saveDecisionConfig(baseConfig);
    setBusy(false);

    if (ok && m_marketplaceController)
        m_marketplaceController->refreshPresets();

    return ok;
}

bool ConfigurationWizardController::ensureControllersReady(QString* errorMessage) const
{
    if (!m_strategyController) {
        if (errorMessage)
            *errorMessage = tr("Brak kontrolera konfiguracji strategii.");
        return false;
    }
    if (!m_marketplaceController) {
        if (errorMessage)
            *errorMessage = tr("Brak kontrolera marketplace.");
        return false;
    }
    return true;
}
