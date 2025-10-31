#include "StrategyConfigController.hpp"

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QLoggingCategory>

Q_LOGGING_CATEGORY(lcStrategyConfig, "bot.shell.strategy.config")

namespace {
QString normalizePath(const QString& path)
{
    if (path.trimmed().isEmpty())
        return {};
    QFileInfo info(path);
    if (info.isRelative())
        info.setFile(QDir::current().absoluteFilePath(path));
    return QDir::cleanPath(info.absoluteFilePath());
}

QVariantMap jsonObjectToVariantMap(const QJsonObject& object)
{
    QVariantMap map;
    for (auto it = object.begin(); it != object.end(); ++it) {
        if (it->isObject()) {
            map.insert(it.key(), jsonObjectToVariantMap(it->toObject()));
        } else if (it->isArray()) {
            QVariantList list;
            const auto array = it->toArray();
            list.reserve(array.size());
            for (const auto& value : array) {
                if (value.isObject())
                    list.append(jsonObjectToVariantMap(value.toObject()));
                else if (value.isArray()) {
                    QVariantList nested;
                    const auto nestedArray = value.toArray();
                    nested.reserve(nestedArray.size());
                    for (const auto& nestedValue : nestedArray) {
                        if (nestedValue.isObject())
                            nested.append(jsonObjectToVariantMap(nestedValue.toObject()));
                        else
                            nested.append(nestedValue.toVariant());
                    }
                    list.append(nested);
                } else {
                    list.append(value.toVariant());
                }
            }
            map.insert(it.key(), list);
        } else {
            map.insert(it.key(), it->toVariant());
        }
    }
    return map;
}

QJsonObject variantMapToJson(const QVariantMap& map)
{
    QJsonObject object;
    for (auto it = map.begin(); it != map.end(); ++it) {
        if (it->typeId() == QMetaType::QVariantMap) {
            object.insert(it.key(), variantMapToJson(it.value().toMap()));
        } else if (it->typeId() == QMetaType::QVariantList) {
            QJsonArray array;
            const QVariantList list = it->toList();
            array.reserve(list.size());
            for (const QVariant& value : list) {
                if (value.typeId() == QMetaType::QVariantMap)
                    array.append(variantMapToJson(value.toMap()));
                else if (value.typeId() == QMetaType::QVariantList) {
                    const QVariantList nested = value.toList();
                    QJsonArray nestedArray;
                    nestedArray.reserve(nested.size());
                    for (const QVariant& nestedValue : nested) {
                        if (nestedValue.typeId() == QMetaType::QVariantMap)
                            nestedArray.append(variantMapToJson(nestedValue.toMap()));
                        else if (nestedValue.typeId() == QMetaType::QVariantList) {
                            nestedArray.append(QJsonValue::fromVariant(nestedValue));
                        } else {
                            nestedArray.append(QJsonValue::fromVariant(nestedValue));
                        }
                    }
                    array.append(nestedArray);
                } else {
                    array.append(QJsonValue::fromVariant(value));
                }
            }
            object.insert(it.key(), array);
        } else {
            object.insert(it.key(), QJsonValue::fromVariant(it.value()));
        }
    }
    return object;
}

} // namespace

StrategyConfigController::StrategyConfigController(QObject* parent)
    : QObject(parent)
{
}

StrategyConfigController::~StrategyConfigController() = default;

void StrategyConfigController::setConfigPath(const QString& path)
{
    const QString normalized = normalizePath(path);
    if (m_configPath == normalized)
        return;
    m_configPath = normalized;
}

void StrategyConfigController::setPythonExecutable(const QString& executable)
{
    if (!executable.trimmed().isEmpty())
        m_pythonExecutable = executable.trimmed();
}

void StrategyConfigController::setScriptPath(const QString& path)
{
    const QString normalized = normalizePath(path);
    if (m_scriptPath == normalized)
        return;
    m_scriptPath = normalized;
}

bool StrategyConfigController::refresh()
{
    if (!ensureReady())
        return false;

    m_busy = true;
    Q_EMIT busyChanged();

    const BridgeResult result = invokeBridge({QStringLiteral("--dump"), QStringLiteral("--section"), QStringLiteral("all")});
    m_busy = false;
    Q_EMIT busyChanged();

    if (!result.ok) {
        m_lastError = result.errorMessage;
        Q_EMIT lastErrorChanged();
        return false;
    }

    if (!parseDump(result.stdoutData)) {
        m_lastError = tr("Nie udało się sparsować konfiguracji mostka.");
        Q_EMIT lastErrorChanged();
        return false;
    }

    if (!m_lastError.isEmpty()) {
        m_lastError.clear();
        Q_EMIT lastErrorChanged();
    }

    Q_EMIT decisionConfigChanged();
    Q_EMIT schedulerListChanged();
    return true;
}

QVariantMap StrategyConfigController::decisionConfigSnapshot() const
{
    return m_decisionConfig;
}

QVariantList StrategyConfigController::schedulerList() const
{
    return m_schedulerList;
}

QVariantMap StrategyConfigController::schedulerConfigSnapshot(const QString& name) const
{
    return m_schedulerConfigs.value(name);
}

bool StrategyConfigController::saveDecisionConfig(const QVariantMap& config)
{
    if (!ensureReady())
        return false;

    QJsonObject root;
    root.insert(QStringLiteral("decision"), variantMapToJson(config));
    const QByteArray payload = QJsonDocument(root).toJson(QJsonDocument::Compact);

    m_busy = true;
    Q_EMIT busyChanged();

    const BridgeResult result = invokeBridge({QStringLiteral("--apply")}, payload);

    m_busy = false;
    Q_EMIT busyChanged();

    if (!handleBridgeValidation(result))
        return false;

    return refresh();
}

bool StrategyConfigController::saveSchedulerConfig(const QString& name, const QVariantMap& config)
{
    if (!ensureReady())
        return false;

    if (name.trimmed().isEmpty()) {
        m_lastError = tr("Nie wskazano nazwy schedulera do zapisania.");
        Q_EMIT lastErrorChanged();
        return false;
    }

    QJsonObject schedulers;
    schedulers.insert(name, variantMapToJson(config));

    QJsonObject root;
    root.insert(QStringLiteral("schedulers"), schedulers);
    const QByteArray payload = QJsonDocument(root).toJson(QJsonDocument::Compact);

    m_busy = true;
    Q_EMIT busyChanged();

    const BridgeResult result = invokeBridge({QStringLiteral("--apply")}, payload);

    m_busy = false;
    Q_EMIT busyChanged();

    if (!handleBridgeValidation(result))
        return false;

    return refresh();
}

bool StrategyConfigController::removeSchedulerConfig(const QString& name)
{
    if (!ensureReady())
        return false;

    const QString trimmed = name.trimmed();
    if (trimmed.isEmpty()) {
        m_lastError = tr("Nie wskazano nazwy schedulera do usunięcia.");
        Q_EMIT lastErrorChanged();
        return false;
    }

    QJsonObject schedulers;
    schedulers.insert(trimmed, QJsonValue());

    QJsonObject root;
    root.insert(QStringLiteral("schedulers"), schedulers);
    const QByteArray payload = QJsonDocument(root).toJson(QJsonDocument::Compact);

    m_busy = true;
    Q_EMIT busyChanged();

    const BridgeResult result = invokeBridge({QStringLiteral("--apply")}, payload);

    m_busy = false;
    Q_EMIT busyChanged();

    if (!handleBridgeValidation(result))
        return false;

    return refresh();
}

bool StrategyConfigController::removeSchedulerConfig(const QString& name)
{
    if (!ensureReady())
        return false;

    const QString trimmed = name.trimmed();
    if (trimmed.isEmpty()) {
        m_lastError = tr("Nie wskazano nazwy schedulera do usunięcia.");
        Q_EMIT lastErrorChanged();
        return false;
    }

    QJsonObject schedulers;
    schedulers.insert(trimmed, QJsonValue());

    QJsonObject root;
    root.insert(QStringLiteral("schedulers"), schedulers);
    const QByteArray payload = QJsonDocument(root).toJson(QJsonDocument::Compact);

    m_busy = true;
    Q_EMIT busyChanged();

    const BridgeResult result = invokeBridge({QStringLiteral("--apply")}, payload);

    m_busy = false;
    Q_EMIT busyChanged();

    if (!result.ok) {
        m_lastError = result.errorMessage;
        Q_EMIT lastErrorChanged();
        return false;
    }

    return refresh();
}

bool StrategyConfigController::runSchedulerNow(const QString& name)
{
    if (!ensureReady())
        return false;

    const QString trimmed = name.trimmed();
    if (trimmed.isEmpty()) {
        m_lastError = tr("Nie wskazano nazwy schedulera do uruchomienia.");
        Q_EMIT lastErrorChanged();
        return false;
    }

    m_busy = true;
    Q_EMIT busyChanged();

    const BridgeResult result = invokeBridge({QStringLiteral("--run-scheduler"), QStringLiteral("--name"), trimmed});

    m_busy = false;
    Q_EMIT busyChanged();

    if (!result.ok) {
        m_lastError = result.errorMessage;
        Q_EMIT lastErrorChanged();
        return false;
    }

    if (!m_lastError.isEmpty()) {
        m_lastError.clear();
        Q_EMIT lastErrorChanged();
    }
    return true;
}

StrategyConfigController::BridgeResult StrategyConfigController::invokeBridge(const QStringList& args,
                                                                             const QByteArray& stdinData) const
{
    BridgeResult result;

    QStringList fullArgs;
    fullArgs << m_scriptPath;
    fullArgs << QStringLiteral("--config") << m_configPath;
    fullArgs << args;

    QProcess process;
    process.setProcessChannelMode(QProcess::SeparateChannels);
    process.start(m_pythonExecutable, fullArgs);
    if (!process.waitForStarted()) {
        result.ok = false;
        result.errorMessage = tr("Nie udało się uruchomić mostka konfiguracji (%1)")
                                  .arg(process.errorString());
        return result;
    }

    if (!stdinData.isEmpty()) {
        process.write(stdinData);
    }
    process.closeWriteChannel();

    if (!process.waitForFinished(-1)) {
        result.ok = false;
        result.errorMessage = tr("Mostek konfiguracji nie zakończył się poprawnie (%1)")
                                  .arg(process.errorString());
        process.kill();
        process.waitForFinished();
        return result;
    }

    result.stdoutData = process.readAllStandardOutput();
    const QByteArray stderrData = process.readAllStandardError();
    if (!stderrData.isEmpty())
        qCWarning(lcStrategyConfig) << "ui_config_bridge stderr:" << stderrData;

    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
        result.ok = false;
        result.errorMessage = QString::fromUtf8(stderrData.isEmpty() ? result.stdoutData : stderrData)
                                  .trimmed();
        if (result.errorMessage.isEmpty())
            result.errorMessage = tr("Mostek konfiguracji zwrócił kod %1")
                                      .arg(process.exitCode());
        return result;
    }

    result.ok = true;
    return result;
}

bool StrategyConfigController::handleBridgeValidation(const BridgeResult& result)
{
    if (!result.ok) {
        if (m_lastError != result.errorMessage) {
            m_lastError = result.errorMessage;
            Q_EMIT lastErrorChanged();
        }
        return false;
    }

    auto clearIssues = [this]() {
        if (!m_validationIssues.isEmpty()) {
            m_validationIssues.clear();
            Q_EMIT validationIssuesChanged();
        }
    };

    if (result.stdoutData.isEmpty()) {
        clearIssues();
        return true;
    }

    QJsonParseError parseError;
    const QJsonDocument doc = QJsonDocument::fromJson(result.stdoutData, &parseError);
    if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
        clearIssues();
        return true;
    }

    const QJsonObject root = doc.object();
    const bool ok = root.value(QStringLiteral("ok")).toBool(true);

    QStringList errors;
    const QJsonValue errorsValue = root.value(QStringLiteral("errors"));
    if (errorsValue.isArray()) {
        const auto array = errorsValue.toArray();
        for (const QJsonValue& entry : array) {
            if (entry.isString())
                errors.append(entry.toString());
        }
    }

    QStringList issues;
    const QJsonValue issuesValue = root.value(QStringLiteral("issues"));
    if (issuesValue.isArray()) {
        const auto array = issuesValue.toArray();
        issues.reserve(array.size());
        for (const QJsonValue& item : array) {
            if (item.isString()) {
                issues.append(item.toString());
                continue;
            }
            if (item.isObject()) {
                const QJsonObject obj = item.toObject();
                const QString severity = obj.value(QStringLiteral("severity")).toString();
                const QString entryName = obj.value(QStringLiteral("entry")).toString();
                const QString field = obj.value(QStringLiteral("field")).toString();
                const QString message = obj.value(QStringLiteral("message")).toString();
                const QString suggested = obj.value(QStringLiteral("suggested")).toString();

                QStringList parts;
                if (!severity.trimmed().isEmpty())
                    parts.append(QStringLiteral("[%1]").arg(severity.trimmed().toUpper()));
                QString location;
                if (!entryName.trimmed().isEmpty() && !field.trimmed().isEmpty())
                    location = QStringLiteral("%1.%2").arg(entryName.trimmed(), field.trimmed());
                else if (!entryName.trimmed().isEmpty())
                    location = entryName.trimmed();
                else if (!field.trimmed().isEmpty())
                    location = field.trimmed();
                if (!location.isEmpty())
                    parts.append(location);
                if (!message.trimmed().isEmpty())
                    parts.append(message.trimmed());
                if (!suggested.trimmed().isEmpty())
                    parts.append(QStringLiteral("(sugerowane: %1)").arg(suggested.trimmed()));
                const QString formatted = parts.join(QStringLiteral(" ")).trimmed();
                if (!formatted.isEmpty())
                    issues.append(formatted);
            }
        }
    }

    if (issues != m_validationIssues) {
        m_validationIssues = issues;
        Q_EMIT validationIssuesChanged();
    }

    if (!ok || !errors.isEmpty()) {
        const QString message = errors.isEmpty() ? tr("Operacja zakończyła się błędami walidacji.")
                                                : errors.join(QStringLiteral("\n"));
        if (m_lastError != message) {
            m_lastError = message;
            Q_EMIT lastErrorChanged();
        }
        return false;
    }

    if (!m_lastError.isEmpty()) {
        m_lastError.clear();
        Q_EMIT lastErrorChanged();
    }

    return true;
}

bool StrategyConfigController::parseDump(const QByteArray& payload)
{
    QJsonParseError error;
    const QJsonDocument doc = QJsonDocument::fromJson(payload, &error);
    if (error.error != QJsonParseError::NoError || !doc.isObject())
        return false;

    const QJsonObject root = doc.object();
    if (root.contains(QStringLiteral("decision")) && root.value(QStringLiteral("decision")).isObject()) {
        m_decisionConfig = jsonObjectToVariantMap(root.value(QStringLiteral("decision")).toObject());
    } else {
        m_decisionConfig.clear();
    }

    m_schedulerList.clear();
    m_schedulerConfigs.clear();

    if (root.contains(QStringLiteral("schedulers")) && root.value(QStringLiteral("schedulers")).isObject()) {
        const QJsonObject schedulersObject = root.value(QStringLiteral("schedulers")).toObject();
        QVariantList list;
        for (auto it = schedulersObject.begin(); it != schedulersObject.end(); ++it) {
            if (!it->isObject())
                continue;
            QVariantMap map = jsonObjectToVariantMap(it->toObject());
            if (!map.contains(QStringLiteral("name")))
                map.insert(QStringLiteral("name"), it.key());
            list.append(map);
            m_schedulerConfigs.insert(map.value(QStringLiteral("name")).toString(), map);
        }
        m_schedulerList = list;
    }

    return true;
}

bool StrategyConfigController::ensureReady() const
{
    if (m_configPath.isEmpty()) {
        qCWarning(lcStrategyConfig) << "Brak ścieżki pliku core.yaml";
        return false;
    }
    if (m_scriptPath.isEmpty()) {
        qCWarning(lcStrategyConfig) << "Brak ścieżki do scripts/ui_config_bridge.py";
        return false;
    }
    return true;
}
