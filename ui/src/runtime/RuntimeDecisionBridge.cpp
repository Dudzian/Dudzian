#include "runtime/RuntimeDecisionBridge.hpp"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLoggingCategory>
#include <QRegularExpression>
#include <QTextStream>
#include <QVariantMap>

#include <algorithm>

#include "runtime/OfflineRuntimeBridge.hpp"
#include "grpc/BotCoreLocalService.hpp"

Q_LOGGING_CATEGORY(lcRuntimeBridge, "bot.shell.runtime.bridge")

namespace {

QString normalizePath(const QString& raw)
{
    QFileInfo info(raw.trimmed());
    if (!info.exists())
        return info.absoluteFilePath();
    return info.absoluteFilePath();
}

void emitAutomationSignals(RuntimeDecisionBridge* bridge, const QVariantMap& snapshot)
{
    if (!bridge)
        return;

    const QVariantMap automation = snapshot.value(QStringLiteral("automation")).toMap();
    if (!automation.isEmpty())
        emit bridge->automationStateChanged(automation.value(QStringLiteral("running")).toBool());

    const QVariantMap alerts = snapshot.value(QStringLiteral("alerts")).toMap();
    if (!alerts.isEmpty())
        emit bridge->alertPreferencesChanged(alerts);
}

QString toString(const QVariant& value)
{
    if (value.typeId() == QMetaType::QString)
        return value.toString();
    if (value.canConvert<double>()) {
        bool ok = false;
        const double numeric = value.toDouble(&ok);
        if (ok)
            return QString::number(numeric, 'f', 8).remove(QRegularExpression(QStringLiteral("0+$")))
                .remove(QRegularExpression(QStringLiteral("\\.$")));
    }
    return value.toString();
}

} // namespace

RuntimeDecisionBridge::RuntimeDecisionBridge(QObject* parent)
    : QObject(parent)
{
}

void RuntimeDecisionBridge::setOfflineBridge(OfflineRuntimeBridge* bridge)
{
    if (m_offlineBridge == bridge)
        return;

    if (!m_offlineBridge.isNull())
        QObject::disconnect(m_offlineBridge.data(), nullptr, this, nullptr);

    m_offlineBridge = bridge;

    if (!m_offlineBridge.isNull()) {
        QObject::connect(m_offlineBridge.data(), &OfflineRuntimeBridge::automationStateChanged,
                         this, &RuntimeDecisionBridge::automationStateChanged);
        QObject::connect(m_offlineBridge.data(), &OfflineRuntimeBridge::alertPreferencesChanged,
                         this, &RuntimeDecisionBridge::alertPreferencesChanged);

        const QVariantMap snapshot = m_offlineBridge->autoModeSnapshot();
        emitAutomationSignals(this, snapshot);

        const QVariantMap storedAlerts = m_offlineBridge->alertPreferences();
        if (!storedAlerts.isEmpty())
            emit alertPreferencesChanged(storedAlerts);
    }
}

void RuntimeDecisionBridge::setLocalService(BotCoreLocalService* service)
{
    if (m_localService == service)
        return;

    if (!m_localService.isNull())
        QObject::disconnect(m_localService.data(), nullptr, this, nullptr);

    m_localService = service;

    if (!m_localService.isNull()) {
        QObject::connect(m_localService.data(), &QObject::destroyed, this, [this]() {
            const QVariantMap snapshot = autoModeSnapshot();
            emitAutomationSignals(this, snapshot);
        });
    }

    const QVariantMap snapshot = autoModeSnapshot();
    if (!snapshot.isEmpty())
        emitAutomationSignals(this, snapshot);
}

void RuntimeDecisionBridge::setLogPath(const QString& path)
{
    const QString normalized = normalizePath(path);
    if (m_logPath == normalized)
        return;
    m_logPath = normalized;
    emit logPathChanged();
}

QVariantList RuntimeDecisionBridge::loadRecentDecisions(int limit)
{
    QString error;
    const QVariantList result = readDecisions(limit, &error);

    if (error.isEmpty()) {
        if (m_errorMessage != QString()) {
            m_errorMessage.clear();
            emit errorMessageChanged();
        }
        m_decisions = result;
        emit decisionsChanged();
        return m_decisions;
    }

    m_decisions.clear();
    emit decisionsChanged();
    if (m_errorMessage != error) {
        m_errorMessage = error;
        emit errorMessageChanged();
    }
    return {};
}

QVariantMap RuntimeDecisionBridge::autoModeSnapshot() const
{
    if (!m_localService.isNull()) {
        const QVariantMap response = m_localService->fetchAutoModeSnapshot();
        if (!response.isEmpty())
            return response;
    }
    if (!m_offlineBridge.isNull())
        return m_offlineBridge->autoModeSnapshot();

    qCWarning(lcRuntimeBridge) << "autoModeSnapshot requested without offline bridge";
    return {};
}

void RuntimeDecisionBridge::toggleAutoMode(bool enabled)
{
    if (!m_localService.isNull()) {
        const QVariantMap response = m_localService->toggleAutoMode(enabled);
        if (!response.isEmpty())
            emitAutomationSignals(this, response);
        return;
    }
    if (!m_offlineBridge.isNull()) {
        m_offlineBridge->toggleAutoMode(enabled);
        return;
    }
    qCWarning(lcRuntimeBridge) << "toggleAutoMode called without offline bridge";
}

void RuntimeDecisionBridge::startAutomation()
{
    if (!m_localService.isNull()) {
        toggleAutoMode(true);
        return;
    }
    if (!m_offlineBridge.isNull()) {
        m_offlineBridge->startAutomation();
        return;
    }
    qCWarning(lcRuntimeBridge) << "startAutomation called without offline bridge";
}

void RuntimeDecisionBridge::stopAutomation()
{
    if (!m_localService.isNull()) {
        toggleAutoMode(false);
        return;
    }
    if (!m_offlineBridge.isNull()) {
        m_offlineBridge->stopAutomation();
        return;
    }
    qCWarning(lcRuntimeBridge) << "stopAutomation called without offline bridge";
}

void RuntimeDecisionBridge::updateAlertPreferences(const QVariantMap& preferences)
{
    if (!m_localService.isNull()) {
        const QVariantMap response = m_localService->updateAutoModeAlerts(preferences);
        if (!response.isEmpty())
            emitAutomationSignals(this, response);
        return;
    }
    if (!m_offlineBridge.isNull()) {
        m_offlineBridge->updateAlertPreferences(preferences);
        return;
    }
    qCWarning(lcRuntimeBridge) << "updateAlertPreferences called without offline bridge";
}

QVariantList RuntimeDecisionBridge::readDecisions(int limit, QString* error) const
{
    if (error)
        error->clear();

    const QStringList files = resolveCandidateFiles();
    if (files.isEmpty())
        return {};

    QVector<QVariantMap> records;
    for (const QString& filePath : files) {
        QFile file(filePath);
        if (!file.exists())
            continue;
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            if (error)
                *error = tr("Nie można otworzyć pliku dziennika decyzji: %1").arg(filePath);
            qCWarning(lcRuntimeBridge) << "Nie można odczytać" << filePath << file.errorString();
            return {};
        }

        while (!file.atEnd()) {
            const QByteArray line = file.readLine();
            const QByteArray trimmed = line.trimmed();
            if (trimmed.isEmpty())
                continue;
            QJsonParseError parseError{};
            const QJsonDocument document = QJsonDocument::fromJson(trimmed, &parseError);
            if (parseError.error != QJsonParseError::NoError || !document.isObject())
                continue;
            records.append(document.object().toVariantMap());
        }
    }

    if (records.isEmpty())
        return {};

    if (limit > 0 && records.size() > limit)
        records = records.mid(records.size() - limit, limit);

    QVariantList payloads;
    payloads.reserve(records.size());
    for (auto it = records.crbegin(); it != records.crend(); ++it)
        payloads.append(buildPayload(*it));

    return payloads;
}

QVariantMap RuntimeDecisionBridge::buildPayload(const QVariantMap& record) const
{
    static const QHash<QString, QString> baseFields{
        {QStringLiteral("event"), QStringLiteral("event")},
        {QStringLiteral("timestamp"), QStringLiteral("timestamp")},
        {QStringLiteral("environment"), QStringLiteral("environment")},
        {QStringLiteral("portfolio"), QStringLiteral("portfolio")},
        {QStringLiteral("risk_profile"), QStringLiteral("risk_profile")},
        {QStringLiteral("schedule"), QStringLiteral("schedule")},
        {QStringLiteral("strategy"), QStringLiteral("strategy")},
        {QStringLiteral("symbol"), QStringLiteral("symbol")},
        {QStringLiteral("side"), QStringLiteral("side")},
        {QStringLiteral("status"), QStringLiteral("status")},
        {QStringLiteral("quantity"), QStringLiteral("quantity")},
        {QStringLiteral("price"), QStringLiteral("price")},
    };

    QVariantMap base;
    for (const QString& key : baseFields.keys())
        base.insert(baseFields.value(key), QVariant());

    QVariantMap decisionPayload;
    QVariantMap aiPayload;
    QVariantMap regimePayload;
    QVariantMap extras;

    if (record.contains(QStringLiteral("confidence")))
        decisionPayload.insert(QStringLiteral("confidence"), record.value(QStringLiteral("confidence")));
    if (record.contains(QStringLiteral("latency_ms")))
        decisionPayload.insert(QStringLiteral("latencyMs"), record.value(QStringLiteral("latency_ms")));

    for (auto it = record.constBegin(); it != record.constEnd(); ++it) {
        const QString& key = it.key();
        const QVariant& value = it.value();

        if (key == QStringLiteral("confidence") || key == QStringLiteral("latency_ms"))
            continue;

        if (baseFields.contains(key)) {
            base.insert(baseFields.value(key), value);
            continue;
        }

        if (key.startsWith(QStringLiteral("decision_"))) {
            const QString normalized = camelize(QStringLiteral("decision_"), key);
            if (normalized.isEmpty())
                continue;
            if (normalized == QStringLiteral("shouldTrade"))
                decisionPayload.insert(normalized, normalizeBoolean(value));
            else
                decisionPayload.insert(normalized, value);
            continue;
        }

        if (key.startsWith(QStringLiteral("ai_"))) {
            const QString normalized = camelize(QStringLiteral("ai_"), key);
            if (!normalized.isEmpty())
                aiPayload.insert(normalized, value);
            continue;
        }

        if (key == QStringLiteral("market_regime")) {
            regimePayload.insert(QStringLiteral("regime"), value);
            continue;
        }

        if (key.startsWith(QStringLiteral("market_regime"))) {
            const QString normalized = camelize(QStringLiteral("market_regime"), key);
            if (!normalized.isEmpty())
                regimePayload.insert(normalized, value);
            continue;
        }

        if (key == QStringLiteral("risk_profile"))
            continue;

        extras.insert(key, value);
    }

    QVariantMap payload;
    payload.insert(QStringLiteral("event"), toString(base.value(QStringLiteral("event"))));
    payload.insert(QStringLiteral("timestamp"), toString(base.value(QStringLiteral("timestamp"))));
    payload.insert(QStringLiteral("environment"), toString(base.value(QStringLiteral("environment"))));
    payload.insert(QStringLiteral("portfolio"), toString(base.value(QStringLiteral("portfolio"))));
    payload.insert(QStringLiteral("riskProfile"), toString(base.value(QStringLiteral("risk_profile"))));

    const auto insertOptional = [&](const QString& targetKey, const QString& sourceKey) {
        const QVariant variant = base.value(sourceKey);
        if (!variant.isNull() && !variant.toString().isEmpty())
            payload.insert(targetKey, toString(variant));
    };

    insertOptional(QStringLiteral("schedule"), QStringLiteral("schedule"));
    insertOptional(QStringLiteral("strategy"), QStringLiteral("strategy"));
    insertOptional(QStringLiteral("symbol"), QStringLiteral("symbol"));
    insertOptional(QStringLiteral("side"), QStringLiteral("side"));
    insertOptional(QStringLiteral("status"), QStringLiteral("status"));
    insertOptional(QStringLiteral("quantity"), QStringLiteral("quantity"));
    insertOptional(QStringLiteral("price"), QStringLiteral("price"));

    payload.insert(QStringLiteral("marketRegime"), regimePayload);
    payload.insert(QStringLiteral("decision"), decisionPayload);
    payload.insert(QStringLiteral("ai"), aiPayload);
    payload.insert(QStringLiteral("metadata"), extras);

    return payload;
}

QStringList RuntimeDecisionBridge::resolveCandidateFiles() const
{
    QStringList files;
    if (m_logPath.trimmed().isEmpty())
        return files;

    const QString normalized = normalizePath(m_logPath);
    QFileInfo info(normalized);
    if (!info.exists())
        return files;

    if (info.isDir()) {
        QDir dir(info.absoluteFilePath());
        const QFileInfoList listing = dir.entryInfoList(QDir::Files | QDir::NoDotAndDotDot, QDir::Name);
        for (const QFileInfo& item : listing)
            files.append(item.absoluteFilePath());
    } else if (info.isFile()) {
        files.append(info.absoluteFilePath());
    }

    return files;
}

QVariant RuntimeDecisionBridge::normalizeBoolean(const QVariant& value)
{
    if (value.typeId() == QMetaType::Bool)
        return value;
    if (value.typeId() == QMetaType::QString) {
        const QString normalized = value.toString().trimmed().toLower();
        if (normalized == QStringLiteral("true") || normalized == QStringLiteral("yes") || normalized == QStringLiteral("1"))
            return true;
        if (normalized == QStringLiteral("false") || normalized == QStringLiteral("no") || normalized == QStringLiteral("0"))
            return false;
    }
    return value;
}

QString RuntimeDecisionBridge::camelize(const QString& prefix, const QString& key)
{
    QString suffix = key.mid(prefix.length());
    suffix = suffix.trimmed();
    if (suffix.startsWith(QLatin1Char('_')))
        suffix = suffix.mid(1);
    if (suffix.isEmpty())
        return {};

    const QStringList parts = suffix.split(QLatin1Char('_'), Qt::SkipEmptyParts);
    if (parts.isEmpty())
        return {};

    QString result = parts.first();
    for (int i = 1; i < parts.size(); ++i) {
        QString part = parts.at(i);
        if (!part.isEmpty()) {
            part[0] = part[0].toUpper();
            result.append(part);
        }
    }
    return result;
}

