#include "runtime/OfflineRuntimeBridge.hpp"

#include <QDateTime>
#include <QJsonParseError>
#include <QLoggingCategory>
#include <QNetworkReply>
#include <QPointer>
#include <QScopedPointer>
#include <QUrlQuery>
#include <QtGlobal>

Q_LOGGING_CATEGORY(lcOfflineBridge, "bot.shell.offline.bridge")

namespace {
constexpr int kDefaultPollIntervalMs = 5000;
}

OfflineRuntimeBridge::OfflineRuntimeBridge(QObject* parent)
    : QObject(parent)
{
    m_pollTimer.setInterval(kDefaultPollIntervalMs);
    m_pollTimer.setSingleShot(false);
    connect(&m_pollTimer, &QTimer::timeout, this, &OfflineRuntimeBridge::handlePollTick);
}

OfflineRuntimeBridge::~OfflineRuntimeBridge() = default;

void OfflineRuntimeBridge::setEndpoint(const QUrl& endpoint)
{
    QUrl normalized = endpoint;
    if (normalized.scheme().isEmpty())
        normalized.setScheme(QStringLiteral("http"));
    if (normalized.path().isEmpty())
        normalized.setPath(QStringLiteral("/"));
    m_endpoint = normalized;
}

void OfflineRuntimeBridge::setInstrument(const TradingClient::InstrumentConfig& config)
{
    m_instrument = config;
}

void OfflineRuntimeBridge::setHistoryLimit(int limit)
{
    m_historyLimit = qMax(1, limit);
}

void OfflineRuntimeBridge::setAutoRunEnabled(bool enabled)
{
    m_autoRunEnabled = enabled;
    if (m_running && enabled && !m_automationRunning)
        startAutomation();
}

void OfflineRuntimeBridge::setStrategyConfig(const QVariantMap& config)
{
    m_strategyConfig = config;
    if (m_running)
        pushStrategyConfig();
}

void OfflineRuntimeBridge::start()
{
    if (m_running)
        return;
    if (!m_endpoint.isValid()) {
        qCWarning(lcOfflineBridge) << "Offline endpoint nieustawiony";
        return;
    }
    m_running = true;
    applyConnectionState(tr("Offline daemon: łączenie…"));
    fetchStatus();
    fetchPerformanceGuard();
    fetchHistory();
    fetchRisk();
    pushStrategyConfig();
    if (m_autoRunEnabled)
        startAutomation();
    m_pollTimer.start();
}

void OfflineRuntimeBridge::stop()
{
    if (!m_running)
        return;
    m_running = false;
    m_pollTimer.stop();
    applyConnectionState(tr("Offline daemon: zatrzymano"));
}

void OfflineRuntimeBridge::refreshRiskNow()
{
    if (!m_running)
        return;
    fetchRisk();
}

void OfflineRuntimeBridge::startAutomation()
{
    if (!m_running)
        return;
    QJsonObject payload;
    payload.insert(QStringLiteral("auto"), m_autoRunEnabled);
    postJson(QStringLiteral("/v1/automation/start"), payload, [this](const QJsonObject& object) {
        handleAutomationPayload(object.value(QStringLiteral("automation")).toObject());
    });
}

void OfflineRuntimeBridge::stopAutomation()
{
    if (!m_running)
        return;
    postJson(QStringLiteral("/v1/automation/stop"), QJsonObject{}, [this](const QJsonObject& object) {
        handleAutomationPayload(object.value(QStringLiteral("automation")).toObject());
    });
}

void OfflineRuntimeBridge::handlePollTick()
{
    if (!m_running)
        return;
    fetchStatus();
    fetchRisk();
}

void OfflineRuntimeBridge::fetchStatus()
{
    getJson(QStringLiteral("/v1/status"), QUrlQuery(), [this](const QJsonObject& object) {
        applyConnectionState(tr("Offline daemon: gotowy"));
        handleAutomationPayload(object.value(QStringLiteral("automation")).toObject());
    });
}

void OfflineRuntimeBridge::fetchHistory()
{
    QUrlQuery query;
    if (!m_instrument.exchange.isEmpty())
        query.addQueryItem(QStringLiteral("exchange"), m_instrument.exchange);
    if (!m_instrument.symbol.isEmpty())
        query.addQueryItem(QStringLiteral("symbol"), m_instrument.symbol);
    if (!m_instrument.granularityIso8601.isEmpty())
        query.addQueryItem(QStringLiteral("granularity"), m_instrument.granularityIso8601);
    query.addQueryItem(QStringLiteral("limit"), QString::number(m_historyLimit));
    getJson(QStringLiteral("/v1/market-data/history"), query, [this](const QJsonObject& object) {
        const QJsonArray candles = object.value(QStringLiteral("candles")).toArray();
        QList<OhlcvPoint> series = parseCandles(candles);
        if (!series.isEmpty())
            emit historyReceived(series);
    });
}

void OfflineRuntimeBridge::fetchRisk()
{
    getJson(QStringLiteral("/v1/risk/state"), QUrlQuery(), [this](const QJsonObject& object) {
        const QJsonObject snapshot = object.value(QStringLiteral("snapshot")).toObject();
        if (snapshot.isEmpty())
            return;
        RiskSnapshotData risk = parseRisk(snapshot);
        emit riskStateReceived(risk);
    });
}

void OfflineRuntimeBridge::fetchPerformanceGuard()
{
    getJson(QStringLiteral("/v1/performance-guard"), QUrlQuery(), [this](const QJsonObject& object) {
        const QJsonObject guardObj = object.value(QStringLiteral("guard")).toObject();
        if (guardObj.isEmpty())
            return;
        PerformanceGuard guard = parseGuard(guardObj);
        emit performanceGuardUpdated(guard);
    });
}

void OfflineRuntimeBridge::pushStrategyConfig()
{
    if (m_strategyConfig.isEmpty())
        return;
    QJsonObject payload = QJsonObject::fromVariantMap(m_strategyConfig);
    postJson(QStringLiteral("/v1/strategy"), payload, [](const QJsonObject&) {});
}

void OfflineRuntimeBridge::applyConnectionState(const QString& state)
{
    if (m_connectionState == state)
        return;
    m_connectionState = state;
    emit connectionStateChanged(state);
}

void OfflineRuntimeBridge::handleAutomationPayload(const QJsonObject& object)
{
    const bool running = object.value(QStringLiteral("running")).toBool(false);
    if (m_automationRunning == running)
        return;
    m_automationRunning = running;
    emit automationStateChanged(running);
}

QList<OhlcvPoint> OfflineRuntimeBridge::parseCandles(const QJsonArray& array) const
{
    QList<OhlcvPoint> result;
    result.reserve(array.size());
    for (const QJsonValue& value : array) {
        const QJsonObject obj = value.toObject();
        if (obj.isEmpty())
            continue;
        OhlcvPoint point;
        point.timestampMs = static_cast<qint64>(obj.value(QStringLiteral("timestamp_ms")).toVariant().toLongLong());
        point.open = obj.value(QStringLiteral("open")).toDouble();
        point.high = obj.value(QStringLiteral("high")).toDouble();
        point.low = obj.value(QStringLiteral("low")).toDouble();
        point.close = obj.value(QStringLiteral("close")).toDouble();
        point.volume = obj.value(QStringLiteral("volume")).toDouble();
        point.closed = obj.value(QStringLiteral("closed")).toBool(true);
        point.sequence = static_cast<quint64>(obj.value(QStringLiteral("sequence")).toVariant().toULongLong());
        result.append(point);
    }
    return result;
}

RiskSnapshotData OfflineRuntimeBridge::parseRisk(const QJsonObject& object) const
{
    RiskSnapshotData data;
    data.hasData = true;
    data.portfolioValue = object.value(QStringLiteral("portfolio_value")).toDouble();
    data.currentDrawdown = object.value(QStringLiteral("current_drawdown")).toDouble();
    data.maxDailyLoss = object.value(QStringLiteral("max_daily_loss")).toDouble();
    data.usedLeverage = object.value(QStringLiteral("used_leverage")).toDouble();
    const qint64 generatedMs = static_cast<qint64>(object.value(QStringLiteral("generated_at_ms")).toVariant().toLongLong());
    if (generatedMs > 0)
        data.generatedAt = QDateTime::fromMSecsSinceEpoch(generatedMs, Qt::UTC);
    const QJsonValue profileValue = object.value(QStringLiteral("profile"));
    if (profileValue.isString()) {
        data.profileLabel = profileValue.toString();
        data.profileEnum = 0;
    } else {
        data.profileEnum = profileValue.toInt();
        data.profileLabel = tr("Profil %1").arg(data.profileEnum);
    }
    const QJsonArray exposures = object.value(QStringLiteral("exposures")).toArray();
    for (const QJsonValue& entryValue : exposures) {
        const QJsonObject entry = entryValue.toObject();
        if (entry.isEmpty())
            continue;
        RiskExposureData exposure;
        exposure.code = entry.value(QStringLiteral("code")).toString();
        exposure.maxValue = entry.value(QStringLiteral("max_value")).toDouble();
        exposure.currentValue = entry.value(QStringLiteral("current_value")).toDouble();
        exposure.thresholdValue = entry.value(QStringLiteral("threshold_value")).toDouble();
        data.exposures.append(exposure);
    }
    return data;
}

PerformanceGuard OfflineRuntimeBridge::parseGuard(const QJsonObject& object) const
{
    QVariantMap map = object.toVariantMap();
    return performanceGuardFromMap(map);
}

void OfflineRuntimeBridge::postJson(const QString& path,
                                    const QJsonObject& payload,
                                    std::function<void(const QJsonObject&)> onSuccess)
{
    if (!m_running)
        return;
    QNetworkRequest request(buildUrl(path));
    request.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/json"));
    const QByteArray body = QJsonDocument(payload).toJson(QJsonDocument::Compact);
    QNetworkReply* reply = m_network.post(request, body);
    connect(reply, &QNetworkReply::finished, this, [this, reply, onSuccess]() {
        QScopedPointer<QNetworkReply, QScopedPointerDeleteLater> guard(reply);
        if (!m_running)
            return;
        if (reply->error() != QNetworkReply::NoError) {
            qCWarning(lcOfflineBridge) << "POST" << reply->url() << "failed:" << reply->errorString();
            applyConnectionState(tr("Offline daemon: błąd"));
            return;
        }
        const QByteArray payload = reply->readAll();
        QJsonParseError err{};
        const QJsonDocument doc = QJsonDocument::fromJson(payload, &err);
        if (err.error != QJsonParseError::NoError) {
            qCWarning(lcOfflineBridge) << "Niepoprawny JSON w odpowiedzi POST" << err.errorString();
            return;
        }
        if (onSuccess)
            onSuccess(doc.object());
    });
}

void OfflineRuntimeBridge::getJson(const QString& path,
                                   const QUrlQuery& query,
                                   std::function<void(const QJsonObject&)> onSuccess)
{
    if (!m_running)
        return;
    QNetworkRequest request(buildUrl(path, query));
    requestJson(request, [this, onSuccess, path](const QJsonDocument& doc) {
        if (!doc.isObject()) {
            qCWarning(lcOfflineBridge) << "Niepoprawna odpowiedź JSON" << path;
            return;
        }
        if (onSuccess)
            onSuccess(doc.object());
    });
}

void OfflineRuntimeBridge::requestJson(const QNetworkRequest& request,
                                       std::function<void(const QJsonDocument&)> callback)
{
    QNetworkReply* reply = m_network.get(request);
    connect(reply, &QNetworkReply::finished, this, [this, reply, callback]() {
        QScopedPointer<QNetworkReply, QScopedPointerDeleteLater> guard(reply);
        if (!m_running)
            return;
        if (reply->error() != QNetworkReply::NoError) {
            qCWarning(lcOfflineBridge) << "GET" << reply->url() << "failed:" << reply->errorString();
            applyConnectionState(tr("Offline daemon: niedostępny"));
            return;
        }
        const QByteArray body = reply->readAll();
        QJsonParseError err{};
        const QJsonDocument doc = QJsonDocument::fromJson(body, &err);
        if (err.error != QJsonParseError::NoError) {
            qCWarning(lcOfflineBridge) << "Niepoprawny JSON" << err.errorString();
            return;
        }
        if (callback)
            callback(doc);
    });
}

QUrl OfflineRuntimeBridge::buildUrl(const QString& path, const QUrlQuery& query) const
{
    QUrl url = m_endpoint;
    QString basePath = url.path();
    if (!basePath.endsWith('/'))
        basePath.append('/');
    QString trimmed = path;
    if (trimmed.startsWith('/'))
        trimmed.remove(0, 1);
    url.setPath(basePath + trimmed);
    url.setQuery(query);
    return url;
}
