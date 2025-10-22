#include "DecisionLogModel.hpp"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLoggingCategory>
#include <QRegularExpression>
#include <QSet>

Q_LOGGING_CATEGORY(lcDecisionLog, "bot.shell.decision.log")

namespace {
QString cleanPath(const QString& path)
{
    QFileInfo info(path);
    if (!info.exists())
        return QDir::cleanPath(info.absoluteFilePath());
    return QDir::cleanPath(info.absoluteFilePath());
}

QStringList sortedFileList(const QFileInfo& info)
{
    QStringList files;
    if (info.isDir()) {
        QDir dir(info.absoluteFilePath());
        const QFileInfoList listing = dir.entryInfoList(QDir::Files | QDir::NoDotAndDotDot, QDir::Name);
        for (const QFileInfo& item : listing)
            files.append(item.absoluteFilePath());
    } else if (info.exists()) {
        files.append(info.absoluteFilePath());
    }
    return files;
}

QString formatQuantity(const QVariant& value)
{
    if (value.canConvert<double>()) {
        bool ok = false;
        const double number = value.toDouble(&ok);
        if (ok)
            return QString::number(number, 'f', number == 0.0 ? 0 : 8).remove(QRegularExpression("0+$")).remove(QRegularExpression("\\.$"));
    }
    return value.toString();
}

} // namespace

DecisionLogModel::DecisionLogModel(QObject* parent)
    : QAbstractListModel(parent)
{
    m_reloadDebounce.setInterval(200);
    m_reloadDebounce.setSingleShot(true);
    connect(&m_reloadDebounce, &QTimer::timeout, this, &DecisionLogModel::performReload);
    connect(&m_watcher, &QFileSystemWatcher::fileChanged, this, &DecisionLogModel::handleWatchedPathChanged);
    connect(&m_watcher, &QFileSystemWatcher::directoryChanged, this, &DecisionLogModel::handleWatchedPathChanged);
}

int DecisionLogModel::rowCount(const QModelIndex& parent) const
{
    if (parent.isValid())
        return 0;
    return m_entries.size();
}

QVariant DecisionLogModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid() || index.row() < 0 || index.row() >= m_entries.size())
        return {};

    const Entry& entry = m_entries.at(index.row());
    switch (role) {
    case TimestampRole:
        return entry.timestampUtc;
    case TimestampDisplayRole:
        if (entry.timestampUtc.isValid())
            return entry.timestampUtc.toLocalTime().toString(Qt::ISODateWithMs);
        return QString();
    case EventRole:
        return entry.event;
    case EnvironmentRole:
        return entry.environment;
    case PortfolioRole:
        return entry.portfolio;
    case RiskProfileRole:
        return entry.riskProfile;
    case ScheduleRole:
        return entry.schedule;
    case StrategyRole:
        return entry.strategy;
    case SymbolRole:
        return entry.symbol;
    case SideRole:
        return entry.side;
    case QuantityRole:
        return entry.quantity;
    case PriceRole:
        return entry.price;
    case ApprovedRole:
        return entry.approved;
    case DecisionStateRole:
        return entry.decisionState;
    case DecisionReasonRole:
        return entry.decisionReason;
    case DecisionModeRole:
        return entry.decisionMode;
    case TelemetryNamespaceRole:
        return entry.telemetryNamespace;
    case DetailsRole:
        return entry.payload;
    default:
        break;
    }
    return {};
}

QHash<int, QByteArray> DecisionLogModel::roleNames() const
{
    QHash<int, QByteArray> roles;
    roles[TimestampRole] = "timestamp";
    roles[TimestampDisplayRole] = "timestampDisplay";
    roles[EventRole] = "event";
    roles[EnvironmentRole] = "environment";
    roles[PortfolioRole] = "portfolio";
    roles[RiskProfileRole] = "riskProfile";
    roles[ScheduleRole] = "schedule";
    roles[StrategyRole] = "strategy";
    roles[SymbolRole] = "symbol";
    roles[SideRole] = "side";
    roles[QuantityRole] = "quantity";
    roles[PriceRole] = "price";
    roles[ApprovedRole] = "approved";
    roles[DecisionStateRole] = "decisionState";
    roles[DecisionReasonRole] = "decisionReason";
    roles[DecisionModeRole] = "decisionMode";
    roles[TelemetryNamespaceRole] = "telemetryNamespace";
    roles[DetailsRole] = "details";
    return roles;
}

void DecisionLogModel::setMaximumEntries(int value)
{
    const int normalized = value <= 0 ? 1 : value;
    if (m_maxEntries == normalized)
        return;
    m_maxEntries = normalized;
    emit maximumEntriesChanged();
    reload();
}

void DecisionLogModel::setLogPath(const QString& path)
{
    const QString normalized = cleanPath(path);
    if (m_logPath == normalized)
        return;
    m_logPath = normalized;
    emit logPathChanged();
    reload();
}

bool DecisionLogModel::reload()
{
    QVector<Entry> entries;
    QStringList watchedFiles;
    if (!readEntries(entries, watchedFiles))
        return false;

    if (m_maxEntries > 0 && entries.size() > m_maxEntries)
        entries = entries.mid(entries.size() - m_maxEntries, m_maxEntries);

    const int previousCount = m_entries.size();

    beginResetModel();
    m_entries = std::move(entries);
    endResetModel();

    if (previousCount != m_entries.size())
        emit countChanged();

    updateWatchers(watchedFiles);
    return true;
}

void DecisionLogModel::handleWatchedPathChanged(const QString& path)
{
    Q_UNUSED(path);
    scheduleReload();
}

void DecisionLogModel::performReload()
{
    reload();
}

void DecisionLogModel::scheduleReload()
{
    if (!m_reloadDebounce.isActive())
        m_reloadDebounce.start();
}

bool DecisionLogModel::readEntries(QVector<Entry>& entries, QStringList& watchedFiles) const
{
    entries.clear();
    watchedFiles.clear();

    if (m_logPath.trimmed().isEmpty())
        return true;

    const QFileInfo pathInfo(m_logPath);
    const QStringList files = sortedFileList(pathInfo);
    for (const QString& filePath : files) {
        QFile file(filePath);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            qCWarning(lcDecisionLog) << "Nie udało się otworzyć decision logu" << filePath << file.errorString();
            continue;
        }
        watchedFiles.append(filePath);
        while (!file.atEnd()) {
            const QByteArray line = file.readLine();
            const QByteArray trimmed = line.trimmed();
            if (trimmed.isEmpty())
                continue;
            QJsonParseError error{};
            const QJsonDocument doc = QJsonDocument::fromJson(trimmed, &error);
            if (error.error != QJsonParseError::NoError || !doc.isObject()) {
                qCWarning(lcDecisionLog) << "Niepoprawny wpis decision logu" << filePath << error.errorString();
                continue;
            }
            const QVariantMap payload = doc.object().toVariantMap();
            entries.append(buildEntry(payload));
        }
    }
    return true;
}

DecisionLogModel::Entry DecisionLogModel::buildEntry(const QVariantMap& payload) const
{
    Entry entry;
    entry.payload = payload;

    const QString timestampText = payload.value(QStringLiteral("timestamp")).toString();
    entry.timestampUtc = parseTimestamp(timestampText);
    entry.event = payload.value(QStringLiteral("event")).toString();
    entry.environment = payload.value(QStringLiteral("environment")).toString();
    entry.portfolio = payload.value(QStringLiteral("portfolio")).toString();
    entry.riskProfile = payload.value(QStringLiteral("risk_profile")).toString();
    entry.schedule = payload.value(QStringLiteral("schedule")).toString();
    entry.strategy = payload.value(QStringLiteral("strategy")).toString();
    entry.symbol = payload.value(QStringLiteral("symbol")).toString();
    entry.side = payload.value(QStringLiteral("side")).toString();
    entry.quantity = formatQuantity(payload.value(QStringLiteral("quantity")));
    entry.price = formatQuantity(payload.value(QStringLiteral("price")));
    entry.approved = coerceBool(payload.value(QStringLiteral("approved")));
    entry.decisionState = payload.value(QStringLiteral("decision_state")).toString();
    entry.decisionReason = payload.value(QStringLiteral("decision_reason")).toString();
    entry.decisionMode = payload.value(QStringLiteral("decision_mode")).toString();
    entry.telemetryNamespace = payload.value(QStringLiteral("telemetry_namespace")).toString();
    return entry;
}

QDateTime DecisionLogModel::parseTimestamp(const QString& text) const
{
    if (text.isEmpty())
        return {};

    QDateTime parsed = QDateTime::fromString(text, Qt::ISODateWithMs);
    if (!parsed.isValid())
        parsed = QDateTime::fromString(text, Qt::ISODate);
    if (!parsed.isValid())
        return {};
    if (parsed.timeSpec() == Qt::LocalTime)
        parsed = parsed.toUTC();
    else if (parsed.timeSpec() == Qt::UTC)
        parsed = parsed.toUTC();
    else
        parsed = parsed.toUTC();
    return parsed;
}

bool DecisionLogModel::coerceBool(const QVariant& value) const
{
    switch (value.typeId()) {
    case QMetaType::Bool:
        return value.toBool();
    case QMetaType::Double:
    case QMetaType::Float:
    case QMetaType::Int:
    case QMetaType::LongLong:
    case QMetaType::UInt:
    case QMetaType::ULongLong:
        return value.toDouble() != 0.0;
    case QMetaType::QString: {
        const QString text = value.toString().trimmed().toLower();
        if (text.isEmpty())
            return false;
        return text == QStringLiteral("1") || text == QStringLiteral("true") || text == QStringLiteral("yes")
            || text == QStringLiteral("approved");
    }
    default:
        break;
    }
    return false;
}

void DecisionLogModel::updateWatchers(const QStringList& files)
{
    QStringList tracked;
    tracked.append(m_watcher.files());
    tracked.append(m_watcher.directories());
    if (!tracked.isEmpty())
        m_watcher.removePaths(tracked);

    if (m_logPath.isEmpty())
        return;

    const QFileInfo info(m_logPath);
    if (info.isDir() && info.exists())
        m_watcher.addPath(info.absoluteFilePath());
    else if (info.isFile() && info.exists())
        m_watcher.addPath(info.absoluteFilePath());

    QSet<QString> uniqueFiles;
    for (const QString& file : files) {
        if (file.isEmpty())
            continue;
        QFileInfo fileInfo(file);
        if (!fileInfo.exists())
            continue;
        const QString absolute = fileInfo.absoluteFilePath();
        if (uniqueFiles.contains(absolute))
            continue;
        uniqueFiles.insert(absolute);
        m_watcher.addPath(absolute);
    }
}

