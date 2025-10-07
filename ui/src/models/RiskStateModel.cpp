#include "RiskStateModel.hpp"

#include <QtGlobal>

RiskStateModel::RiskStateModel(QObject* parent)
    : QAbstractListModel(parent) {
    qRegisterMetaType<RiskSnapshotData>("RiskSnapshotData");
    qRegisterMetaType<RiskExposureData>("RiskExposureData");
}

int RiskStateModel::rowCount(const QModelIndex& parent) const {
    if (parent.isValid()) {
        return 0;
    }
    return m_snapshot.exposures.size();
}

QVariant RiskStateModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid()) {
        return {};
    }
    if (index.row() < 0 || index.row() >= m_snapshot.exposures.size()) {
        return {};
    }
    const auto& exposure = m_snapshot.exposures.at(index.row());
    switch (role) {
    case CodeRole:
        return exposure.code;
    case CurrentValueRole:
        return exposure.currentValue;
    case MaxValueRole:
        return exposure.maxValue;
    case ThresholdValueRole:
        return exposure.thresholdValue;
    case BreachRole:
        return exposure.isBreached();
    default:
        return {};
    }
}

QHash<int, QByteArray> RiskStateModel::roleNames() const {
    return {
        {CodeRole, QByteArrayLiteral("code")},
        {CurrentValueRole, QByteArrayLiteral("currentValue")},
        {MaxValueRole, QByteArrayLiteral("maxValue")},
        {ThresholdValueRole, QByteArrayLiteral("thresholdValue")},
        {BreachRole, QByteArrayLiteral("breached")},
    };
}

void RiskStateModel::updateFromSnapshot(const RiskSnapshotData& snapshot) {
    beginResetModel();
    m_snapshot = snapshot;
    m_snapshot.hasData = true;
    endResetModel();
    Q_EMIT riskStateChanged();
}

void RiskStateModel::clear() {
    beginResetModel();
    m_snapshot = RiskSnapshotData{};
    endResetModel();
    Q_EMIT riskStateChanged();
}
