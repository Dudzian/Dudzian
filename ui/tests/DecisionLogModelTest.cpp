#include <QDateTime>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QtTest/QtTest>

#include "models/DecisionLogModel.hpp"

namespace {

QByteArray makeEntry(const QString& timestamp,
                     const QString& event,
                     const QVariantMap& fields = {})
{
    QJsonObject object;
    object.insert(QStringLiteral("timestamp"), timestamp);
    object.insert(QStringLiteral("event"), event);

    for (auto it = fields.cbegin(); it != fields.cend(); ++it)
        object.insert(it.key(), QJsonValue::fromVariant(it.value()));

    return QJsonDocument(object).toJson(QJsonDocument::Compact);
}

} // namespace

class DecisionLogModelTest : public QObject {
    Q_OBJECT

private slots:
    void loadsEntriesFromDirectory();
    void trimsToMaximumEntries();
    void reloadsWhenFileChanges();
    void reloadsWhenNewFileAppears();
};

void DecisionLogModelTest::loadsEntriesFromDirectory()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());

    QFile file(dir.filePath(QStringLiteral("journal.jsonl")));
    QVERIFY(file.open(QIODevice::WriteOnly | QIODevice::Text));

    QVariantMap firstFields{{QStringLiteral("environment"), QStringLiteral("prod")},
                            {QStringLiteral("portfolio"), QStringLiteral("core")},
                            {QStringLiteral("risk_profile"), QStringLiteral("balanced")},
                            {QStringLiteral("schedule"), QStringLiteral("daily")},
                            {QStringLiteral("strategy"), QStringLiteral("alpha")},
                            {QStringLiteral("symbol"), QStringLiteral("BTCUSDT")},
                            {QStringLiteral("side"), QStringLiteral("buy")},
                            {QStringLiteral("quantity"), 1.5},
                            {QStringLiteral("price"), 27000.125},
                            {QStringLiteral("approved"), true},
                            {QStringLiteral("decision_state"), QStringLiteral("approved")},
                            {QStringLiteral("decision_reason"), QStringLiteral("risk_ok")},
                            {QStringLiteral("decision_mode"), QStringLiteral("auto")},
                            {QStringLiteral("telemetry_namespace"), QStringLiteral("prod.alpha")}};

    file.write(makeEntry(QStringLiteral("2024-06-01T10:00:00Z"), QStringLiteral("strategy_decision"), firstFields));
    file.write("\n");

    QVariantMap secondFields = firstFields;
    secondFields.insert(QStringLiteral("approved"), QStringLiteral("false"));
    secondFields.insert(QStringLiteral("decision_state"), QStringLiteral("rejected"));
    secondFields.insert(QStringLiteral("decision_reason"), QStringLiteral("risk_limit"));
    secondFields.insert(QStringLiteral("strategy"), QStringLiteral("beta"));
    secondFields.insert(QStringLiteral("symbol"), QStringLiteral("ETHUSDT"));
    secondFields.insert(QStringLiteral("side"), QStringLiteral("sell"));
    secondFields.insert(QStringLiteral("quantity"), 2.0);
    secondFields.insert(QStringLiteral("price"), 1835.50);

    file.write(makeEntry(QStringLiteral("2024-06-01T10:01:00Z"), QStringLiteral("strategy_decision"), secondFields));
    file.write("\n");
    file.close();

    DecisionLogModel model;
    model.setLogPath(dir.path());
    QVERIFY(model.reload());

    QCOMPARE(model.rowCount(), 2);

    const QModelIndex firstIndex = model.index(0, 0);
    QCOMPARE(model.data(firstIndex, DecisionLogModel::EventRole).toString(), QStringLiteral("strategy_decision"));
    QCOMPARE(model.data(firstIndex, DecisionLogModel::PortfolioRole).toString(), QStringLiteral("core"));
    QCOMPARE(model.data(firstIndex, DecisionLogModel::QuantityRole).toString(), QStringLiteral("1.5"));
    QCOMPARE(model.data(firstIndex, DecisionLogModel::ApprovedRole).toBool(), true);
    QCOMPARE(model.data(firstIndex, DecisionLogModel::TimestampRole).toDateTime(),
             QDateTime::fromString(QStringLiteral("2024-06-01T10:00:00Z"), Qt::ISODate).toUTC());

    const QModelIndex secondIndex = model.index(1, 0);
    QCOMPARE(model.data(secondIndex, DecisionLogModel::StrategyRole).toString(), QStringLiteral("beta"));
    QCOMPARE(model.data(secondIndex, DecisionLogModel::ApprovedRole).toBool(), false);
    QCOMPARE(model.data(secondIndex, DecisionLogModel::DecisionReasonRole).toString(), QStringLiteral("risk_limit"));
    QCOMPARE(model.data(secondIndex, DecisionLogModel::SymbolRole).toString(), QStringLiteral("ETHUSDT"));
}

void DecisionLogModelTest::trimsToMaximumEntries()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());

    QFile file(dir.filePath(QStringLiteral("trim.jsonl")));
    QVERIFY(file.open(QIODevice::WriteOnly | QIODevice::Text));

    for (int i = 0; i < 4; ++i) {
        const QString timestamp = QStringLiteral("2024-06-01T10:%1:00Z").arg(i, 2, 10, QLatin1Char('0'));
        QVariantMap fields{{QStringLiteral("strategy"), QStringLiteral("S%1").arg(i)}};
        file.write(makeEntry(timestamp, QStringLiteral("decision"), fields));
        file.write("\n");
    }
    file.close();

    DecisionLogModel model;
    model.setMaximumEntries(2);
    model.setLogPath(file.fileName());
    QVERIFY(model.reload());

    QCOMPARE(model.rowCount(), 2);
    const QModelIndex firstIndex = model.index(0, 0);
    QCOMPARE(model.data(firstIndex, DecisionLogModel::StrategyRole).toString(), QStringLiteral("S2"));
    const QModelIndex secondIndex = model.index(1, 0);
    QCOMPARE(model.data(secondIndex, DecisionLogModel::StrategyRole).toString(), QStringLiteral("S3"));
}

void DecisionLogModelTest::reloadsWhenFileChanges()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());

    const QString filePath = dir.filePath(QStringLiteral("watch.jsonl"));
    QFile file(filePath);
    QVERIFY(file.open(QIODevice::WriteOnly | QIODevice::Text));
    file.write(makeEntry(QStringLiteral("2024-06-01T10:00:00Z"), QStringLiteral("initial")));
    file.write("\n");
    file.close();

    DecisionLogModel model;
    model.setLogPath(filePath);
    QVERIFY(model.reload());
    QCOMPARE(model.rowCount(), 1);

    QSignalSpy countSpy(&model, &DecisionLogModel::countChanged);

    QVERIFY(file.open(QIODevice::Append | QIODevice::Text));
    file.write(makeEntry(QStringLiteral("2024-06-01T10:01:00Z"), QStringLiteral("update")));
    file.write("\n");
    file.flush();
    file.close();

    QTRY_VERIFY_WITH_TIMEOUT(countSpy.count() > 0, 2000);
    QTRY_COMPARE_WITH_TIMEOUT(model.rowCount(), 2, 2000);
}

void DecisionLogModelTest::reloadsWhenNewFileAppears()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());

    DecisionLogModel model;
    model.setLogPath(dir.path());
    QVERIFY(model.reload());
    QCOMPARE(model.rowCount(), 0);

    QSignalSpy countSpy(&model, &DecisionLogModel::countChanged);

    QFile file(dir.filePath(QStringLiteral("new.jsonl")));
    QVERIFY(file.open(QIODevice::WriteOnly | QIODevice::Text));
    file.write(makeEntry(QStringLiteral("2024-06-01T10:02:00Z"), QStringLiteral("directory")));
    file.write("\n");
    file.close();

    QTRY_VERIFY_WITH_TIMEOUT(countSpy.count() > 0, 2000);
    QTRY_COMPARE_WITH_TIMEOUT(model.rowCount(), 1, 2000);
}

QTEST_MAIN(DecisionLogModelTest)
#include "DecisionLogModelTest.moc"
