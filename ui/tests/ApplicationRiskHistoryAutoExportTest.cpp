#include <QtTest/QtTest>
#include <QQmlApplicationEngine>
#include <QTemporaryDir>
#include <QDir>
#include <QFileInfo>
#include <algorithm>
#include <QDateTime>
#include <QUrl>

#include "app/Application.hpp"
#include "models/RiskHistoryModel.hpp"
#include "models/RiskTypes.hpp"

class ApplicationRiskHistoryAutoExportTest : public QObject {
    Q_OBJECT

private slots:
    void testAutoExportCreatesFiles();
    void testLocalTimeFileNameIncludesOffset();
};

void ApplicationRiskHistoryAutoExportTest::testAutoExportCreatesFiles()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());

    QQmlApplicationEngine engine;
    Application app(engine);

    auto* historyModel = qobject_cast<RiskHistoryModel*>(app.riskHistoryModel());
    QVERIFY(historyModel);

    QVERIFY(app.setRiskHistoryExportLastDirectory(QUrl::fromLocalFile(dir.path())));
    QVERIFY(app.setRiskHistoryExportLimitEnabled(true));
    QVERIFY(app.setRiskHistoryExportLimitValue(2));
    QVERIFY(app.setRiskHistoryAutoExportBasename(QStringLiteral("auto-risk")));
    QVERIFY(app.setRiskHistoryAutoExportIntervalMinutes(1));
    QVERIFY(app.setRiskHistoryAutoExportEnabled(true));

    RiskSnapshotData snapshot;
    snapshot.hasData = true;
    snapshot.profileLabel = QStringLiteral("Profil testowy");
    snapshot.currentDrawdown = 0.021;
    snapshot.usedLeverage = 1.4;
    snapshot.portfolioValue = 125000.0;
    snapshot.generatedAt = QDateTime::fromString(QStringLiteral("2024-05-01T10:00:00Z"), Qt::ISODate);
    historyModel->recordSnapshot(snapshot);

    QDir exportDir(dir.path());
    QStringList files = exportDir.entryList(QStringList() << QStringLiteral("*.csv"), QDir::Files, QDir::Name);
    QCOMPARE(files.size(), 1);
    const QString firstFile = files.first();
    QVERIFY(firstFile.startsWith(QStringLiteral("auto-risk_20240501_100000")));
    QCOMPARE(app.riskHistoryLastAutoExportPath().toLocalFile(), exportDir.absoluteFilePath(firstFile));
    QVERIFY(app.riskHistoryLastAutoExportAt().isValid());

    // Druga próbka w krótkim odstępie nie powinna wyzwolić kolejnego eksportu
    snapshot.generatedAt = QDateTime::fromString(QStringLiteral("2024-05-01T10:00:15Z"), Qt::ISODate);
    historyModel->recordSnapshot(snapshot);
    files = exportDir.entryList(QStringList() << QStringLiteral("*.csv"), QDir::Files, QDir::Name);
    QCOMPARE(files.size(), 1);

    // Cofnij zegar auto-eksportu i sprawdź, że powstaje nowy plik
    app.setLastRiskHistoryAutoExportForTesting(QDateTime::currentDateTimeUtc().addSecs(-120));
    snapshot.generatedAt = QDateTime::fromString(QStringLiteral("2024-05-01T10:05:00Z"), Qt::ISODate);
    historyModel->recordSnapshot(snapshot);

    files = exportDir.entryList(QStringList() << QStringLiteral("*.csv"), QDir::Files, QDir::Name);
    QCOMPARE(files.size(), 2);
    QVERIFY(std::any_of(files.cbegin(), files.cend(), [](const QString& name) {
        return name.contains(QStringLiteral("100500"));
    }));
}

void ApplicationRiskHistoryAutoExportTest::testLocalTimeFileNameIncludesOffset()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());

    QQmlApplicationEngine engine;
    Application app(engine);

    auto* historyModel = qobject_cast<RiskHistoryModel*>(app.riskHistoryModel());
    QVERIFY(historyModel);

    QVERIFY(app.setRiskHistoryExportLastDirectory(QUrl::fromLocalFile(dir.path())));
    QVERIFY(app.setRiskHistoryAutoExportEnabled(true));
    QVERIFY(app.setRiskHistoryAutoExportUseLocalTime(true));
    QVERIFY(app.setRiskHistoryAutoExportBasename(QStringLiteral("offset-test")));
    QVERIFY(app.setRiskHistoryAutoExportIntervalMinutes(1));

    RiskSnapshotData snapshot;
    snapshot.hasData = true;
    snapshot.generatedAt = QDateTime::fromString(QStringLiteral("2024-08-19T14:30:00Z"), Qt::ISODate);
    snapshot.currentDrawdown = 0.01;
    snapshot.usedLeverage = 1.0;
    snapshot.portfolioValue = 50000.0;
    historyModel->recordSnapshot(snapshot);

    QDir exportDir(dir.path());
    const QStringList files = exportDir.entryList(QStringList() << QStringLiteral("*.csv"), QDir::Files, QDir::Name);
    QCOMPARE(files.size(), 1);

    const QDateTime localTimestamp = snapshot.generatedAt.toLocalTime();
    const QString expectedTime = localTimestamp.toString(QStringLiteral("yyyyMMdd_HHmmss"));
    const int offsetSeconds = localTimestamp.offsetFromUtc();
    const int offsetMinutes = offsetSeconds / 60;
    const int offsetHours = offsetMinutes / 60;
    const int remainingMinutes = qAbs(offsetMinutes % 60);
    const QChar sign = offsetSeconds >= 0 ? QLatin1Char('+') : QLatin1Char('-');
    const QString offsetTag = QStringLiteral("%1%2%3")
                                  .arg(sign)
                                  .arg(qAbs(offsetHours), 2, 10, QLatin1Char('0'))
                                  .arg(remainingMinutes, 2, 10, QLatin1Char('0'));
    const QString expectedPrefix = QStringLiteral("offset-test_%1_%2").arg(expectedTime, offsetTag);
    QVERIFY(files.first().startsWith(expectedPrefix));
}

QTEST_MAIN(ApplicationRiskHistoryAutoExportTest)
#include "ApplicationRiskHistoryAutoExportTest.moc"

