#include <QtTest/QSignalSpy>
#include <QtTest/QTest>
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QIODevice>
#include <QTemporaryDir>

#include "reporting/ReportCenterController.hpp"

class ReportCenterControllerTest : public QObject
{
    Q_OBJECT

private Q_SLOTS:
    void refreshesWhenExportCreated();
};

void ReportCenterControllerTest::refreshesWhenExportCreated()
{
    QTemporaryDir tempDir;
    QVERIFY(tempDir.isValid());

    ReportCenterController controller;
    controller.setReportsRoot(tempDir.path());

    QCOMPARE(controller.exports().size(), 0);

    const QString reportDirPath = tempDir.path() + QLatin1String("/daily");
    QVERIFY(QDir().mkpath(reportDirPath));

    QTRY_VERIFY_WITH_TIMEOUT(controller.watchedDirectories().contains(QDir(reportDirPath).absolutePath()), 3000);

    QSignalSpy exportsSpy(&controller, &ReportCenterController::exportsChanged);
    const int initialCount = exportsSpy.count();

    const QString exportFilePath = reportDirPath + QLatin1String("/orders.csv");
    QFile file(exportFilePath);
    QVERIFY(file.open(QIODevice::WriteOnly | QIODevice::Text));
    file.write("id,value\n1,42\n");
    file.close();

    QVERIFY(exportsSpy.wait(3000));
    QVERIFY(exportsSpy.count() > initialCount);

    const QStringList exports = controller.exports();
    QVERIFY(exports.contains(QDir(exportFilePath).absolutePath()));
}

QTEST_MAIN(ReportCenterControllerTest)

#include "ReportCenterControllerTest.moc"

