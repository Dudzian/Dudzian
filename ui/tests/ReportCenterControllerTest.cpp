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

    const QString reportDirPath = QDir(tempDir.path()).filePath(QStringLiteral("daily"));
    QVERIFY(QDir().mkpath(reportDirPath));

    const QString seedPath = QDir(reportDirPath).filePath(QStringLiteral("seed.csv"));
    {
        QFile seed(seedPath);
        QVERIFY(seed.open(QIODevice::WriteOnly | QIODevice::Text));
        seed.write("timestamp,foo\n");
        seed.close();
    }

    ReportCenterController controller;
    controller.setReportsDirectory(QDir(tempDir.path()).absolutePath());

    QSignalSpy overviewSpy(&controller, &ReportCenterController::overviewReady);
    const auto takeLastOverviewResult = [&overviewSpy]() {
        QVERIFY(!overviewSpy.isEmpty());
        const auto signals = overviewSpy.takeFirst();
        bool result = signals.at(0).toBool();
        while (!overviewSpy.isEmpty())
            result = overviewSpy.takeFirst().at(0).toBool();
        return result;
    };

    QVERIFY(controller.refresh());
    QVERIFY(overviewSpy.wait(10000));
    QVERIFY2(takeLastOverviewResult(), "Initial overview refresh failed");

    const QString exportPath = QDir(reportDirPath).filePath(QStringLiteral("orders.csv"));
    {
        QFile exportFile(exportPath);
        QVERIFY(exportFile.open(QIODevice::WriteOnly | QIODevice::Text));
        exportFile.write("timestamp,foo\n");
        exportFile.close();
    }

    QVERIFY(controller.refresh());
    QVERIFY(overviewSpy.wait(10000));
    QVERIFY2(takeLastOverviewResult(), "Second overview refresh failed");

    QStringList exportPaths;
    const QVariantList reports = controller.reports();
    for (const QVariant& reportVar : reports) {
        const QVariantMap report = reportVar.toMap();
        const QVariantList exports = report.value(QStringLiteral("exports")).toList();
        for (const QVariant& exportVar : exports) {
            const QVariantMap exp = exportVar.toMap();
            const QString abs = exp.value(QStringLiteral("absolute_path")).toString();
            if (!abs.isEmpty())
                exportPaths.append(abs);
        }
    }

    bool found = false;
    for (const QString& path : exportPaths) {
        if (path.endsWith(QStringLiteral("/orders.csv")) || path.endsWith(QStringLiteral("\\orders.csv"))) {
            found = true;
            break;
        }
    }

    QVERIFY2(found,
             qPrintable(QStringLiteral("orders.csv not found in overview exports; got: %1")
                            .arg(exportPaths.join(QStringLiteral(", ")))));
}

QTEST_MAIN(ReportCenterControllerTest)

#include "ReportCenterControllerTest.moc"
