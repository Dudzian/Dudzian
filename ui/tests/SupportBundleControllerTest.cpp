#include <QtTest/QtTest>
#include <QDir>
#include <QFile>
#include <QIODevice>
#include <QTemporaryDir>
#include <QSignalSpy>

#include "support/SupportBundleController.hpp"

class SupportBundleControllerTest : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();
    void test_export_bundle_success();
    void test_export_bundle_missing_script();
};

void SupportBundleControllerTest::initTestCase()
{
    QVERIFY2(QFile::exists(QDir::current().absoluteFilePath(QStringLiteral("scripts/export_support_bundle.py"))),
             "Nie znaleziono scripts/export_support_bundle.py w katalogu roboczym testów");
}

void SupportBundleControllerTest::test_export_bundle_success()
{
    SupportBundleController controller;
    controller.setPythonExecutable(QStringLiteral("python3"));
    controller.setScriptPath(QDir::current().absoluteFilePath(QStringLiteral("scripts/export_support_bundle.py")));

    QTemporaryDir tempDir;
    QVERIFY2(tempDir.isValid(), "Nie udało się utworzyć katalogu tymczasowego");

    QDir base(tempDir.path());
    QVERIFY(base.mkpath(QStringLiteral("logs")));
    QVERIFY(base.mkpath(QStringLiteral("var/reports")));
    QVERIFY(base.mkpath(QStringLiteral("var/licenses")));
    QVERIFY(base.mkpath(QStringLiteral("var/metrics")));

    QFile logFile(base.filePath(QStringLiteral("logs/runtime.log")));
    QVERIFY(logFile.open(QIODevice::WriteOnly | QIODevice::Text));
    logFile.write("log entry\n");
    logFile.close();

    controller.setLogsPath(base.filePath(QStringLiteral("logs")));
    controller.setReportsPath(base.filePath(QStringLiteral("var/reports")));
    controller.setLicensesPath(base.filePath(QStringLiteral("var/licenses")));
    controller.setMetricsPath(base.filePath(QStringLiteral("var/metrics")));
    controller.setIncludeAudit(false);
    controller.setOutputDirectory(base.filePath(QStringLiteral("out")));

    QSignalSpy spy(&controller, &SupportBundleController::exportFinished);
    QVERIFY(controller.exportBundle());
    QVERIFY2(spy.wait(15000), "Nie otrzymano sygnału exportFinished");

    const QList<QVariant> arguments = spy.takeFirst();
    QVERIFY(arguments.size() == 2);
    const bool success = arguments.at(0).toBool();
    const QVariantMap payload = arguments.at(1).toMap();
    QVERIFY2(success, "Eksport pakietu wsparcia zwrócił błąd");
    QCOMPARE(payload.value(QStringLiteral("status")).toString(), QStringLiteral("ok"));
    const QString bundlePath = payload.value(QStringLiteral("bundle_path")).toString();
    QVERIFY2(!bundlePath.isEmpty(), "Brak ścieżki pakietu w wynikach");
    QVERIFY2(QFile::exists(bundlePath), "Plik pakietu wsparcia nie istnieje");
}

void SupportBundleControllerTest::test_export_bundle_missing_script()
{
    SupportBundleController controller;
    controller.setPythonExecutable(QStringLiteral("python3"));
    controller.setScriptPath(QStringLiteral("/nonexistent/export_support_bundle.py"));

    QSignalSpy spy(&controller, &SupportBundleController::exportFinished);
    QVERIFY(!controller.exportBundle());
    QVERIFY(spy.isEmpty());
}

QTEST_MAIN(SupportBundleControllerTest)
#include "SupportBundleControllerTest.moc"
