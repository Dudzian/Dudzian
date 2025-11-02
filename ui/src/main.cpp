#include <QApplication>
#include <QCommandLineParser>
#include <QGuiApplication>
#include <QObject>
#include <QMessageBox>
#include <QStringList>
#include <QTextStream>
#include <QQmlApplicationEngine>
#include <QtQml>

#include "app/Application.hpp"
#include "utils/PerformanceGuard.hpp"
#include "utils/RuntimeUtils.hpp"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    QGuiApplication::setOrganizationName(QStringLiteral("bot_core"));
    QGuiApplication::setApplicationName(QStringLiteral("Bot Trading Shell"));
    QGuiApplication::setApplicationVersion(QStringLiteral("0.1.0"));

    const QString platform = QGuiApplication::platformName();
    const bool showDialog = platform.compare(QStringLiteral("offscreen"), Qt::CaseInsensitive) != 0;

    const QString lockPath = bot::shell::utils::runtimeLockFilePath();
    QString directoryError;
    if (!bot::shell::utils::ensureLockFileDirectory(lockPath, &directoryError)) {
        const QString message = directoryError.isEmpty()
            ? QObject::tr("Nie udało się przygotować katalogu pliku blokady instancji.")
            : directoryError;
        QTextStream(stderr) << message << Qt::endl;
        if (showDialog)
            QMessageBox::critical(nullptr, QObject::tr("Bot Trading Shell"), message);
        return EXIT_FAILURE;
    }

    bot::shell::utils::SingleInstanceGuard guard(lockPath);
    if (!guard.tryAcquire()) {
        if (guard.hasConflict()) {
            const auto conflict = guard.conflictInfo();
            QStringList parts;
            if (conflict.pid > 0)
                parts << QObject::tr("PID %1").arg(conflict.pid);
            if (!conflict.hostname.isEmpty())
                parts << QObject::tr("host %1").arg(conflict.hostname);
            if (!conflict.applicationId.isEmpty())
                parts << QObject::tr("aplikacja %1").arg(conflict.applicationId);

            const QString suffix = parts.isEmpty() ? QString() : QStringLiteral(" (%1)").arg(parts.join(QStringLiteral(", ")));
            const QString message = QObject::tr("Bot Trading Shell jest już uruchomiony%1.").arg(suffix);

            QTextStream(stderr) << message << Qt::endl;
            if (showDialog)
                QMessageBox::critical(nullptr, QObject::tr("Bot Trading Shell"), message);

            const QString pythonExecutable = bot::shell::utils::detectSecurityPythonExecutable();
            QString reportError;
            bot::shell::utils::reportSingleInstanceConflict(
                pythonExecutable, lockPath, conflict, &reportError);
            if (!reportError.isEmpty())
                QTextStream(stderr) << reportError << Qt::endl;
        } else {
            const QString message = guard.errorString().isEmpty()
                ? QObject::tr("Nie udało się zarezerwować blokady instancji (kod błędu %1).").arg(guard.lastError())
                : guard.errorString();
            QTextStream(stderr) << message << Qt::endl;
            if (showDialog)
                QMessageBox::critical(nullptr, QObject::tr("Bot Trading Shell"), message);
        }

        return EXIT_FAILURE;
    }

    qmlRegisterUncreatableType<PerformanceGuard>("BotCore", 1, 0, "PerformanceGuard", QStringLiteral("PerformanceGuard is provided by the controller"));

    QQmlApplicationEngine engine;
    Application controller(engine);

    QCommandLineParser parser;
    controller.configureParser(parser);
    parser.process(app);
    controller.applyParser(parser);

    engine.load(QUrl(QStringLiteral("qrc:/qml/main.qml")));
    if (engine.rootObjects().isEmpty()) {
        return -1;
    }

    QMetaObject::invokeMethod(&controller, &Application::start, Qt::QueuedConnection);
    const int exitCode = app.exec();
    controller.stop();
    return exitCode;
}
