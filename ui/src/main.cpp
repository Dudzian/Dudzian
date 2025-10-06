#include <QCommandLineParser>
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QtQml>

#include "app/Application.hpp"
#include "utils/PerformanceGuard.hpp"

int main(int argc, char* argv[]) {
    QGuiApplication app(argc, argv);
    QGuiApplication::setOrganizationName(QStringLiteral("bot_core"));
    QGuiApplication::setApplicationName(QStringLiteral("Bot Trading Shell"));
    QGuiApplication::setApplicationVersion(QStringLiteral("0.1.0"));

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
