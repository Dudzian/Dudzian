#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QVariantList>

class ReportCenterController : public QObject {
    Q_OBJECT
    Q_PROPERTY(QVariantList reports READ reports NOTIFY reportsChanged)
    Q_PROPERTY(bool busy READ isBusy NOTIFY busyChanged)

public:
    explicit ReportCenterController(QObject* parent = nullptr);

    QVariantList reports() const { return m_reports; }
    bool isBusy() const { return m_busy; }

    Q_INVOKABLE bool refresh();
    Q_INVOKABLE bool deleteReport(const QString& path);

    void setPythonExecutable(const QString& executable);
    void setReportsRoot(const QString& root);

signals:
    void reportsChanged();
    void busyChanged();
    void reportDeleted(const QString& path);
    void reportOperationFailed(const QString& message);

private:
    bool runBridge(const QStringList& arguments, QByteArray* stdoutData, QByteArray* stderrData) const;
    bool loadReportsFromJson(const QByteArray& data);

    QString m_pythonExecutable = QStringLiteral("python3");
    QString m_reportsRoot;
    QVariantList m_reports;
    bool m_busy = false;
};
