#include <QCoreApplication>
#include <QEvent>
#include <QTranslator>
#include <QtTest>

#include "models/RiskCostModel.hpp"
#include "models/RiskLimitsModel.hpp"

namespace {

class ScopedTranslatorInstall {
public:
    explicit ScopedTranslatorInstall(QTranslator* translator)
        : m_translator(translator)
    {
        if (m_translator != nullptr)
            QCoreApplication::installTranslator(m_translator);
    }

    ScopedTranslatorInstall(const ScopedTranslatorInstall&) = delete;
    ScopedTranslatorInstall& operator=(const ScopedTranslatorInstall&) = delete;

    ~ScopedTranslatorInstall()
    {
        if (m_translator != nullptr)
            QCoreApplication::removeTranslator(m_translator);
    }

private:
    QTranslator* m_translator = nullptr;
};

class ScopedLocaleDefault {
public:
    explicit ScopedLocaleDefault(const QLocale& locale)
        : m_previous(QLocale())
    {
        QLocale::setDefault(locale);
    }

    ScopedLocaleDefault(const ScopedLocaleDefault&) = delete;
    ScopedLocaleDefault& operator=(const ScopedLocaleDefault&) = delete;

    ~ScopedLocaleDefault()
    {
        QLocale::setDefault(m_previous);
    }

private:
    QLocale m_previous;
};

} // namespace

class FakeRiskTranslator final : public QTranslator {
public:
    QString translate(const char* context,
                      const char* sourceText,
                      const char* disambiguation = nullptr,
                      int n = -1) const override
    {
        Q_UNUSED(disambiguation);
        Q_UNUSED(n);

        const QString key = QStringLiteral("%1|%2")
                                .arg(QString::fromUtf8(context), QString::fromUtf8(sourceText));

        if (key == QStringLiteral("RiskLimitsModel|Liczba pozycji"))
            return QStringLiteral("Positions count");
        if (key == QStringLiteral("RiskCostModel|Aktywne pozycje"))
            return QStringLiteral("Open positions");
        return QString();
    }
};

class RiskModelsTranslationRefreshTest : public QObject {
    Q_OBJECT

private slots:
    void riskLimitsModelRefreshesLabelsOnLanguageChange();
    void riskCostModelRefreshesLabelsAndFormattedValuesOnLanguageChange();
};

void RiskModelsTranslationRefreshTest::riskLimitsModelRefreshesLabelsOnLanguageChange()
{
    RiskLimitsModel model;
    RiskSnapshotData snapshot;
    snapshot.limits.insert(QStringLiteral("max_positions"), 5.0);

    model.updateFromSnapshot(snapshot);

    QModelIndex target;
    for (int row = 0; row < model.rowCount(); ++row) {
        const QModelIndex idx = model.index(row, 0);
        if (model.data(idx, RiskLimitsModel::KeyRole).toString() == QStringLiteral("max_positions")) {
            target = idx;
            break;
        }
    }

    QVERIFY(target.isValid());
    QCOMPARE(model.data(target, RiskLimitsModel::LabelRole).toString(), QStringLiteral("Liczba pozycji"));

    FakeRiskTranslator translator;
    ScopedTranslatorInstall scopedTranslator(&translator);
    QEvent languageChange(QEvent::LanguageChange);
    QCoreApplication::sendEvent(&model, &languageChange);

    QCOMPARE(model.data(target, RiskLimitsModel::LabelRole).toString(), QStringLiteral("Positions count"));
}

void RiskModelsTranslationRefreshTest::riskCostModelRefreshesLabelsAndFormattedValuesOnLanguageChange()
{
    RiskCostModel model;
    RiskSnapshotData snapshot;
    snapshot.statistics.insert(QStringLiteral("activePositions"), 3);
    snapshot.statistics.insert(QStringLiteral("dailyLossPct"), 0.1234);

    ScopedLocaleDefault englishLocale(QLocale(QLocale::English, QLocale::UnitedStates));
    model.updateFromSnapshot(snapshot);

    QModelIndex activeIndex;
    QModelIndex lossIndex;
    for (int row = 0; row < model.rowCount(); ++row) {
        const QModelIndex idx = model.index(row, 0);
        const QString key = model.data(idx, RiskCostModel::KeyRole).toString();
        if (key == QStringLiteral("activePositions"))
            activeIndex = idx;
        if (key == QStringLiteral("dailyLossPct"))
            lossIndex = idx;
    }

    QVERIFY(activeIndex.isValid());
    QVERIFY(lossIndex.isValid());
    QCOMPARE(model.data(activeIndex, RiskCostModel::LabelRole).toString(), QStringLiteral("Aktywne pozycje"));
    QCOMPARE(model.data(lossIndex, RiskCostModel::FormattedRole).toString(), QStringLiteral("12.34 %"));

    FakeRiskTranslator translator;
    ScopedTranslatorInstall scopedTranslator(&translator);
    ScopedLocaleDefault polishLocale(QLocale(QLocale::Polish, QLocale::Poland));
    QEvent languageChange(QEvent::LanguageChange);
    QCoreApplication::sendEvent(&model, &languageChange);

    QCOMPARE(model.data(activeIndex, RiskCostModel::LabelRole).toString(), QStringLiteral("Open positions"));
    QCOMPARE(model.data(lossIndex, RiskCostModel::FormattedRole).toString(), QStringLiteral("12,34 %"));
}

QTEST_MAIN(RiskModelsTranslationRefreshTest)
#include "RiskModelsTranslationRefreshTest.moc"
