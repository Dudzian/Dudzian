import QtQuick
import QtTest
import "../../qml/components" as Components
import "../../qml/views" as Views
import "../../qml/styles" as Styles

TestCase {
    name: "StyleCompliance"

    property var resultsModel: ({
        cumulativeReturn: 0.12,
        maxDrawdown: 0.08,
        sharpeRatio: 1.4,
        annualizedVolatility: 0.21,
        winRate: 0.57,
        equityTimeline: [
            { timestamp: Date.now(), portfolio: 10452.1, drawdown: 0.05, breach: false }
        ],
        exposureHighlights: function() {
            return [{ code: "BTC/USDT", current: 1.0, threshold: 1.5, breached: false }]
        }
    })

    function createComponent(path, properties) {
        const component = Qt.createComponent(path)
        verify(component.status === Component.Ready, "Nie udało się załadować komponentu: " + component.errorString())
        const obj = component.createObject(testCase, properties || {})
        verify(obj !== null, "Nie udało się utworzyć instancji komponentu")
        return obj
    }

    function test_resultsDashboard_themeBindings() {
        const dashboard = createComponent("../../qml/components/ResultsDashboard.qml", { model: resultsModel })
        const column = dashboard.findChild("resultsDashboardColumn")
        verify(column !== null, "Brak kolumny wynikowej")
        compare(column.spacing, Styles.AppTheme.spacingMd)
        const equityGroup = dashboard.findChild("resultsEquityGroup")
        verify(equityGroup !== null)
        compare(equityGroup.background.color, Styles.AppTheme.cardBackground(0.82))
        dashboard.destroy()
    }

    function test_updateManager_usesTheme() {
        const panel = createComponent("../../qml/components/UpdateManagerPanel.qml")
        const column = panel.findChild("updateManagerPanel")
        verify(column !== null)
        compare(column.spacing, Styles.AppTheme.spacingMd)
        const availableGroup = panel.findChild("availableUpdatesGroup")
        verify(availableGroup !== null)
        compare(availableGroup.background.color, Styles.AppTheme.cardBackground(0.82))
        panel.destroy()
    }

    function test_wizards_share_theme() {
        const firstRun = createComponent("../../qml/components/FirstRunWizard.qml")
        const card = firstRun.findChild("firstRunWizardCard")
        verify(card !== null)
        compare(card.background.color, Styles.AppTheme.cardBackground(0.95))
        firstRun.destroy()

        const setup = createComponent("../../qml/views/SetupWizard.qml")
        const frame = setup.findChild("setupWizardFrame")
        verify(frame !== null)
        compare(frame.background.color, Styles.AppTheme.cardBackground(0.9))
        setup.destroy()
    }
}
