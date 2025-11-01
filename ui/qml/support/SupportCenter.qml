import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: root
    objectName: "supportCenterRoot"
    implicitWidth: 1024
    implicitHeight: 640

    property var supportController: (typeof supportController !== "undefined" ? supportController : null)
    property int articleCount: supportController ? supportController.filteredArticles.length : 0
    property string selectedArticleTitle: supportController && supportController.selectedArticle.title
                                        ? supportController.selectedArticle.title
                                        : ""
    property var diagnosticsController: (typeof diagnosticsController !== "undefined" ? diagnosticsController : null)

    signal runbookRequested(string path)

    function triggerSearch() {
        if (supportController) {
            supportController.searchArticles(searchInput.text)
        }
    }

    function clearSearch() {
        searchInput.text = ""
        if (supportController) {
            supportController.searchArticles("")
        }
    }

    Component.onCompleted: {
        if (supportController) {
            supportController.refreshArticles()
        }
    }

    Connections {
        target: supportController
        ignoreUnknownSignals: true

        function onSearchQueryChanged() {
            if (!supportController)
                return
            if (searchInput.text !== supportController.searchQuery)
                searchInput.text = supportController.searchQuery
        }

        function onSelectedArticleChanged() {
            if (!supportController)
                return
            articleBody.cursorPosition = 0
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 24
        spacing: 16

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            TextField {
                id: searchInput
                objectName: "supportCenterSearchField"
                Layout.fillWidth: true
                placeholderText: qsTrId("supportCenter.search.placeholder")
                onAccepted: root.triggerSearch()
            }

            Button {
                id: searchButton
                objectName: "supportCenterSearchButton"
                text: qsTrId("supportCenter.search.action")
                enabled: !!supportController
                onClicked: root.triggerSearch()
            }

            Button {
                id: clearButton
                objectName: "supportCenterClearButton"
                text: qsTrId("supportCenter.search.clear")
                enabled: !!supportController
                onClicked: root.clearSearch()
            }

            Button {
                id: refreshButton
                objectName: "supportCenterRefreshButton"
                text: qsTrId("supportCenter.refresh")
                enabled: !!supportController
                onClicked: supportController && supportController.refreshArticles()
            }

            Button {
                id: ticketButton
                objectName: "supportCenterTicketButton"
                text: qsTrId("supportCenter.ticket.action")
                enabled: !!diagnosticsController
                onClicked: ticketDialog.open()
            }
        }

        Rectangle {
            id: errorBanner
            objectName: "supportCenterErrorBanner"
            Layout.fillWidth: true
            visible: supportController && supportController.errorMessage.length > 0
            color: Qt.rgba(0.8, 0.25, 0.25, 0.9)
            radius: 6
            implicitHeight: visible ? 40 : 0

            Text {
                anchors.centerIn: parent
                text: supportController ? supportController.errorMessage : ""
                color: "white"
                font.pointSize: 11
                horizontalAlignment: Text.AlignHCenter
            }
        }

        RowLayout {
            Layout.fillWidth: true

            Label {
                id: statusLabel
                objectName: "supportCenterStatusLabel"
                text: supportController && supportController.lastUpdated && supportController.lastUpdated.length > 0
                      ? qsTrId("supportCenter.lastUpdated").arg(supportController.lastUpdated)
                      : qsTrId("supportCenter.lastUpdatedUnknown")
                font.pointSize: 11
                color: "#555555"
            }

            Item { Layout.fillWidth: true }

            Label {
                id: totalLabel
                objectName: "supportCenterTotalLabel"
                text: qsTrId("supportCenter.resultsTotal").arg(root.articleCount)
                font.pointSize: 11
                color: "#555555"
            }
        }

    SplitView {
        id: splitView
        Layout.fillWidth: true
        Layout.fillHeight: true
        handleWidth: 4

            ListView {
                id: articleList
                objectName: "supportCenterArticleList"
                implicitWidth: 340
                clip: true
                model: supportController ? supportController.filteredArticles : []
                delegate: Frame {
                    id: articleFrame
                    property var article: modelData
                    objectName: "supportCenterArticleCard"
                    Layout.fillWidth: true
                    padding: 12
                    background: Rectangle {
                        radius: 6
                        color: supportController && supportController.selectedArticle.id === article.id
                               ? Qt.rgba(0.14, 0.5, 0.8, 0.18)
                               : Qt.rgba(0, 0, 0, 0)
                        border.color: Qt.rgba(0.14, 0.5, 0.8, 0.35)
                        border.width: 1
                    }

                    ColumnLayout {
                        anchors.fill: parent
                        spacing: 6

                        Label {
                            text: article.title
                            font.bold: true
                            wrapMode: Text.WordWrap
                        }

                        Label {
                            text: article.summary
                            font.pointSize: 11
                            wrapMode: Text.WordWrap
                            color: "#444444"
                        }

                        Flow {
                            width: parent.width
                            spacing: 6

                            Repeater {
                                model: article.tags
                                delegate: Rectangle {
                                    radius: 8
                                    color: Qt.rgba(0.14, 0.5, 0.8, 0.15)
                                    border.color: Qt.rgba(0.14, 0.5, 0.8, 0.4)
                                    border.width: 1
                                    height: tagLabel.implicitHeight + 4
                                    width: tagLabel.implicitWidth + 16

                                    Text {
                                        id: tagLabel
                                        anchors.centerIn: parent
                                        text: modelData
                                        font.pointSize: 10
                                    }
                                }
                            }
                        }
                    }

                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        onClicked: supportController && supportController.selectArticle(article.id)
                    }
                }
            }

            ScrollView {
                id: articleView
                objectName: "supportCenterArticleView"
                Layout.fillWidth: true
                Layout.fillHeight: true

                ColumnLayout {
                    width: articleView.width - 24
                    anchors.margins: 12
                    spacing: 12

                    Label {
                        id: articleTitle
                        objectName: "supportCenterArticleTitle"
                        text: supportController && supportController.selectedArticle.title
                              ? supportController.selectedArticle.title
                              : qsTrId("supportCenter.noSelection")
                        font.pixelSize: 20
                        font.bold: true
                        wrapMode: Text.WordWrap
                    }

                    Label {
                        id: articleCategory
                        visible: supportController && supportController.selectedArticle.category && supportController.selectedArticle.category.length > 0
                        text: supportController && supportController.selectedArticle.category
                              ? supportController.selectedArticle.category
                              : ""
                        font.pointSize: 11
                        color: "#666666"
                    }

                    Flow {
                        width: parent.width
                        spacing: 6
                        visible: supportController && supportController.selectedArticle.tags && supportController.selectedArticle.tags.length > 0

                        Repeater {
                            model: supportController && supportController.selectedArticle.tags
                                   ? supportController.selectedArticle.tags
                                   : []
                            delegate: Rectangle {
                                radius: 8
                                color: Qt.rgba(0.14, 0.5, 0.8, 0.15)
                                border.color: Qt.rgba(0.14, 0.5, 0.8, 0.4)
                                border.width: 1
                                height: tagItem.implicitHeight + 4
                                width: tagItem.implicitWidth + 16

                                Text {
                                    id: tagItem
                                    anchors.centerIn: parent
                                    text: modelData
                                    font.pointSize: 10
                                }
                            }
                        }
                    }

                    TextArea {
                        id: articleBody
                        objectName: "supportCenterArticleBody"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        readOnly: true
                        wrapMode: TextEdit.WordWrap
                        textFormat: TextEdit.MarkdownText
                        text: supportController && supportController.selectedArticle.body
                              ? supportController.selectedArticle.body
                              : qsTrId("supportCenter.noContent")
                    }

                    Flow {
                        width: parent.width
                        spacing: 8
                        visible: supportController && supportController.selectedArticle.runbooks && supportController.selectedArticle.runbooks.length > 0

                        Repeater {
                            model: supportController && supportController.selectedArticle.runbooks
                                   ? supportController.selectedArticle.runbooks
                                   : []
                            delegate: Button {
                                text: modelData.title
                                ToolTip.visible: hovered
                                ToolTip.delay: 250
                                ToolTip.text: modelData.relativePath
                                onClicked: {
                                    if (supportController && supportController.openRunbook(modelData.path)) {
                                        Qt.openUrlExternally("file://" + modelData.path)
                                        root.runbookRequested(modelData.path)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    TicketDialog {
        id: ticketDialog
        diagnosticsController: root.diagnosticsController
    }
}
