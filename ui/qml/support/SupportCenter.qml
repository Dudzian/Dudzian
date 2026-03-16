import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: root
    objectName: "supportCenterRoot"
    implicitWidth: 1024
    implicitHeight: 640

    property var supportControllerRef: (typeof supportController !== "undefined" ? supportController : null)
    property int articleCount: supportControllerRef ? (supportControllerRef.filteredArticles ? supportControllerRef.filteredArticles.length : 0) : 0
    property var selectedArticleRef: (supportControllerRef && supportControllerRef.selectedArticle) ? supportControllerRef.selectedArticle : null
    property string selectedArticleTitle: selectedArticleRef && selectedArticleRef.title
                                        ? selectedArticleRef.title
                                        : ""
    property var diagnosticsControllerRef: (typeof diagnosticsController !== "undefined" ? diagnosticsController : null)

    signal runbookRequested(string path)

    function triggerSearch() {
        if (supportControllerRef) {
            supportControllerRef.searchArticles(searchInput.text)
        }
    }

    function clearSearch() {
        searchInput.text = ""
        if (supportControllerRef) {
            supportControllerRef.searchArticles("")
        }
    }

    Component.onCompleted: {
        if (supportControllerRef) {
            supportControllerRef.refreshArticles()
        }
    }

    Connections {
        target: supportControllerRef
        ignoreUnknownSignals: true

        function onSearchQueryChanged() {
            if (!supportControllerRef)
                return
            if (searchInput.text !== supportControllerRef.searchQuery)
                searchInput.text = supportControllerRef.searchQuery
        }

        function onSelectedArticleChanged() {
            if (!supportControllerRef)
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
                enabled: !!supportControllerRef
                onClicked: root.triggerSearch()
            }

            Button {
                id: clearButton
                objectName: "supportCenterClearButton"
                text: qsTrId("supportCenter.search.clear")
                enabled: !!supportControllerRef
                onClicked: root.clearSearch()
            }

            Button {
                id: refreshButton
                objectName: "supportCenterRefreshButton"
                text: qsTrId("supportCenter.refresh")
                enabled: !!supportControllerRef
                onClicked: supportControllerRef && supportControllerRef.refreshArticles()
            }

            Button {
                id: ticketButton
                objectName: "supportCenterTicketButton"
                text: qsTrId("supportCenter.ticket.action")
                enabled: !!diagnosticsControllerRef
                onClicked: ticketDialog.open()
            }
        }

        Rectangle {
            id: errorBanner
            objectName: "supportCenterErrorBanner"
            Layout.fillWidth: true
            visible: supportControllerRef && supportControllerRef.errorMessage.length > 0
            color: Qt.rgba(0.8, 0.25, 0.25, 0.9)
            radius: 6
            implicitHeight: visible ? 40 : 0

            Text {
                anchors.centerIn: parent
                text: supportControllerRef ? supportControllerRef.errorMessage : ""
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
                text: supportControllerRef && supportControllerRef.lastUpdated && supportControllerRef.lastUpdated.length > 0
                      ? qsTrId("supportCenter.lastUpdated").arg(supportControllerRef.lastUpdated)
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

            ListView {
                id: articleList
                objectName: "supportCenterArticleList"
                implicitWidth: 340
                clip: true
                model: supportControllerRef ? supportControllerRef.filteredArticles : []
                delegate: Frame {
                    id: articleFrame
                    property var article: modelData
                    objectName: "supportCenterArticleCard"
                    Layout.fillWidth: true
                    padding: 12
                    background: Rectangle {
                        radius: 6
                        color: selectedArticleRef && selectedArticleRef.id === article.id
                               ? Qt.rgba(0.14, 0.5, 0.8, 0.18)
                               : Qt.rgba(0, 0, 0, 0)
                        border.color: Qt.rgba(0.14, 0.5, 0.8, 0.35)
                        border.width: 1
                    }

                    ColumnLayout {
                        anchors.fill: parent
                        spacing: 6

                        Label {
                            text: article && article.title ? article.title : ""
                            font.bold: true
                            wrapMode: Text.WordWrap
                        }

                        Label {
                            text: article && article.summary ? article.summary : ""
                            font.pointSize: 11
                            wrapMode: Text.WordWrap
                            color: "#444444"
                        }

                        Flow {
                            width: parent.width
                            spacing: 6

                            Repeater {
                                model: article && article.tags ? article.tags : []
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
                        onClicked: supportControllerRef && supportControllerRef.selectArticle(article.id)
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
                        text: selectedArticleRef && selectedArticleRef.title
                              ? selectedArticleRef.title
                              : qsTrId("supportCenter.noSelection")
                        font.pixelSize: 20
                        font.bold: true
                        wrapMode: Text.WordWrap
                    }

                    Label {
                        id: articleCategory
                        visible: selectedArticleRef && selectedArticleRef.category && selectedArticleRef.category.length > 0
                        text: selectedArticleRef && selectedArticleRef.category
                              ? selectedArticleRef.category
                              : ""
                        font.pointSize: 11
                        color: "#666666"
                    }

                    Flow {
                        width: parent.width
                        spacing: 6
                        visible: selectedArticleRef && selectedArticleRef.tags && selectedArticleRef.tags.length > 0

                        Repeater {
                            model: selectedArticleRef && selectedArticleRef.tags
                                   ? selectedArticleRef.tags
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
                        text: selectedArticleRef && selectedArticleRef.body
                              ? selectedArticleRef.body
                              : qsTrId("supportCenter.noContent")
                    }

                    Flow {
                        width: parent.width
                        spacing: 8
                        visible: selectedArticleRef && selectedArticleRef.runbooks && selectedArticleRef.runbooks.length > 0

                        Repeater {
                            model: selectedArticleRef && selectedArticleRef.runbooks
                                   ? selectedArticleRef.runbooks
                                   : []
                            delegate: Button {
                                text: modelData && modelData.title ? modelData.title : ""
                                ToolTip.visible: hovered
                                ToolTip.delay: 250
                                ToolTip.text: modelData && modelData.relativePath ? modelData.relativePath : ""
                                onClicked: {
                                    if (modelData && modelData.path && supportControllerRef && supportControllerRef.openRunbook(modelData.path)) {
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
        parent: root
        diagnosticsController: root.diagnosticsControllerRef
    }
}
