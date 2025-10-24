cmake_minimum_required(VERSION 3.21)

project(bot_trading_shell LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 6.5 COMPONENTS Quick QuickControls2 Charts REQUIRED)

# --- Testing toggle ---
include(CTest)
if(BUILD_TESTING)
    # Pull in all modules needed by tests only when testing is enabled
    find_package(Qt6 6.5 COMPONENTS Test Gui Qml Quick Charts REQUIRED)
endif()

# --- Protobuf / gRPC ---
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

set(PROTO_SRC ${CMAKE_SOURCE_DIR}/proto/trading.proto)
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_DIR})

add_custom_command(
    OUTPUT
        ${GENERATED_DIR}/trading.pb.cc
        ${GENERATED_DIR}/trading.pb.h
        ${GENERATED_DIR}/trading.grpc.pb.cc
        ${GENERATED_DIR}/trading.grpc.pb.h
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
        --proto_path=${CMAKE_SOURCE_DIR}/proto
        --cpp_out=${GENERATED_DIR}
        --grpc_out=${GENERATED_DIR}
        --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
        ${PROTO_SRC}
    DEPENDS
        ${PROTO_SRC}
        gRPC::grpc_cpp_plugin
    COMMENT "Generating C++ sources from ${PROTO_SRC}"
)

add_library(trading_proto STATIC
    ${GENERATED_DIR}/trading.pb.cc
    ${GENERATED_DIR}/trading.grpc.pb.cc
)
target_include_directories(trading_proto PUBLIC ${GENERATED_DIR})
target_link_libraries(trading_proto
    PUBLIC
        protobuf::libprotobuf
        gRPC::grpc++
        gRPC::grpc++_reflection
)

# --- App sources ---
set(UI_SOURCES
    src/main.cpp
    src/app/Application.cpp
    src/app/UiModuleManager.cpp
    src/app/UiModuleViewsModel.cpp
    src/app/ActivationController.cpp
    src/app/StrategyConfigController.cpp
    src/grpc/BotCoreLocalService.cpp
    src/grpc/TradingClient.cpp
    src/grpc/MetricsClient.cpp
    src/grpc/HealthClient.cpp
    src/runtime/OfflineRuntimeBridge.cpp
    src/models/OhlcvListModel.cpp
    src/models/AlertsModel.cpp
    src/models/AlertsFilterProxyModel.cpp
    src/models/RiskHistoryModel.cpp
    src/models/RiskStateModel.cpp
    src/models/DecisionLogModel.cpp
    src/utils/PerformanceGuard.cpp
    src/utils/FrameRateMonitor.cpp
    src/utils/PathUtils.cpp
    src/telemetry/UiTelemetryReporter.cpp
    src/license/LicenseActivationController.cpp
    src/security/SecurityAdminController.cpp
    src/reporting/ReportCenterController.cpp
    src/support/SupportBundleController.cpp
    src/health/HealthStatusController.cpp
)

set(UI_HEADERS
    src/app/Application.hpp
    src/app/UiModuleManager.hpp
    src/app/UiModuleInterface.hpp
    src/app/UiModuleViewsModel.hpp
    src/app/ActivationController.hpp
    src/app/StrategyConfigController.hpp
    src/grpc/BotCoreLocalService.hpp
    src/grpc/TradingClient.hpp
    src/grpc/MetricsClient.hpp
    src/grpc/HealthClient.hpp
    src/runtime/OfflineRuntimeBridge.hpp
    src/grpc/GrpcTlsConfig.hpp
    src/models/AlertsModel.hpp
    src/models/AlertsFilterProxyModel.hpp
    src/models/OhlcvListModel.hpp
    src/models/RiskHistoryModel.hpp
    src/models/RiskStateModel.hpp
    src/models/DecisionLogModel.hpp
    src/models/RiskTypes.hpp
    src/utils/PerformanceGuard.hpp
    src/utils/FrameRateMonitor.hpp
    src/utils/PathUtils.hpp
    src/telemetry/TelemetryReporter.hpp
    src/telemetry/UiTelemetryReporter.hpp
    src/license/LicenseActivationController.hpp
    src/security/SecurityAdminController.hpp
    src/reporting/ReportCenterController.hpp
    src/support/SupportBundleController.hpp
    src/health/HealthStatusController.hpp
)

qt_add_executable(bot_trading_shell
    ${UI_SOURCES}
    ${UI_HEADERS}
    qml/qml.qrc
)

target_include_directories(bot_trading_shell PRIVATE src ${GENERATED_DIR})
target_link_libraries(bot_trading_shell
    PRIVATE
        Qt6::Quick
        Qt6::QuickControls2
        Qt6::Charts
        trading_proto
)

install(TARGETS bot_trading_shell RUNTIME DESTINATION bin)

# --- Tests ---
if(BUILD_TESTING)
    # UiTelemetryReporter tests (needs protobuf/gRPC stubs)
    add_executable(ui_telemetry_tests
        tests/UiTelemetryReporterTest.cpp
        src/utils/PerformanceGuard.cpp
        src/telemetry/UiTelemetryReporter.cpp
    )
    target_include_directories(ui_telemetry_tests PRIVATE src ${GENERATED_DIR})
    target_link_libraries(ui_telemetry_tests
        PRIVATE
            Qt6::Core
            Qt6::Test
            trading_proto
            protobuf::libprotobuf
            gRPC::grpc++
    )
    add_test(NAME ui_telemetry_tests COMMAND ui_telemetry_tests)

    # Application-level telemetry tests
    add_executable(ui_application_telemetry_tests
        tests/ApplicationTelemetryTest.cpp
        tests/ApplicationHealthConfigTest.cpp
        tests/ApplicationAuthTokenWatcherTest.cpp
        src/app/Application.cpp
        src/app/UiModuleViewsModel.cpp
        src/models/DecisionLogModel.cpp
        src/utils/PathUtils.cpp
        src/grpc/TradingClient.cpp
        src/grpc/HealthClient.cpp
        src/health/HealthStatusController.cpp
        src/models/AlertsModel.cpp
        src/models/AlertsFilterProxyModel.cpp
        src/models/OhlcvListModel.cpp
        src/models/RiskHistoryModel.cpp
        src/utils/PerformanceGuard.cpp
        src/utils/FrameRateMonitor.cpp
        src/telemetry/UiTelemetryReporter.cpp
    )
    target_include_directories(ui_application_telemetry_tests PRIVATE src ${GENERATED_DIR})
    target_link_libraries(ui_application_telemetry_tests
        PRIVATE
            Qt6::Core
            Qt6::Gui
            Qt6::Quick
            Qt6::Test
            trading_proto
            protobuf::libprotobuf
            gRPC::grpc++
    )
    add_test(NAME ui_application_telemetry_tests COMMAND ui_application_telemetry_tests)

    add_executable(metrics_client_preflight_tests
        tests/MetricsClientPreflightTest.cpp
        src/grpc/MetricsClient.cpp
    )
    target_include_directories(metrics_client_preflight_tests PRIVATE src ${GENERATED_DIR})
    target_link_libraries(metrics_client_preflight_tests
        PRIVATE
            Qt6::Core
            Qt6::Test
            trading_proto
            protobuf::libprotobuf
            gRPC::grpc++
    )
    add_test(NAME metrics_client_preflight_tests COMMAND metrics_client_preflight_tests)

    add_executable(health_client_tests
        tests/HealthClientTest.cpp
        src/grpc/HealthClient.cpp
    )
    target_include_directories(health_client_tests PRIVATE src ${GENERATED_DIR})
    target_link_libraries(health_client_tests
        PRIVATE
            Qt6::Core
            Qt6::Test
            trading_proto
            protobuf::libprotobuf
            gRPC::grpc++
    )
    add_test(NAME health_client_tests COMMAND health_client_tests)

    add_executable(path_utils_tests
        tests/PathUtilsTest.cpp
        src/utils/PathUtils.cpp
    )
    target_include_directories(path_utils_tests PRIVATE src)
    target_link_libraries(path_utils_tests
        PRIVATE
            Qt6::Core
            Qt6::Test
    )
    add_test(NAME path_utils_tests COMMAND path_utils_tests)

    add_executable(health_status_controller_tests
        tests/HealthStatusControllerTest.cpp
        src/health/HealthStatusController.cpp
        src/grpc/HealthClient.cpp
    )
    target_include_directories(health_status_controller_tests PRIVATE src ${GENERATED_DIR})
    target_link_libraries(health_status_controller_tests
        PRIVATE
            Qt6::Core
            Qt6::Test
            trading_proto
            protobuf::libprotobuf
            gRPC::grpc++
    )
    add_test(NAME health_status_controller_tests COMMAND health_status_controller_tests)

    add_executable(application_risk_refresh_tests
        tests/ApplicationRiskRefreshTest.cpp
        src/app/Application.cpp
        src/app/UiModuleViewsModel.cpp
        src/utils/PathUtils.cpp
        src/grpc/TradingClient.cpp
        src/grpc/BotCoreLocalService.cpp
        src/models/OhlcvListModel.cpp
        src/models/AlertsModel.cpp
        src/models/AlertsFilterProxyModel.cpp
        src/models/RiskHistoryModel.cpp
        src/models/RiskStateModel.cpp
        src/models/DecisionLogModel.cpp
        src/utils/PerformanceGuard.cpp
        src/utils/FrameRateMonitor.cpp
        src/telemetry/UiTelemetryReporter.cpp
        src/license/LicenseActivationController.cpp
        src/security/SecurityAdminController.cpp
        src/reporting/ReportCenterController.cpp
    )
    target_include_directories(application_risk_refresh_tests PRIVATE src ${GENERATED_DIR})
    target_link_libraries(application_risk_refresh_tests
        PRIVATE
            Qt6::Core
            Qt6::Gui
            Qt6::Quick
            Qt6::Test
            trading_proto
            protobuf::libprotobuf
            gRPC::grpc++
    )
    add_test(NAME application_risk_refresh_tests COMMAND application_risk_refresh_tests)

    add_executable(application_settings_persistence_tests
        tests/ApplicationSettingsPersistenceTest.cpp
        src/app/Application.cpp
        src/app/UiModuleViewsModel.cpp
        src/utils/PathUtils.cpp
        src/grpc/TradingClient.cpp
        src/grpc/BotCoreLocalService.cpp
        src/models/OhlcvListModel.cpp
        src/models/AlertsModel.cpp
        src/models/AlertsFilterProxyModel.cpp
        src/models/RiskHistoryModel.cpp
        src/models/RiskStateModel.cpp
        src/models/DecisionLogModel.cpp
        src/utils/PerformanceGuard.cpp
        src/utils/FrameRateMonitor.cpp
        src/telemetry/UiTelemetryReporter.cpp
        src/license/LicenseActivationController.cpp
        src/security/SecurityAdminController.cpp
        src/reporting/ReportCenterController.cpp
    )
    target_include_directories(application_settings_persistence_tests PRIVATE src ${GENERATED_DIR})
    target_link_libraries(application_settings_persistence_tests
        PRIVATE
            Qt6::Core
            Qt6::Gui
            Qt6::Quick
            Qt6::Test
            trading_proto
            protobuf::libprotobuf
            gRPC::grpc++
    )
    add_test(NAME application_settings_persistence_tests COMMAND application_settings_persistence_tests)

    add_executable(ui_module_manager_tests
        tests/UiModuleManagerTest.cpp
        src/app/UiModuleManager.cpp
    )
    target_include_directories(ui_module_manager_tests PRIVATE src)
    target_link_libraries(ui_module_manager_tests
        PRIVATE
            Qt6::Core
            Qt6::Test
    )
    add_test(NAME ui_module_manager_tests COMMAND ui_module_manager_tests)

    add_executable(ui_module_views_model_tests
        tests/UiModuleViewsModelTest.cpp
        src/app/UiModuleManager.cpp
        src/app/UiModuleViewsModel.cpp
    )
    target_include_directories(ui_module_views_model_tests PRIVATE src)
    target_link_libraries(ui_module_views_model_tests
        PRIVATE
            Qt6::Core
            Qt6::Test
    )
    add_test(NAME ui_module_views_model_tests COMMAND ui_module_views_model_tests)

    add_executable(application_risk_history_auto_export_tests
        tests/ApplicationRiskHistoryAutoExportTest.cpp
        src/app/Application.cpp
        src/app/UiModuleViewsModel.cpp
        src/utils/PathUtils.cpp
        src/grpc/TradingClient.cpp
        src/grpc/BotCoreLocalService.cpp
        src/models/OhlcvListModel.cpp
        src/models/AlertsModel.cpp
        src/models/AlertsFilterProxyModel.cpp
        src/models/RiskHistoryModel.cpp
        src/models/RiskStateModel.cpp
        src/models/DecisionLogModel.cpp
        src/utils/PerformanceGuard.cpp
        src/utils/FrameRateMonitor.cpp
        src/telemetry/UiTelemetryReporter.cpp
        src/license/LicenseActivationController.cpp
        src/security/SecurityAdminController.cpp
        src/reporting/ReportCenterController.cpp
    )
    target_include_directories(application_risk_history_auto_export_tests PRIVATE src ${GENERATED_DIR})
    target_link_libraries(application_risk_history_auto_export_tests
        PRIVATE
            Qt6::Core
            Qt6::Gui
            Qt6::Quick
            Qt6::Test
            trading_proto
            protobuf::libprotobuf
            gRPC::grpc++
    )
    add_test(NAME application_risk_history_auto_export_tests COMMAND application_risk_history_auto_export_tests)

    add_executable(application_decision_log_tests
        tests/ApplicationDecisionLogTest.cpp
        src/app/Application.cpp
        src/app/UiModuleViewsModel.cpp
        src/app/UiModuleManager.cpp
        src/utils/PathUtils.cpp
        src/grpc/TradingClient.cpp
        src/grpc/BotCoreLocalService.cpp
        src/models/OhlcvListModel.cpp
        src/models/AlertsModel.cpp
        src/models/AlertsFilterProxyModel.cpp
        src/models/RiskHistoryModel.cpp
        src/models/RiskStateModel.cpp
        src/models/DecisionLogModel.cpp
        src/utils/PerformanceGuard.cpp
        src/utils/FrameRateMonitor.cpp
        src/telemetry/UiTelemetryReporter.cpp
        src/license/LicenseActivationController.cpp
        src/security/SecurityAdminController.cpp
        src/reporting/ReportCenterController.cpp
        src/support/SupportBundleController.cpp
        src/health/HealthStatusController.cpp
    )
    target_include_directories(application_decision_log_tests PRIVATE src ${GENERATED_DIR})
    target_link_libraries(application_decision_log_tests
        PRIVATE
            Qt6::Core
            Qt6::Gui
            Qt6::Quick
            Qt6::Test
            trading_proto
            protobuf::libprotobuf
            gRPC::grpc++
    )
    add_test(NAME application_decision_log_tests COMMAND application_decision_log_tests)

    add_executable(application_ui_modules_tests
        tests/ApplicationUiModulesTest.cpp
        src/app/Application.cpp
        src/app/UiModuleManager.cpp
        src/app/UiModuleViewsModel.cpp
        src/utils/PathUtils.cpp
        src/grpc/TradingClient.cpp
        src/grpc/BotCoreLocalService.cpp
        src/models/OhlcvListModel.cpp
        src/models/AlertsModel.cpp
        src/models/AlertsFilterProxyModel.cpp
        src/models/RiskHistoryModel.cpp
        src/models/RiskStateModel.cpp
        src/models/DecisionLogModel.cpp
        src/utils/PerformanceGuard.cpp
        src/utils/FrameRateMonitor.cpp
        src/telemetry/UiTelemetryReporter.cpp
        src/license/LicenseActivationController.cpp
        src/security/SecurityAdminController.cpp
        src/reporting/ReportCenterController.cpp
        src/support/SupportBundleController.cpp
        src/health/HealthStatusController.cpp
    )
    target_include_directories(application_ui_modules_tests PRIVATE src ${GENERATED_DIR})
    target_link_libraries(application_ui_modules_tests
        PRIVATE
            Qt6::Core
            Qt6::Gui
            Qt6::Quick
            Qt6::Test
            trading_proto
            protobuf::libprotobuf
            gRPC::grpc++
    )
    add_test(NAME application_ui_modules_tests COMMAND application_ui_modules_tests)

    # FrameRateMonitor tests
    add_executable(frame_rate_monitor_tests
        tests/FrameRateMonitorTest.cpp
        src/utils/PerformanceGuard.cpp
        src/utils/FrameRateMonitor.cpp
    )
    target_include_directories(frame_rate_monitor_tests PRIVATE src)
    target_link_libraries(frame_rate_monitor_tests PRIVATE Qt6::Core Qt6::Test)
    add_test(NAME frame_rate_monitor_tests COMMAND frame_rate_monitor_tests)

    # CandlestickChartView tests (needs QML/Charts)
    add_executable(candlestick_chart_view_tests
        tests/CandlestickChartViewTest.cpp
        src/utils/PerformanceGuard.cpp
    )
    target_include_directories(candlestick_chart_view_tests PRIVATE src ${GENERATED_DIR})
    qt_add_resources(candlestick_chart_view_tests candlestick_chart_view_qml
        PREFIX "/qml/components"
        FILES qml/components/CandlestickChartView.qml
    )
    target_link_libraries(candlestick_chart_view_tests
        PRIVATE Qt6::Gui Qt6::Qml Qt6::Quick Qt6::Charts Qt6::Test
    )
    add_test(NAME candlestick_chart_view_tests COMMAND candlestick_chart_view_tests)

    # RiskStateModel tests
    add_executable(risk_state_model_tests
        tests/RiskStateModelTest.cpp
        src/models/RiskStateModel.cpp
    )
    target_include_directories(risk_state_model_tests PRIVATE src)
    target_link_libraries(risk_state_model_tests PRIVATE Qt6::Core Qt6::Test)
    add_test(NAME risk_state_model_tests COMMAND risk_state_model_tests)

    add_executable(risk_history_model_tests
        tests/RiskHistoryModelTest.cpp
        src/models/RiskHistoryModel.cpp
    )
    target_include_directories(risk_history_model_tests PRIVATE src)
    target_link_libraries(risk_history_model_tests PRIVATE Qt6::Core Qt6::Test)
    add_test(NAME risk_history_model_tests COMMAND risk_history_model_tests)

    add_executable(decision_log_model_tests
        tests/DecisionLogModelTest.cpp
        src/models/DecisionLogModel.cpp
    )
    target_include_directories(decision_log_model_tests PRIVATE src)
    target_link_libraries(decision_log_model_tests PRIVATE Qt6::Core Qt6::Test)
    add_test(NAME decision_log_model_tests COMMAND decision_log_model_tests)

    add_executable(alerts_model_tests
        tests/AlertsModelTest.cpp
        src/models/AlertsModel.cpp
    )
    target_include_directories(alerts_model_tests PRIVATE src)
    target_link_libraries(alerts_model_tests PRIVATE Qt6::Core Qt6::Test)
    add_test(NAME alerts_model_tests COMMAND alerts_model_tests)

    add_executable(alerts_filter_proxy_model_tests
        tests/AlertsFilterProxyModelTest.cpp
        src/models/AlertsModel.cpp
        src/models/AlertsFilterProxyModel.cpp
    )
    target_include_directories(alerts_filter_proxy_model_tests PRIVATE src)
    target_link_libraries(alerts_filter_proxy_model_tests PRIVATE Qt6::Core Qt6::Test)
    add_test(NAME alerts_filter_proxy_model_tests COMMAND alerts_filter_proxy_model_tests)

    # SidePanel QML tests
    add_executable(side_panel_tests
        tests/SidePanelTest.cpp
        src/models/RiskStateModel.cpp
        src/utils/PerformanceGuard.cpp
    )
    target_include_directories(side_panel_tests PRIVATE src)
    qt_add_resources(side_panel_tests side_panel_qml
        PREFIX "/qml/components"
        FILES qml/components/SidePanel.qml
    )
    target_link_libraries(side_panel_tests
        PRIVATE
            Qt6::Core
            Qt6::Gui
            Qt6::Qml
            Qt6::Quick
            Qt6::QuickControls2
            Qt6::Test
    )
    add_test(NAME side_panel_tests COMMAND side_panel_tests)

    add_executable(license_activation_controller_tests
        tests/LicenseActivationControllerTest.cpp
        src/license/LicenseActivationController.cpp
        src/utils/PathUtils.cpp
    )
    target_include_directories(license_activation_controller_tests PRIVATE src)
    target_link_libraries(license_activation_controller_tests
        PRIVATE
            Qt6::Core
            Qt6::Test
    )
    add_test(NAME license_activation_controller_tests COMMAND license_activation_controller_tests)

    add_executable(security_admin_controller_tests
        tests/SecurityAdminControllerTest.cpp
        src/security/SecurityAdminController.cpp
    )
    target_include_directories(security_admin_controller_tests PRIVATE src)
    target_link_libraries(security_admin_controller_tests PRIVATE Qt6::Core Qt6::Test)
    add_test(NAME security_admin_controller_tests COMMAND security_admin_controller_tests)

    add_executable(admin_dialog_e2e_tests
        tests/AdminDialogE2ETest.cpp
        src/security/SecurityAdminController.cpp
    )
    target_include_directories(admin_dialog_e2e_tests PRIVATE src)
    qt_add_resources(admin_dialog_e2e_tests admin_dialog_e2e_qml
        PREFIX "/qml"
        FILES qml/components/AdminDialog.qml
    )
    target_link_libraries(admin_dialog_e2e_tests
        PRIVATE
            Qt6::Core
            Qt6::Gui
            Qt6::Qml
            Qt6::Quick
            Qt6::QuickControls2
            Qt6::Test
    )
    add_test(NAME admin_dialog_e2e_tests COMMAND admin_dialog_e2e_tests)

    add_executable(first_run_wizard_e2e_tests
        tests/FirstRunWizardE2ETest.cpp
    )
    target_include_directories(first_run_wizard_e2e_tests PRIVATE src)
    qt_add_resources(first_run_wizard_e2e_tests first_run_wizard_e2e_qml
        PREFIX "/qml/components"
        FILES qml/components/FirstRunWizard.qml
    )
    target_link_libraries(first_run_wizard_e2e_tests
        PRIVATE
            Qt6::Core
            Qt6::Gui
            Qt6::Qml
            Qt6::Quick
            Qt6::QuickControls2
            Qt6::Test
    )
    add_test(NAME first_run_wizard_e2e_tests COMMAND first_run_wizard_e2e_tests)

    add_executable(strategy_workbench_e2e_tests
        tests/StrategyWorkbenchE2ETest.cpp
    )
    target_include_directories(strategy_workbench_e2e_tests PRIVATE src)
    qt_add_resources(strategy_workbench_e2e_tests strategy_workbench_e2e_qml
        PREFIX "/qml/components/workbench"
        FILES
            qml/components/workbench/StrategyWorkbench.qml
            qml/components/workbench/StrategyWorkbenchViewModel.qml
    )
    target_link_libraries(strategy_workbench_e2e_tests
        PRIVATE
            Qt6::Core
            Qt6::Gui
            Qt6::Qml
            Qt6::Quick
            Qt6::QuickControls2
            Qt6::Test
    )
    add_test(NAME strategy_workbench_e2e_tests COMMAND strategy_workbench_e2e_tests)

    add_executable(bot_core_local_service_tests
        tests/BotCoreLocalServiceTest.cpp
        src/grpc/BotCoreLocalService.cpp
        src/grpc/TradingClient.cpp
    )
    target_include_directories(bot_core_local_service_tests PRIVATE src ${GENERATED_DIR})
    target_link_libraries(bot_core_local_service_tests
        PRIVATE
            Qt6::Core
            Qt6::Gui
            Qt6::Qml
            Qt6::Quick
            Qt6::Test
            trading_proto
            protobuf::libprotobuf
            gRPC::grpc++
    )
    add_test(NAME bot_core_local_service_tests COMMAND bot_core_local_service_tests)

    add_executable(trading_client_auth_metadata_tests
        tests/TradingClientAuthMetadataTest.cpp
        src/grpc/TradingClient.cpp
    )
    target_include_directories(trading_client_auth_metadata_tests PRIVATE src ${GENERATED_DIR})
    target_link_libraries(trading_client_auth_metadata_tests
        PRIVATE
            Qt6::Core
            Qt6::Test
            Qt6::Network
            trading_proto
            protobuf::libprotobuf
            gRPC::grpc++
    )
    add_test(NAME trading_client_auth_metadata_tests COMMAND trading_client_auth_metadata_tests)

    add_executable(report_center_controller_tests
        tests/ReportCenterControllerTest.cpp
        src/reporting/ReportCenterController.cpp
    )
    target_include_directories(report_center_controller_tests PRIVATE src)
    target_link_libraries(report_center_controller_tests PRIVATE Qt6::Core Qt6::Test)
    add_test(NAME report_center_controller_tests COMMAND report_center_controller_tests)

    add_executable(strategy_config_controller_tests
        tests/StrategyConfigControllerTest.cpp
        src/app/StrategyConfigController.cpp
    )
    target_include_directories(strategy_config_controller_tests PRIVATE src ${GENERATED_DIR})
    target_link_libraries(strategy_config_controller_tests PRIVATE Qt6::Core Qt6::Test)
    add_test(NAME strategy_config_controller_tests COMMAND strategy_config_controller_tests)

    add_executable(support_bundle_controller_tests
        tests/SupportBundleControllerTest.cpp
        src/support/SupportBundleController.cpp
    )
    target_include_directories(support_bundle_controller_tests PRIVATE src ${GENERATED_DIR})
    target_link_libraries(support_bundle_controller_tests PRIVATE Qt6::Core Qt6::Test)
    add_test(NAME support_bundle_controller_tests COMMAND support_bundle_controller_tests)
endif()
