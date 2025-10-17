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
    src/app/ActivationController.cpp
    src/grpc/TradingClient.cpp
    src/grpc/MetricsClient.cpp
    src/models/OhlcvListModel.cpp
    src/models/RiskStateModel.cpp
    src/utils/PerformanceGuard.cpp
    src/utils/FrameRateMonitor.cpp
    src/telemetry/UiTelemetryReporter.cpp
    src/license/LicenseActivationController.cpp
    src/security/SecurityAdminController.cpp
)

set(UI_HEADERS
    src/app/Application.hpp
    src/app/ActivationController.hpp
    src/grpc/TradingClient.hpp
    src/grpc/MetricsClient.hpp
    src/models/OhlcvListModel.hpp
    src/models/RiskStateModel.hpp
    src/models/RiskTypes.hpp
    src/utils/PerformanceGuard.hpp
    src/utils/FrameRateMonitor.hpp
    src/telemetry/TelemetryReporter.hpp
    src/telemetry/UiTelemetryReporter.hpp
    src/license/LicenseActivationController.hpp
    src/security/SecurityAdminController.hpp
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
        src/app/Application.cpp
        src/grpc/TradingClient.cpp
        src/models/OhlcvListModel.cpp
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
endif()
