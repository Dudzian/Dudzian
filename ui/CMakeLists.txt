cmake_minimum_required(VERSION 3.21)

project(bot_trading_shell LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 6.5 COMPONENTS Quick QuickControls2 Charts REQUIRED)

# --- Testing (from HEAD) ---
include(CTest)
if(BUILD_TESTING)
    find_package(Qt6 6.5 COMPONENTS Test REQUIRED)
endif()

find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

# ---- Protobuf / gRPC codegen ----
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

# ---- App sources ----
set(UI_SOURCES
    src/main.cpp
    src/app/Application.cpp
    src/grpc/TradingClient.cpp
    src/models/OhlcvListModel.cpp
    src/utils/PerformanceGuard.cpp
    src/utils/FrameRateMonitor.cpp
    src/telemetry/UiTelemetryReporter.cpp
)

set(UI_HEADERS
    src/app/Application.hpp
    src/grpc/TradingClient.hpp
    src/models/OhlcvListModel.hpp
    src/utils/PerformanceGuard.hpp
    src/utils/FrameRateMonitor.hpp
    src/telemetry/TelemetryReporter.hpp
    src/telemetry/UiTelemetryReporter.hpp
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

# ---- Tests (from HEAD) ----
if(BUILD_TESTING)
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
endif()
