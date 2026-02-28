# Windows UI build prerequisites

`python -m scripts.packaging.qt_bundle --platform windows ...` expects:

- Qt 6.10+ (MSVC kit) available via `--qt-prefix`.
- CMake + Ninja from Qt tools (or in `PATH`).
- MSVC Build Tools 2022 environment.
- vcpkg toolchain and triplet (`x64-windows`) with protobuf + grpc.

Notes for this repository:

- UI proto generation is configured against `<repo>/proto/trading.proto` automatically (no `ui/proto` junction required).
- Production UI target links `Qt6::Network` and `Qt6::Concurrent` modules.
- QML production resources come from `ui/qml/qml.qrc`; tests in `ui/tests` are only built when `BUILD_TESTING=ON`.
