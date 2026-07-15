"""FUNCTIONAL-PREVIEW-18.2 Block P desktop EXE packaging inventory matrix."""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_p_desktop_exe_packaging_source_inventory import (
    build_preview_block_p_desktop_exe_packaging_source_inventory,
)

SCHEMA_VERSION: Final[str] = "preview_block_p_desktop_exe_packaging_inventory_matrix.v1"
KIND: Final[str] = "functional_preview_block_p_desktop_exe_packaging_inventory_matrix"
BLOCK_ID: Final[str] = "P"
STEP_ID: Final[str] = "18.2"
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-18.3"
NEXT_STEP_TITLE: Final[str] = "BLOCK P DESKTOP EXE PACKAGING CONTRACT"
STATUS: Final[str] = "ready_for_functional_preview_18_3_block_p_desktop_exe_packaging_contract"
BLOCKED_STATUS: Final[str] = (
    "blocked_for_functional_preview_18_3_block_p_desktop_exe_packaging_inventory_matrix_source_not_accepted"
)
INVENTORY_MATRIX_STATUS: Final[str] = (
    "source_18_1_consumed_source_inventory_preserved_static_source_only_matrix_complete_four_entrypoint_rows_evaluated_qml_inventory_evaluated_dependency_declarations_evaluated_packaging_metadata_evaluated_cli_preview_packaging_evaluated_as_separate_scope_exclusion_policy_evaluated_11_findings_evaluated_3_packaging_scopes_evaluated_8_packaging_requirements_evaluated_unresolved_contract_blockers_recorded_no_entrypoint_selection_no_validation_no_approval_no_packaging_no_build_no_artifact_no_release_no_runtime_no_orders_only_source_only_handoff_to_18_3_allowed"
)
INVENTORY_MATRIX_DECISION: Final[str] = INVENTORY_MATRIX_STATUS.upper()
MAX_DIAGNOSTIC_CONTAINER_DEPTH: Final[int] = 64
TOP_LEVEL_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_p_desktop_exe_packaging_inventory_matrix_kind",
    "block",
    "step",
    "block_p_desktop_exe_packaging_inventory_matrix_status",
    "block_p_desktop_exe_packaging_inventory_matrix_decision",
    "inventory_matrix_artifact_complete",
    "ready_for_block_p_3",
    "next_step",
    "next_step_title",
    "block_p_desktop_exe_packaging_source_inventory_reference",
    "inventory_matrix_summary",
    "source_inventory_preservation",
    "desktop_entrypoint_matrix_rows",
    "qml_bundle_matrix_rows",
    "python_dependency_matrix_rows",
    "packaging_metadata_matrix_rows",
    "existing_preview_packaging_matrix_rows",
    "artifact_exclusion_policy_matrix_rows",
    "inventory_finding_matrix_rows",
    "packaging_scope_matrix_rows",
    "packaging_requirement_matrix_rows",
    "unresolved_contract_blocker_rows",
    "real_capability_matrix_state",
    "fail_closed_matrix_decision",
    "non_execution_matrix_evidence",
    "matrix_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
TOP_LEVEL_FIELDS_18_1: Final[list[str]] = [
    "schema_version",
    "block_p_desktop_exe_packaging_source_inventory_kind",
    "block",
    "step",
    "block_p_desktop_exe_packaging_source_inventory_status",
    "block_p_desktop_exe_packaging_source_inventory_decision",
    "source_inventory_artifact_complete",
    "ready_for_block_p_2",
    "next_step",
    "next_step_title",
    "block_p_desktop_exe_packaging_entry_contract_reference",
    "source_inventory_summary",
    "desktop_entrypoint_inventory_rows",
    "qml_source_inventory",
    "config_and_runtime_reference_inventory_rows",
    "python_dependency_inventory",
    "packaging_metadata_inventory",
    "existing_cli_preview_packaging_inventory",
    "artifact_exclusion_policy_inventory",
    "inventory_findings",
    "real_capability_inventory_state",
    "fail_closed_inventory_decision",
    "non_execution_inventory_evidence",
    "inventory_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
SOURCE_BOUNDARY_FIELDS_18_2: Final[list[str]] = [
    "source_block_p_desktop_exe_packaging_source_inventory",
    "source_inventory_preserved",
    "can_build_desktop_exe_packaging_inventory_matrix",
    "inventory_matrix_artifact_complete",
    "can_build_desktop_exe_packaging_contract",
    "can_feed_18_3",
]
SUMMARY_OWNED_FIELDS_18_2: Final[list[str]] = [
    "inventory_matrix_artifact_complete",
    "inventory_matrix_evaluated",
    "packaging_contract_conditions_satisfied",
    "unresolved_contract_blockers_present",
    "desktop_entrypoint_selected",
    "qml_bundle_validated",
    "build_ready",
    "packaging_authorized",
    "build_authorized",
]
FAIL_CLOSED_OWNED_FIELDS_18_2: Final[list[str]] = [
    "block_p_inventory_matrix_in_18_2",
    "block_p_packaging_contract_in_18_3",
    "only_source_only_18_3_handoff_allowed",
    "build_ready_by_18_2",
    "packaging_authorized_by_18_2",
    "build_authorized_by_18_2",
]
EXPECTED_SOURCE: Final[dict[str, Any]] = {
    "schema_version": "preview_block_p_desktop_exe_packaging_source_inventory.v1",
    "block_p_desktop_exe_packaging_source_inventory_kind": "functional_preview_block_p_desktop_exe_packaging_source_inventory",
    "block": "P",
    "step": "18.1",
    "block_p_desktop_exe_packaging_source_inventory_status": "source_18_0_consumed_block_o_closed_block_p_open_source_only_plain_data_static_inventory_current_repository_packaging_sources_inventoried_desktop_entrypoint_candidates_observed_qml_roots_and_assets_observed_dependency_declarations_observed_cli_preview_packaging_sources_observed_separately_package_discovery_observations_recorded_exclusion_policy_observed_not_applied_inventory_artifact_complete_for_18_1_no_approval_no_validation_no_packaging_no_build_no_artifact_no_release_no_runtime_no_orders_only_source_only_handoff_to_18_2_allowed",
    "block_p_desktop_exe_packaging_source_inventory_decision": "SOURCE_18_0_CONSUMED_BLOCK_O_CLOSED_BLOCK_P_OPEN_SOURCE_ONLY_PLAIN_DATA_STATIC_INVENTORY_CURRENT_REPOSITORY_PACKAGING_SOURCES_INVENTORIED_DESKTOP_ENTRYPOINT_CANDIDATES_OBSERVED_QML_ROOTS_AND_ASSETS_OBSERVED_DEPENDENCY_DECLARATIONS_OBSERVED_CLI_PREVIEW_PACKAGING_SOURCES_OBSERVED_SEPARATELY_PACKAGE_DISCOVERY_OBSERVATIONS_RECORDED_EXCLUSION_POLICY_OBSERVED_NOT_APPLIED_INVENTORY_ARTIFACT_COMPLETE_FOR_18_1_NO_APPROVAL_NO_VALIDATION_NO_PACKAGING_NO_BUILD_NO_ARTIFACT_NO_RELEASE_NO_RUNTIME_NO_ORDERS_ONLY_SOURCE_ONLY_HANDOFF_TO_18_2_ALLOWED",
    "source_inventory_artifact_complete": True,
    "ready_for_block_p_2": True,
    "next_step": "FUNCTIONAL-PREVIEW-18.2",
    "next_step_title": "BLOCK P DESKTOP EXE PACKAGING INVENTORY MATRIX",
    "block_p_desktop_exe_packaging_entry_contract_reference": {
        "schema_version": "preview_block_p_desktop_exe_packaging_entry_contract.v1",
        "block_p_desktop_exe_packaging_entry_contract_kind": "functional_preview_block_p_desktop_exe_packaging_entry_contract",
        "block": "P",
        "step": "18.0",
        "block_p_desktop_exe_packaging_entry_contract_status": "source_17_8_consumed_block_o_closed_block_p_opened_block_n_closed_block_m_preserved_desktop_exe_final_product_direction_source_only_plain_data_static_contract_cli_preview_plan_not_approved_as_final_desktop_packaging_desktop_packaging_inventory_not_performed_packaging_requirements_missing_all_real_capabilities_blocked_build_package_release_unauthorized_runtime_orders_unauthorized_future_explicit_build_execution_gate_required_only_source_only_handoff_to_18_1_allowed",
        "block_p_desktop_exe_packaging_entry_contract_decision": "SOURCE_17_8_CONSUMED_BLOCK_O_CLOSED_BLOCK_P_OPENED_BLOCK_N_CLOSED_BLOCK_M_PRESERVED_DESKTOP_EXE_FINAL_PRODUCT_DIRECTION_SOURCE_ONLY_PLAIN_DATA_STATIC_CONTRACT_CLI_PREVIEW_PLAN_NOT_APPROVED_AS_FINAL_DESKTOP_PACKAGING_DESKTOP_PACKAGING_INVENTORY_NOT_PERFORMED_PACKAGING_REQUIREMENTS_MISSING_ALL_REAL_CAPABILITIES_BLOCKED_BUILD_PACKAGE_RELEASE_UNAUTHORIZED_RUNTIME_ORDERS_UNAUTHORIZED_FUTURE_EXPLICIT_BUILD_EXECUTION_GATE_REQUIRED_ONLY_SOURCE_ONLY_HANDOFF_TO_18_1_ALLOWED",
        "block_p_opened": True,
        "ready_for_block_p_1": True,
        "next_step": "FUNCTIONAL-PREVIEW-18.1",
        "next_step_title": "BLOCK P DESKTOP EXE PACKAGING SOURCE INVENTORY",
        "status": "ready_for_functional_preview_18_1_block_p_desktop_exe_packaging_source_inventory",
        "source_block_p_desktop_exe_packaging_entry_contract_step": "FUNCTIONAL-PREVIEW-18.0",
        "source_entry_contract_read_by_18_1": True,
        "block_p_available_before_source_inventory": True,
        "static_source_inventory_only": True,
        "source_inventory_built_by_18_1": True,
        "source_inventory_artifact_complete_by_18_1": True,
        "ready_for_functional_preview_18_2": True,
        "filesystem_inventory_performed_at_runtime": False,
        "environment_inspection_performed": False,
        "secret_file_read": False,
        "dependency_import_performed": False,
        "dependency_resolution_performed": False,
        "pyside_imported": False,
        "qml_loaded": False,
        "qt_plugin_discovery_performed": False,
        "packaging_profile_validated": False,
        "build_tool_selected": False,
        "build_command_created": False,
        "build_command_executed": False,
        "packaging_performed": False,
        "artifact_created": False,
        "release_performed": False,
        "runtime_started": False,
        "orders_enabled": False,
        "network_opened": False,
        "credentials_read": False,
    },
    "source_inventory_summary": {
        "source_18_0_accepted": True,
        "block_p_open": True,
        "source_only": True,
        "plain_data": True,
        "static_inventory": True,
        "source_inventory_artifact_complete": True,
        "desktop_entrypoint_candidates_observed": True,
        "qml_sources_observed": True,
        "config_references_observed": True,
        "python_dependencies_observed": True,
        "packaging_metadata_observed": True,
        "existing_cli_preview_packaging_observed": True,
        "artifact_exclusion_policy_observed": True,
        "inventory_findings_recorded": 11,
        "inventory_validated": False,
        "desktop_entrypoint_approved": False,
        "qml_bundle_approved": False,
        "dependency_bundle_approved": False,
        "packaging_profile_approved": False,
        "build_tool_approved": False,
        "build_ready": False,
        "packaging_authorized": False,
        "build_authorized": False,
        "artifact_creation_authorized": False,
        "release_authorized": False,
        "runtime_authorized": False,
        "orders_authorized": False,
        "only_source_only_18_2_handoff_allowed": True,
        "all_real_capabilities_blocked_at_18_1": True,
    },
    "desktop_entrypoint_inventory_rows": [
        {
            "source_observed": True,
            "final_desktop_entrypoint_approved": False,
            "validation_performed": False,
            "runtime_import_performed": False,
            "inventory_result": "observed_for_18_2_matrix_only",
            "inventory_row_id": "desktop_module_launcher",
            "path": "ui/pyside_app/__main__.py",
            "source_kind": "python_module_launcher",
            "observed_role": "desktop_module_launcher",
            "observed_symbol": "ui.pyside_app.app.main",
            "desktop_entrypoint_candidate": True,
            "current_cli_preview_scope": False,
            "inventory_classification": "observed_desktop_launcher_requires_matrix_evaluation",
        },
        {
            "source_observed": True,
            "final_desktop_entrypoint_approved": False,
            "validation_performed": False,
            "runtime_import_performed": False,
            "inventory_result": "observed_for_18_2_matrix_only",
            "inventory_row_id": "desktop_application_main",
            "path": "ui/pyside_app/app.py",
            "source_kind": "python_desktop_application_source",
            "observed_role": "pyside_qml_application_entrypoint",
            "observed_symbol": "main",
            "desktop_entrypoint_candidate": True,
            "current_cli_preview_scope": False,
            "defines_app_options": True,
            "defines_bot_pyside_application": True,
            "uses_qguiapplication": True,
            "uses_qqmlapplicationengine": True,
            "default_config_path": "ui/config/example.yaml",
            "default_qml_path": "ui/pyside_app/qml/MainWindow.qml",
            "inventory_classification": "observed_desktop_application_requires_matrix_evaluation",
        },
        {
            "source_observed": True,
            "final_desktop_entrypoint_approved": False,
            "validation_performed": False,
            "runtime_import_performed": False,
            "inventory_result": "observed_for_18_2_matrix_only",
            "inventory_row_id": "cli_preview_run_local_bot",
            "path": "scripts/run_local_bot.py",
            "source_kind": "python_cli_preview_source",
            "observed_role": "cli_preview_entrypoint",
            "observed_symbol": "main",
            "desktop_entrypoint_candidate": False,
            "current_cli_preview_scope": True,
            "inventory_classification": "observed_cli_preview_source_separate_from_desktop_contract",
        },
        {
            "source_observed": True,
            "final_desktop_entrypoint_approved": False,
            "validation_performed": False,
            "runtime_import_performed": False,
            "inventory_result": "observed_for_18_2_matrix_only",
            "inventory_row_id": "cli_preview_operator_preview_bundle",
            "path": "scripts/operator_preview_bundle.py",
            "source_kind": "python_cli_preview_source",
            "observed_role": "cli_preview_operator_bundle",
            "observed_symbol": "main",
            "desktop_entrypoint_candidate": False,
            "current_cli_preview_scope": True,
            "inventory_classification": "observed_cli_preview_source_separate_from_desktop_contract",
        },
    ],
    "qml_source_inventory": {
        "default_qml_entrypoint": "ui/pyside_app/qml/MainWindow.qml",
        "pyside_qml_root": "ui/pyside_app/qml",
        "shared_qml_root": "ui/qml",
        "styles_module_qmldir": "ui/pyside_app/qml/Styles/qmldir",
        "pyside_qml_source_files": [
            "ui/pyside_app/qml/MainWindow.qml",
            "ui/pyside_app/qml/Styles/DesignSystem.qml",
            "ui/pyside_app/qml/Styles/qmldir",
            "ui/pyside_app/qml/components/IconButton.qml",
            "ui/pyside_app/qml/components/IconGlyph.qml",
            "ui/pyside_app/qml/components/PreviewCard.qml",
            "ui/pyside_app/qml/components/StyledProgressBar.qml",
            "ui/pyside_app/qml/components/StyledScrollView.qml",
            "ui/pyside_app/qml/components/StyledSpinBox.qml",
            "ui/pyside_app/qml/components/StyledSwitch.qml",
            "ui/pyside_app/qml/components/StyledTextField.qml",
            "ui/pyside_app/qml/components/layout/DockManager.qml",
            "ui/pyside_app/qml/components/layout/PanelPlaceholder.qml",
            "ui/pyside_app/qml/views/AiControlCenter.qml",
            "ui/pyside_app/qml/views/AiDecisionsView.qml",
            "ui/pyside_app/qml/views/MarketScanner.qml",
            "ui/pyside_app/qml/views/ModeWizard.qml",
            "ui/pyside_app/qml/views/OperatorDashboard.qml",
            "ui/pyside_app/qml/views/PaperTerminal.qml",
            "ui/pyside_app/qml/views/PortfolioPerformance.qml",
            "ui/pyside_app/qml/views/RiskControls.qml",
            "ui/pyside_app/qml/views/Strategies.qml",
            "ui/pyside_app/qml/views/StrategyManager.qml",
            "ui/pyside_app/qml/views/TradingUniverse.qml",
        ],
        "shared_qml_source_files": [
            "ui/qml/Icon.qml",
            "ui/qml/components/ActivationDialog.qml",
            "ui/qml/components/AdminDialog.qml",
            "ui/qml/components/AdminPanel.qml",
            "ui/qml/components/AlertCenterPanel.qml",
            "ui/qml/components/AlertToastOverlay.qml",
            "ui/qml/components/AssetHeatmapDashboard.qml",
            "ui/qml/components/BotAppWindow.qml",
            "ui/qml/components/CandlestickChartView.qml",
            "ui/qml/components/ChartWindow.qml",
            "ui/qml/components/ConfigurationWizardGallery.qml",
            "ui/qml/components/DecisionAlertLogPanel.qml",
            "ui/qml/components/EquityCurveDashboard.qml",
            "ui/qml/components/EquityCurveDashboardChart.qml",
            "ui/qml/components/FirstRunWizard.qml",
            "ui/qml/components/IconButton.qml",
            "ui/qml/components/LicenseActivationOverlay.qml",
            "ui/qml/components/LiveRiskDashboard.qml",
            "ui/qml/components/MarketMultiStreamView.qml",
            "ui/qml/components/ModuleBrowser.qml",
            "ui/qml/components/PortfolioManagerView.qml",
            "ui/qml/components/ReportBrowser.qml",
            "ui/qml/components/ResultsDashboard.qml",
            "ui/qml/components/RiskMonitorPanel.qml",
            "ui/qml/components/SidePanel.qml",
            "ui/qml/components/StatusFooter.qml",
            "ui/qml/components/UpdateManagerPanel.qml",
            "ui/qml/components/UserProfileManagementPanel.qml",
            "ui/qml/components/security/HwidManagementView.qml",
            "ui/qml/components/security/LicenseActivationView.qml",
            "ui/qml/components/security/LicenseHistoryView.qml",
            "ui/qml/components/security/LocalSecurityStore.js",
            "ui/qml/components/workbench/StrategyWorkbench.qml",
            "ui/qml/components/workbench/StrategyWorkbenchPresets.js",
            "ui/qml/components/workbench/StrategyWorkbenchUtils.js",
            "ui/qml/components/workbench/StrategyWorkbenchViewModel.qml",
            "ui/qml/components/workbench/panels/ActivityLogPanel.qml",
            "ui/qml/components/workbench/panels/AiConfigurationPanel.qml",
            "ui/qml/components/workbench/panels/AutoModePanel.qml",
            "ui/qml/components/workbench/panels/AutomationRulesPanel.qml",
            "ui/qml/components/workbench/panels/CapitalAllocationPanel.qml",
            "ui/qml/components/workbench/panels/ComplianceOverviewPanel.qml",
            "ui/qml/components/workbench/panels/ExchangeManagementPanel.qml",
            "ui/qml/components/workbench/panels/ExecutionDiagnosticsPanel.qml",
            "ui/qml/components/workbench/panels/InstrumentOverviewPanel.qml",
            "ui/qml/components/workbench/panels/LicenseStatusPanel.qml",
            "ui/qml/components/workbench/panels/MarketSentimentPanel.qml",
            "ui/qml/components/workbench/panels/NewsFeedPanel.qml",
            "ui/qml/components/workbench/panels/OpenPositionsPanel.qml",
            "ui/qml/components/workbench/panels/PendingOrdersPanel.qml",
            "ui/qml/components/workbench/panels/PerformanceComparisonPanel.qml",
            "ui/qml/components/workbench/panels/ResultsAnalysisPanel.qml",
            "ui/qml/components/workbench/panels/RiskTimelinePanel.qml",
            "ui/qml/components/workbench/panels/RuntimeStatusPanel.qml",
            "ui/qml/components/workbench/panels/ScenarioTestingPanel.qml",
            "ui/qml/components/workbench/panels/SignalAlertsPanel.qml",
            "ui/qml/components/workbench/panels/StrategyControlPanel.qml",
            "ui/qml/components/workbench/panels/StrategyDashboardPanel.qml",
            "ui/qml/components/workbench/panels/TradeHistoryPanel.qml",
            "ui/qml/dashboard/CompliancePanel.qml",
            "ui/qml/dashboard/RiskJournalPanel.qml",
            "ui/qml/dashboard/RunbookPanel.qml",
            "ui/qml/dashboard/RuntimeOverview.qml",
            "ui/qml/dashboard/StrategyOverviewPanel.qml",
            "ui/qml/design-system/FrostedGlass.qml",
            "ui/qml/design-system/Icon.qml",
            "ui/qml/design-system/Palette.qml",
            "ui/qml/design-system/ResponsiveGrid.qml",
            "ui/qml/design-system/StrategyAiPanel.qml",
            "ui/qml/design-system/Typography.qml",
            "ui/qml/design-system/charts/HeatmapChart.qml",
            "ui/qml/design-system/charts/PnlChart.qml",
            "ui/qml/design-system/components/Card.qml",
            "ui/qml/design-system/components/MetricTile.qml",
            "ui/qml/design-system/fonts/FontAwesomeData.js",
            "ui/qml/design-system/qmldir",
            "ui/qml/main.qml",
            "ui/qml/onboarding/DecisionLogSetupStep.qml",
            "ui/qml/onboarding/LicenseWizard.qml",
            "ui/qml/onboarding/StrategySetupStep.qml",
            "ui/qml/settings/DashboardSettings.qml",
            "ui/qml/settings/PrivacySettings.qml",
            "ui/qml/settings/ThemePersonalization.qml",
            "ui/qml/settings/UpdateDialog.qml",
            "ui/qml/styles/AppTheme.qml",
            "ui/qml/styles/qmldir",
            "ui/qml/support/SupportCenter.qml",
            "ui/qml/support/TicketDialog.qml",
            "ui/qml/views/AiDecisionHistory.qml",
            "ui/qml/views/AnalyticsDashboard.qml",
            "ui/qml/views/AnalyticsUtils.js",
            "ui/qml/views/DecisionMonitoringView.qml",
            "ui/qml/views/HypercareDashboard.qml",
            "ui/qml/views/LicenseAuditView.qml",
            "ui/qml/views/Marketplace.qml",
            "ui/qml/views/MonitoringDashboard.qml",
            "ui/qml/views/OperationsCenter.qml",
            "ui/qml/views/PortfolioDashboard.qml",
            "ui/qml/views/RiskControls.qml",
            "ui/qml/views/SetupWizard.qml",
            "ui/qml/views/StrategyBuilder.qml",
            "ui/qml/views/StrategyBuilderUtils.js",
            "ui/qml/views/StrategyConfigurator.qml",
            "ui/qml/views/StrategyExperience.qml",
            "ui/qml/views/StrategyManagement.qml",
            "ui/qml/views/StrategyManager.qml",
            "ui/qml/views/StrategyRiskConsole.qml",
        ],
        "qml_support_files": [],
        "pyside_qml_source_file_count": 24,
        "shared_qml_source_file_count": 107,
        "qml_support_file_count": 0,
        "all_qml_inventory_paths_unique": True,
        "main_window_imports": [
            "QtQuick",
            "QtQuick.Controls",
            "QtQuick.Layouts",
            "QtQuick.Effects",
            "components",
            "components/layout",
            "Styles 1.0",
            "views",
        ],
        "styles_module_observation": {
            "module": "Styles",
            "type_name": "DesignSystem",
            "version": "1.0",
            "source": "DesignSystem.qml",
        },
        "windows_shared_qml_import_path_observation": {
            "shared_qml_root_added_on_non_windows": True,
            "shared_qml_root_added_on_windows": False,
            "observation_requires_18_2_evaluation": True,
            "observation_is_validation_failure_in_18_1": False,
        },
        "qml_inventory_observed": True,
        "qml_inventory_validated": False,
        "qml_bundle_approved": False,
    },
    "config_and_runtime_reference_inventory_rows": [
        {
            "inventory_row_id": "example_config",
            "path": "ui/config/example.yaml",
            "source_kind": "example_ui_config",
            "source_observed": True,
            "bundle_candidate": "requires_18_2_evaluation",
            "approved_for_bundle": False,
        },
        {
            "inventory_row_id": "sample_dataset",
            "source_kind": "referenced_local_sample_dataset",
            "source_path": "data/sample_ohlcv/trend.csv",
            "existence_observed_in_repo": True,
            "approved_for_bundle": False,
        },
        {
            "inventory_row_id": "grpc_tls_root_cert",
            "field_path": "grpc.tls.root_cert",
            "source_kind": "local_secret_file_reference",
            "secret_content_read": False,
            "existence_checked": False,
            "bundle_allowed": False,
            "requires_exclusion": True,
        },
        {
            "inventory_row_id": "grpc_tls_client_cert",
            "field_path": "grpc.tls.client_cert",
            "source_kind": "local_secret_file_reference",
            "secret_content_read": False,
            "existence_checked": False,
            "bundle_allowed": False,
            "requires_exclusion": True,
        },
        {
            "inventory_row_id": "grpc_tls_client_key",
            "field_path": "grpc.tls.client_key",
            "source_kind": "local_secret_file_reference",
            "secret_content_read": False,
            "existence_checked": False,
            "bundle_allowed": False,
            "requires_exclusion": True,
        },
        {
            "inventory_row_id": "telemetry_metrics_auth_token",
            "source_kind": "example_config_sensitive_field_reference",
            "field_path": "telemetry.metrics_auth_token",
            "value_copied_to_inventory": False,
            "bundle_allowed": False,
            "requires_18_2_policy_evaluation": True,
        },
    ],
    "python_dependency_inventory": {
        "project_dependency_specs": [
            "pyarrow>=21.0.0",
            "protobuf>=4.24,<6",
            "grpcio>=1.62,<2",
            "PyYAML>=6.0",
            "cryptography>=48.0.1,<49",
            "PyNaCl>=1.5.0,<2",
            "pydantic>=2.5",
            "numpy>=1.26",
            "pandas>=2.2",
            "joblib>=1.3",
            "scipy>=1.16",
            "lightgbm>=4.6",
            "c" + "cxt>=4.0.0,<5",
            "key" + "ring>=24.3",
            "secretstorage>=3.3; platform_system == 'Linux'",
            "requests>=2.31",
            "responses>=0.25",
            "httpx>=0.26",
            "anyio>=4.0",
            "aiosqlite>=0.20",
            "respx>=0.20",
            "jsonschema>=4.21",
            "sqlalchemy[asyncio]>=2.0",
            "PySide6>=6.7,<6.11",
            "packaging>=23.2",
        ],
        "desktop_optional_dependency_specs": [
            "pyinstaller>=6.5",
            "briefcase>=0.3.18",
            "PySide6>=6.7,<6.11",
        ],
        "project_dependency_count": 25,
        "desktop_optional_dependency_count": 3,
        "pyside6_declared_in_project_dependencies": True,
        "pyside6_declared_in_desktop_optional_dependencies": True,
        "pyinstaller_declared_in_desktop_optional_dependencies": True,
        "briefcase_declared_in_desktop_optional_dependencies": True,
        "dependency_resolution_performed": False,
        "dependencies_imported": False,
        "dependencies_validated": False,
        "dependency_bundle_approved": False,
    },
    "packaging_metadata_inventory": {
        "setuptools_package_include_patterns": [
            "bot_core*",
            "core*",
            "scripts*",
            "stage6_samples*",
        ],
        "setuptools_package_data_keys": [
            "bot_core.auto_trader",
            "bot_core.ai",
            "bot_core.ai._defaults",
            "bot_core.risk",
            "bot_core.execution",
            "bot_core.product",
            "core",
        ],
        "ui_package_discovery_pattern_present": False,
        "qml_package_data_declaration_present": False,
        "requires_18_2_evaluation": True,
        "treated_as_build_failure_by_18_1": False,
        "packaging_metadata_observed": True,
        "packaging_metadata_validated": False,
        "deploy_packaging_source_files": [
            "deploy/packaging/README.md",
            "deploy/packaging/__init__.py",
            "deploy/packaging/_toml_compat.py",
            "deploy/packaging/assets/README.md",
            "deploy/packaging/assets/demo/config/core.yaml",
            "deploy/packaging/assets/demo/daemon/sample_daemon.txt",
            "deploy/packaging/assets/demo/reports/README.txt",
            "deploy/packaging/assets/demo/reports/champion_overview.json",
            "deploy/packaging/assets/demo/resources/extras/readme.txt",
            "deploy/packaging/assets/demo/signing.key",
            "deploy/packaging/assets/demo/ui/sample_ui.txt",
            "deploy/packaging/assets/demo/var/models/quality/README.txt",
            "deploy/packaging/assets/demo/var/models/quality/sample_model/challengers.json",
            "deploy/packaging/assets/demo/var/models/quality/sample_model/champion.json",
            "deploy/packaging/assets/demo/var/models/quality/sample_model/latest.json",
            "deploy/packaging/assets/demo/wheels/README.txt",
            "deploy/packaging/assets/demo/wheels/aiohttp-3.9.5-cp311-cp311-manylinux.whl",
            "deploy/packaging/assets/demo/wheels/c" + "cxt-4.0.0-py3-none-any.whl",
            "deploy/packaging/assets/demo/wheels/lightgbm-4.6.0-cp311-cp311-manylinux.whl",
            "deploy/packaging/assets/prod/.gitkeep",
            "deploy/packaging/assets/prod/config/.gitkeep",
            "deploy/packaging/assets/prod/daemon/.gitkeep",
            "deploy/packaging/assets/prod/reports/.gitkeep",
            "deploy/packaging/assets/prod/resources/.gitkeep",
            "deploy/packaging/assets/prod/ui/.gitkeep",
            "deploy/packaging/assets/prod/var/.gitkeep",
            "deploy/packaging/assets/prod/wheels/.gitkeep",
            "deploy/packaging/build_core_bundle.py",
            "deploy/packaging/build_pyinstaller_bundle.py",
            "deploy/packaging/build_strategy_bundle.py",
            "deploy/packaging/desktop_installer.py",
            "deploy/packaging/installer_profile.py",
            "deploy/packaging/offline_distribution.py",
            "deploy/packaging/offline_package.py",
            "deploy/packaging/pipeline.py",
            "deploy/packaging/profiles/demo/linux.toml",
            "deploy/packaging/profiles/demo/macos.toml",
            "deploy/packaging/profiles/demo/windows.toml",
            "deploy/packaging/profiles/linux.toml",
            "deploy/packaging/profiles/macos.toml",
            "deploy/packaging/profiles/preview/linux.toml",
            "deploy/packaging/profiles/preview/macos.toml",
            "deploy/packaging/profiles/preview/windows.toml",
            "deploy/packaging/profiles/windows.toml",
            "deploy/packaging/requirements-desktop.lock",
            "deploy/packaging/requirements-desktop.txt",
        ],
        "deployment_documentation_files": [
            "docs/deployment/bitmex_go_live.md",
            "docs/deployment/bybit_go_live.md",
            "docs/deployment/deribit_go_live.md",
            "docs/deployment/desktop_install.md",
            "docs/deployment/desktop_installer.md",
            "docs/deployment/desktop_offline_dependencies.md",
            "docs/deployment/installer_automation.md",
            "docs/deployment/installer_build.md",
            "docs/deployment/kraken_go_live.md",
            "docs/deployment/oem_installation.md",
            "docs/deployment/oem_runtime_operations.md",
            "docs/deployment/offline_packaging.md",
            "docs/deployment/okx_go_live.md",
            "docs/deployment/qa_checklist.md",
            "docs/deployment/recovery.md",
            "docs/deployment/sandbox_proof_pack.md",
            "docs/deployment/update_runbook.md",
        ],
        "deploy_packaging_source_file_count": 46,
        "deployment_documentation_file_count": 17,
        "all_deployment_inventory_paths_unique": True,
    },
    "existing_cli_preview_packaging_inventory": {
        "current_preview_packaging_scope": "cli_preview",
        "final_desktop_packaging_contract": False,
        "usable_as_final_desktop_build_profile": False,
        "requires_18_2_evaluation": True,
        "rows": [
            {
                "path": "scripts/safe_exe_preview_build_plan.py",
                "current_plan_entrypoint": "scripts/run_local_bot.py",
                "current_optional_entrypoint": "scripts/operator_preview_bundle.py",
                "current_artifact_type": "cli-preview-exe",
                "current_artifact_name": "dudzian-bot-preview",
                "current_selected_build_tool": "pyinstaller",
                "build_command_execution_allowed": False,
                "build_command_executed": False,
            },
            {
                "path": "deploy/packaging/profiles/preview/windows.toml",
                "profile_platform": "windows",
                "profile_pyinstaller_entrypoint_targets_run_local_bot": True,
                "profile_targets_desktop_pyside_entrypoint": False,
                "profile_runtime_name": "dudzian-bot-preview",
                "profile_has_briefcase_section": True,
                "profile_validated_for_final_desktop_product": False,
            },
            {
                "path": "deploy/packaging/profiles/preview/linux.toml",
                "profile_platform": "linux",
                "profile_pyinstaller_entrypoint_targets_run_local_bot": True,
                "profile_targets_desktop_pyside_entrypoint": False,
                "profile_runtime_name": "dudzian-bot-preview",
                "profile_has_briefcase_section": True,
                "profile_validated_for_final_desktop_product": False,
            },
            {
                "path": "deploy/packaging/profiles/preview/macos.toml",
                "profile_platform": "macos",
                "profile_pyinstaller_entrypoint_targets_run_local_bot": True,
                "profile_targets_desktop_pyside_entrypoint": False,
                "profile_runtime_name": "dudzian-bot-preview",
                "profile_has_briefcase_section": True,
                "profile_validated_for_final_desktop_product": False,
            },
        ],
    },
    "artifact_exclusion_policy_inventory": {
        "policy_source": "scripts/safe_exe_preview_build_plan.py",
        "policy_version": "security_packaging_artifact_policy.v1",
        "denied_artifact_patterns": [
            ".env",
            "*.env",
            "trading.db",
            "bot_core/logs",
            "logs",
            "reports",
            "test-results",
            "var/security",
            "*api_key*",
            "*api_secret*",
            "*secret*",
            "*token*",
            "*keychain*",
        ],
        "policy_observed": True,
        "policy_applied_by_18_1": False,
        "artifact_scanned_by_18_1": False,
        "secret_files_read_by_18_1": False,
        "policy_validated_for_desktop_bundle": False,
    },
    "inventory_findings": [
        {
            "finding_id": "desktop_module_launcher_observed",
            "observation_present": True,
            "source_paths": ["ui/pyside_app/__main__.py"],
            "requires_18_2_evaluation": True,
            "validated": False,
            "approved": False,
            "severity_classification": "inventory_observation",
            "inventory_result": "inventory_observation",
        },
        {
            "finding_id": "desktop_application_main_observed",
            "observation_present": True,
            "source_paths": ["ui/pyside_app/app.py"],
            "requires_18_2_evaluation": True,
            "validated": False,
            "approved": False,
            "severity_classification": "inventory_observation",
            "inventory_result": "inventory_observation",
        },
        {
            "finding_id": "default_qml_entrypoint_observed",
            "observation_present": True,
            "source_paths": ["ui/pyside_app/qml/MainWindow.qml"],
            "requires_18_2_evaluation": True,
            "validated": False,
            "approved": False,
            "severity_classification": "inventory_observation",
            "inventory_result": "inventory_observation",
        },
        {
            "finding_id": "two_qml_source_roots_observed",
            "observation_present": True,
            "source_paths": ["ui/pyside_app/qml", "ui/qml"],
            "requires_18_2_evaluation": True,
            "validated": False,
            "approved": False,
            "severity_classification": "inventory_observation",
            "inventory_result": "inventory_observation",
        },
        {
            "finding_id": "shared_qml_import_path_is_platform_conditional",
            "observation_present": True,
            "source_paths": ["ui/pyside_app/app.py", "ui/pyside_app/qml/MainWindow.qml", "ui/qml"],
            "requires_18_2_evaluation": True,
            "validated": False,
            "approved": False,
            "severity_classification": "inventory_observation",
            "inventory_result": "inventory_observation",
        },
        {
            "finding_id": "cli_preview_plan_targets_non_desktop_entrypoint",
            "observation_present": True,
            "source_paths": [
                "scripts/safe_exe_preview_build_plan.py",
                "deploy/packaging/profiles/preview/windows.toml",
                "deploy/packaging/profiles/preview/linux.toml",
                "deploy/packaging/profiles/preview/macos.toml",
            ],
            "requires_18_2_evaluation": True,
            "validated": False,
            "approved": False,
            "severity_classification": "inventory_observation",
            "inventory_result": "inventory_observation",
        },
        {
            "finding_id": "desktop_build_tools_declared_as_optional_dependencies",
            "observation_present": True,
            "source_paths": ["pyproject.toml"],
            "requires_18_2_evaluation": True,
            "validated": False,
            "approved": False,
            "severity_classification": "inventory_observation",
            "inventory_result": "inventory_observation",
        },
        {
            "finding_id": "ui_package_discovery_not_declared_in_current_setuptools_include",
            "observation_present": True,
            "source_paths": ["pyproject.toml"],
            "requires_18_2_evaluation": True,
            "validated": False,
            "approved": False,
            "severity_classification": "inventory_observation",
            "inventory_result": "inventory_observation",
        },
        {
            "finding_id": "qml_package_data_not_declared_in_current_setuptools_metadata",
            "observation_present": True,
            "source_paths": ["pyproject.toml"],
            "requires_18_2_evaluation": True,
            "validated": False,
            "approved": False,
            "severity_classification": "inventory_observation",
            "inventory_result": "inventory_observation",
        },
        {
            "finding_id": "example_config_references_local_secret_paths",
            "observation_present": True,
            "source_paths": ["ui/config/example.yaml"],
            "requires_18_2_evaluation": True,
            "validated": False,
            "approved": False,
            "severity_classification": "inventory_observation",
            "inventory_result": "inventory_observation",
        },
        {
            "finding_id": "example_config_contains_sensitive_field_reference",
            "observation_present": True,
            "source_paths": ["ui/config/example.yaml"],
            "requires_18_2_evaluation": True,
            "validated": False,
            "approved": False,
            "severity_classification": "inventory_observation",
            "inventory_result": "inventory_observation",
        },
    ],
    "real_capability_inventory_state": {
        "inherited_18_0_capabilities": {
            "inherited_block_o_capabilities": {
                "release_execution": "blocked",
                "release_publish": "blocked",
                "release_sign": "blocked",
                "release_smoke": "blocked",
                "release_workflow": "blocked",
                "release_notes": "blocked",
                "release_tag": "blocked",
                "release_upload": "blocked",
                "release_export": "blocked",
                "artifact_creation": "blocked",
                "artifact_mutation": "blocked",
                "artifact_deletion": "blocked",
                "artifact_smoke": "blocked",
                "artifact_sign": "blocked",
                "artifact_publish": "blocked",
                "artifact_name": "blocked",
                "artifact_location": "blocked",
                "artifact_checksum": "blocked",
                "artifact_metadata": "blocked",
                "artifact_audit": "blocked",
                "artifact_cleanup": "blocked",
                "packaging_dry_run": "blocked",
                "packaging": "blocked",
                "pyinstaller": "blocked",
                "build": "blocked",
                "build_artifact": "blocked",
                "installer": "blocked",
                "workflow": "blocked",
                "environment": "blocked",
                "dependency": "blocked",
                "asset": "blocked",
                "qml_asset": "blocked",
                "filesystem": "blocked",
                "gate_evaluation": "blocked",
                "gate_condition": "blocked",
                "gate_opening": "blocked",
                "gate_mutation": "blocked",
                "confirmation_acceptance": "blocked",
                "environment_validation": "blocked",
                "artifact_validation": "blocked",
                "release_validation": "blocked",
                "runtime_validation": "blocked",
                "credentials_validation": "blocked",
                "dependency_validation": "blocked",
                "runtime_activation": "blocked",
                "paper_runtime": "blocked",
                "testnet_runtime": "blocked",
                "live_canary": "blocked",
                "live_trading": "blocked",
                "runtime_loop": "blocked",
                "runtime_gates": "blocked",
                "order_generation": "blocked",
                "create_" + "order": "blocked",
                "submit_order": "blocked",
                "cancel_order": "blocked",
                "replace_order": "blocked",
                "fetch_" + "balance": "blocked",
                "private_endpoint": "blocked",
                "network": "blocked",
                "credentials": "blocked",
                "config_env_secrets": "blocked",
                "qml_bridge": "blocked",
                "c" + "cxt": "blocked",
            },
            "block_p_capabilities": {
                "desktop_entrypoint_selection": "blocked",
                "desktop_entrypoint_validation": "blocked",
                "qml_asset_inventory": "blocked",
                "qml_asset_validation": "blocked",
                "qt_plugin_inventory": "blocked",
                "qt_plugin_validation": "blocked",
                "python_dependency_inventory": "blocked",
                "python_dependency_validation": "blocked",
                "packaging_profile_selection": "blocked",
                "packaging_profile_validation": "blocked",
                "secret_exclusion_validation": "blocked",
                "local_data_exclusion_validation": "blocked",
                "windows_toolchain_selection": "blocked",
                "windows_toolchain_validation": "blocked",
                "pyinstaller_configuration": "blocked",
                "pyinstaller_execution": "blocked",
                "build_command_creation": "blocked",
                "build_command_execution": "blocked",
                "exe_artifact_creation": "blocked",
                "artifact_scan": "blocked",
                "artifact_hash_manifest": "blocked",
                "artifact_signing": "blocked",
                "installer_creation": "blocked",
                "release_creation": "blocked",
                "release_upload": "blocked",
                "runtime_activation": "blocked",
                "order_activity": "blocked",
            },
        },
        "inventory_capabilities": {
            "runtime_filesystem_inventory": "blocked",
            "home_directory_scan": "blocked",
            "secret_directory_scan": "blocked",
            "secret_file_read": "blocked",
            "dependency_import": "blocked",
            "dependency_resolution": "blocked",
            "qml_load": "blocked",
            "qml_runtime_validation": "blocked",
            "qt_plugin_discovery": "blocked",
            "qt_plugin_validation": "blocked",
            "packaging_profile_validation": "blocked",
            "build_tool_selection": "blocked",
            "build_tool_execution": "blocked",
            "spec_file_creation": "blocked",
            "build_command_creation": "blocked",
            "build_command_execution": "blocked",
            "exe_artifact_creation": "blocked",
            "artifact_scan": "blocked",
            "artifact_signing": "blocked",
            "installer_creation": "blocked",
            "release_creation": "blocked",
            "runtime_activation": "blocked",
            "order_activity": "blocked",
        },
        "inherited_block_o_capabilities_known_blocked": True,
        "inherited_block_p_capabilities_known_blocked": True,
        "inherited_18_0_capabilities_known_blocked": True,
        "inventory_capabilities_known_blocked": True,
        "all_real_capabilities_blocked_at_18_1": True,
    },
    "fail_closed_inventory_decision": {
        "block_p_entry_contract_in_18_0": "preserved",
        "block_p_source_inventory_in_18_1": "complete",
        "block_p_inventory_matrix_in_18_2": "allowed",
        "only_source_only_18_2_handoff_allowed": True,
        "desktop_entrypoint_approved_by_18_1": False,
        "qml_bundle_approved_by_18_1": False,
        "dependency_bundle_approved_by_18_1": False,
        "packaging_profile_approved_by_18_1": False,
        "build_tool_approved_by_18_1": False,
        "build_ready_by_18_1": False,
        "packaging_authorized_by_18_1": False,
        "build_authorized_by_18_1": False,
        "build_executed_by_18_1": False,
        "artifact_created_by_18_1": False,
        "release_authorized_by_18_1": False,
        "runtime_enabled_by_18_1": False,
        "orders_enabled_by_18_1": False,
    },
    "non_execution_inventory_evidence": {
        "source_builder_called": True,
        "source_accepted": True,
        "identity_valid": True,
        "block_o_closure_reference_valid": True,
        "summary_valid": True,
        "inherited_closure_valid": True,
        "product_direction_valid": True,
        "scope_rows_valid": True,
        "requirement_rows_valid": True,
        "real_capability_valid": True,
        "fail_closed_valid": True,
        "evidence_valid": True,
        "entry_boundaries_valid": True,
        "source_boundaries_valid": True,
        "future_steps_valid": True,
        "inventory_constants_populated": True,
        "inventory_path_uniqueness": True,
        "inventory_artifact_complete": True,
        "all_operational_by_18_1_flags_false": True,
    },
    "inventory_boundaries": {
        "reads_18_0_only": True,
        "production_runtime_filesystem_scan": False,
        "production_environment_scan": False,
        "home_directory_scan": False,
        "secret_directory_scan": False,
        "secret_file_read": False,
        "dependency_import": False,
        "dependency_resolution": False,
        "pyside_import": False,
        "qml_load": False,
        "qt_plugin_discovery": False,
        "packaging_profile_validation": False,
        "build_tool_selection": False,
        "build_tool_execution": False,
        "packaging_performed": False,
        "artifact_created": False,
        "artifact_signed": False,
        "installer_created": False,
        "release_performed": False,
        "runtime_started": False,
        "orders_enabled": False,
        "network_opened": False,
        "credentials_read": False,
        "qml_bridge_gateway_controller_changed": False,
        "can_feed_only_18_2_inventory_matrix": True,
    },
    "source_boundaries": {
        "inherited_source_boundaries": {
            "inherited_source_boundaries": {
                "allowed_imports_only": True,
                "source_block_n_safety_gate_readiness_contract": "FUNCTIONAL-PREVIEW-16.6",
                "source_block_n_safety_gate_readiness_matrix": "FUNCTIONAL-PREVIEW-16.5",
                "source_block_n_safety_gate_read_model": "FUNCTIONAL-PREVIEW-16.4",
                "source_block_n_safety_gate_readiness_contract_boundaries": {
                    "allowed_imports_only": True,
                    "source_block_n_safety_gate_readiness_matrix": "FUNCTIONAL-PREVIEW-16.5",
                    "source_block_n_safety_gate_read_model": "FUNCTIONAL-PREVIEW-16.4",
                    "plain_data_source_only": True,
                    "static_non_evaluating": True,
                    "non_mutating": True,
                    "non_authorizing": True,
                    "can_feed_16_7": True,
                    "can_feed_16_8": True,
                },
                "forbidden_packaging_calls_present": False,
                "forbidden_pyinstaller_calls_present": False,
                "forbidden_build_calls_present": False,
                "forbidden_release_calls_present": False,
                "forbidden_runtime_calls_present": False,
                "forbidden_gate_evaluation_calls_present": False,
                "forbidden_gate_execution_calls_present": False,
                "forbidden_gate_mutation_calls_present": False,
                "forbidden_validation_calls_present": False,
                "forbidden_confirmation_calls_present": False,
                "forbidden_authorization_calls_present": False,
                "forbidden_readiness_recalculation_calls_present": False,
                "forbidden_io_calls_present": False,
                "forbidden_network_calls_present": False,
                "forbidden_private_endpoint_calls_present": False,
                "forbidden_ui_bridge_calls_present": False,
                "source_block_n_safety_gate_readiness_read_model": "FUNCTIONAL-PREVIEW-16.7",
                "can_feed_16_8": True,
                "can_close_block_n": True,
                "can_feed_17_0": True,
                "forbidden_git_calls_present": False,
                "source_block_n_closure_audit": "FUNCTIONAL-PREVIEW-16.8",
                "block_n_closure_audit_source_preserved": True,
                "can_open_block_o": True,
                "can_feed_17_1": True,
                "source_block_o_entry_contract": "FUNCTIONAL-PREVIEW-17.0",
                "block_o_entry_contract_source_preserved": True,
                "can_build_block_o_read_model": True,
                "can_feed_17_2": True,
                "source_block_o_read_model": "FUNCTIONAL-PREVIEW-17.1",
                "block_o_read_model_source_preserved": True,
                "can_build_execution_authorization_matrix": True,
                "can_feed_17_3": True,
                "source_block_o_execution_authorization_matrix": "FUNCTIONAL-PREVIEW-17.2",
                "matrix_source_preserved": True,
                "can_build_execution_authorization_contract": True,
                "can_feed_17_4": True,
                "source_block_o_execution_authorization_contract": "FUNCTIONAL-PREVIEW-17.3",
                "contract_source_preserved": True,
                "can_build_execution_authorization_read_model": True,
                "can_feed_17_5": True,
                "source_block_o_execution_authorization_read_model": "FUNCTIONAL-PREVIEW-17.4",
                "read_model_source_preserved": True,
                "can_build_execution_authorization_readiness_matrix": True,
                "can_feed_17_6": True,
                "source_block_o_execution_authorization_readiness_matrix": "FUNCTIONAL-PREVIEW-17.5",
                "readiness_matrix_source_preserved": True,
                "can_build_execution_authorization_readiness_contract": True,
                "can_feed_17_7": True,
                "source_block_o_execution_authorization_readiness_contract": "FUNCTIONAL-PREVIEW-17.6",
                "readiness_contract_source_preserved": True,
                "can_build_execution_authorization_readiness_read_model": True,
                "can_feed_17_8": True,
                "source_block_o_execution_authorization_readiness_read_model": "FUNCTIONAL-PREVIEW-17.7",
                "readiness_read_model_source_preserved": True,
                "can_close_block_o": True,
                "block_o_closed": True,
                "can_feed_next_explicit_block": True,
            },
            "source_block_o_closure_audit": True,
            "block_o_closure_audit_source_preserved": True,
            "can_open_block_p": True,
            "block_p_opened": True,
            "can_build_desktop_exe_packaging_source_inventory": True,
            "can_feed_18_1": True,
        },
        "source_block_p_desktop_exe_packaging_entry_contract": True,
        "block_p_entry_contract_source_preserved": True,
        "can_build_desktop_exe_packaging_source_inventory": True,
        "source_inventory_artifact_complete": True,
        "can_build_desktop_exe_packaging_inventory_matrix": True,
        "can_feed_18_2": True,
    },
    "future_steps": [
        {
            "step": "18.2",
            "title": "BLOCK P DESKTOP EXE PACKAGING INVENTORY MATRIX",
            "source_only": True,
            "build_performed": False,
        },
        {
            "step": "18.3",
            "title": "BLOCK P DESKTOP EXE PACKAGING CONTRACT",
            "source_only": True,
            "build_performed": False,
        },
        {
            "step": "18.4",
            "title": "BLOCK P DESKTOP EXE PACKAGING READ MODEL",
            "source_only": True,
            "build_performed": False,
        },
        {
            "step": "18.5",
            "title": "BLOCK P DESKTOP EXE BUILD READINESS MATRIX",
            "source_only": True,
            "build_performed": False,
        },
        {
            "step": "18.6",
            "title": "BLOCK P DESKTOP EXE BUILD READINESS CONTRACT",
            "source_only": True,
            "build_performed": False,
        },
        {
            "step": "18.7",
            "title": "BLOCK P DESKTOP EXE BUILD READINESS READ MODEL",
            "source_only": True,
            "build_performed": False,
        },
        {
            "step": "18.8",
            "title": "BLOCK P CLOSURE AUDIT",
            "source_only": True,
            "build_performed": False,
        },
    ],
    "status": "ready_for_functional_preview_18_2_block_p_desktop_exe_packaging_inventory_matrix",
}
SOURCE_IDENTITY_EXPECTED: Final[dict[str, Any]] = {
    "schema_version": "preview_block_p_desktop_exe_packaging_source_inventory.v1",
    "block_p_desktop_exe_packaging_source_inventory_kind": "functional_preview_block_p_desktop_exe_packaging_source_inventory",
    "block": "P",
    "step": "18.1",
    "block_p_desktop_exe_packaging_source_inventory_status": "source_18_0_consumed_block_o_closed_block_p_open_source_only_plain_data_static_inventory_current_repository_packaging_sources_inventoried_desktop_entrypoint_candidates_observed_qml_roots_and_assets_observed_dependency_declarations_observed_cli_preview_packaging_sources_observed_separately_package_discovery_observations_recorded_exclusion_policy_observed_not_applied_inventory_artifact_complete_for_18_1_no_approval_no_validation_no_packaging_no_build_no_artifact_no_release_no_runtime_no_orders_only_source_only_handoff_to_18_2_allowed",
    "block_p_desktop_exe_packaging_source_inventory_decision": "SOURCE_18_0_CONSUMED_BLOCK_O_CLOSED_BLOCK_P_OPEN_SOURCE_ONLY_PLAIN_DATA_STATIC_INVENTORY_CURRENT_REPOSITORY_PACKAGING_SOURCES_INVENTORIED_DESKTOP_ENTRYPOINT_CANDIDATES_OBSERVED_QML_ROOTS_AND_ASSETS_OBSERVED_DEPENDENCY_DECLARATIONS_OBSERVED_CLI_PREVIEW_PACKAGING_SOURCES_OBSERVED_SEPARATELY_PACKAGE_DISCOVERY_OBSERVATIONS_RECORDED_EXCLUSION_POLICY_OBSERVED_NOT_APPLIED_INVENTORY_ARTIFACT_COMPLETE_FOR_18_1_NO_APPROVAL_NO_VALIDATION_NO_PACKAGING_NO_BUILD_NO_ARTIFACT_NO_RELEASE_NO_RUNTIME_NO_ORDERS_ONLY_SOURCE_ONLY_HANDOFF_TO_18_2_ALLOWED",
    "source_inventory_artifact_complete": True,
    "ready_for_block_p_2": True,
    "next_step": "FUNCTIONAL-PREVIEW-18.2",
    "next_step_title": "BLOCK P DESKTOP EXE PACKAGING INVENTORY MATRIX",
    "status": "ready_for_functional_preview_18_2_block_p_desktop_exe_packaging_inventory_matrix",
}


def _copy_plain(value: Any) -> Any:
    if type(value) is dict:
        return {key: _copy_plain(item) for key, item in value.items()}
    if type(value) is list:
        return [_copy_plain(item) for item in value]
    return value


def _all_plain_json(value: Any, max_depth: int) -> bool:
    stack: list[tuple[Any, int, bool]] = [(value, 0, False)]
    active: set[int] = set()
    while stack:
        item, depth, leaving = stack.pop()
        item_type = type(item)
        if item_type in (str, int, bool) or item is None:
            continue
        if item_type not in (dict, list) or depth > max_depth:
            return False
        item_id = id(item)
        if leaving:
            active.discard(item_id)
            continue
        if item_id in active:
            return False
        active.add(item_id)
        stack.append((item, depth, True))
        if item_type is dict:
            for key, child in item.items():
                if type(key) is not str:
                    return False
                stack.append((child, depth + 1, False))
        else:
            for child in item:
                stack.append((child, depth + 1, False))
    return True


def _exact_plain_matches(value: Any, expected: Any) -> bool:
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        if list(value.keys()) != list(expected.keys()):
            return False
        for key in expected:
            if not _exact_plain_matches(value[key], expected[key]):
                return False
        return True
    if type(expected) is list:
        return len(value) == len(expected) and all(
            _exact_plain_matches(a, b) for a, b in zip(value, expected, strict=True)
        )
    return value == expected


def _plain_dict_section(source: dict[str, Any], key: str) -> dict[str, Any]:
    value = source.get(key)
    return value if type(value) is dict else {}


def _plain_list_section(source: dict[str, Any], key: str) -> list[Any]:
    value = source.get(key)
    return value if type(value) is list else []


def _section_valid(section: Any, expected: Any) -> bool:
    return _all_plain_json(section, MAX_DIAGNOSTIC_CONTAINER_DEPTH) and _exact_plain_matches(
        section, expected
    )


def _safe_top_level_source(raw_source: dict[Any, Any]) -> tuple[dict[str, Any], bool]:
    safe: dict[str, Any] = {}
    ok = True
    for key, value in raw_source.items():
        if type(key) is str:
            safe[key] = value
        else:
            ok = False
    return safe, ok


def _owned_fields_are_unshadowed(
    section: dict[Any, Any], owned_fields: list[str], expected: dict[str, Any]
) -> bool:
    for key, value in section.items():
        if type(key) is not str or key not in owned_fields:
            continue
        if key not in expected or not _section_valid(value, expected[key]):
            return False
    return True


def _no_shadowing(source: dict[str, Any]) -> bool:
    return (
        _owned_fields_are_unshadowed(
            _plain_dict_section(source, "source_inventory_summary"),
            SUMMARY_OWNED_FIELDS_18_2,
            EXPECTED_SOURCE["source_inventory_summary"],
        )
        and _owned_fields_are_unshadowed(
            _plain_dict_section(source, "fail_closed_inventory_decision"),
            FAIL_CLOSED_OWNED_FIELDS_18_2,
            EXPECTED_SOURCE["fail_closed_inventory_decision"],
        )
        and _owned_fields_are_unshadowed(
            _plain_dict_section(source, "source_boundaries"),
            SOURCE_BOUNDARY_FIELDS_18_2,
            {
                "can_build_desktop_exe_packaging_inventory_matrix": EXPECTED_SOURCE[
                    "source_boundaries"
                ]["can_build_desktop_exe_packaging_inventory_matrix"]
            },
        )
    )


def _source_identity_valid(source: dict[str, Any]) -> bool:
    for key, expected in SOURCE_IDENTITY_EXPECTED.items():
        actual = source.get(key)
        if type(actual) is not type(expected):
            return False
        if actual != expected:
            return False
    return True


def _scalar_reference(source: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _copy_plain(source[key])
        for key in SOURCE_IDENTITY_EXPECTED
        if key in source and type(source[key]) in (str, int, bool)
    }


def _matrix_capabilities() -> dict[str, str]:
    return {
        key: "blocked"
        for key in [
            "runtime_inventory_matrix_evaluation",
            "desktop_entrypoint_selection",
            "desktop_entrypoint_validation",
            "qml_bundle_validation",
            "qt_plugin_discovery",
            "qt_plugin_validation",
            "dependency_resolution",
            "dependency_validation",
            "packaging_metadata_mutation",
            "packaging_profile_selection",
            "packaging_profile_validation",
            "artifact_exclusion_policy_application",
            "artifact_exclusion_policy_validation",
            "windows_toolchain_selection",
            "windows_toolchain_validation",
            "packaging_contract_approval",
            "build_readiness_grant",
            "packaging_authorization",
            "build_authorization",
            "spec_file_creation",
            "build_command_creation",
            "build_command_execution",
            "pyinstaller_execution",
            "briefcase_execution",
            "exe_artifact_creation",
            "artifact_scan",
            "artifact_signing",
            "installer_creation",
            "release_creation",
            "runtime_activation",
            "order_activity",
        ]
    }


def _entrypoint_rows(source_ok: bool) -> list[dict[str, Any]]:
    rows = EXPECTED_SOURCE["desktop_entrypoint_inventory_rows"] if source_ok else []
    out = []
    for index, row in enumerate(rows):
        cli = index > 1
        out.append(
            {
                "matrix_row_id": row["inventory_row_id"] + "_matrix",
                "source_inventory_row_id": row["inventory_row_id"],
                "path": row["path"],
                "source_kind": row["source_kind"],
                "observed_role": row["observed_role"],
                "observed_symbol": row["observed_symbol"],
                "source_observed": True,
                "source_inventory_preserved": source_ok,
                "desktop_entrypoint_candidate": not cli,
                "excluded_from_final_desktop_contract": cli,
                "selection_required": not cli,
                "validation_required": not cli,
                "selected_as_final_desktop_entrypoint": False,
                "approved_for_packaging_contract": False,
                "approved_for_build": False,
                "build_ready": False,
                "matrix_classification": [
                    "desktop_launcher_candidate_requires_contract_selection",
                    "desktop_application_candidate_requires_contract_selection",
                    "cli_preview_source_excluded_from_final_desktop_contract",
                    "cli_preview_source_excluded_from_final_desktop_contract",
                ][index],
                "matrix_result": [
                    "desktop_launcher_observed_selection_and_validation_pending",
                    "desktop_application_entrypoint_observed_selection_and_validation_pending",
                    "cli_preview_source_preserved_as_separate_scope_not_final_desktop_entrypoint",
                    "cli_preview_source_preserved_as_separate_scope_not_final_desktop_entrypoint",
                ][index],
            }
        )
    return out


def _qml_rows(source_ok: bool) -> list[dict[str, Any]]:
    paths = [
        ["ui/pyside_app/qml/MainWindow.qml"],
        ["ui/pyside_app/qml"],
        ["ui/qml"],
        [
            EXPECTED_SOURCE["qml_source_inventory"]["styles_module_qmldir"],
            "ui/pyside_app/qml/Styles/"
            + EXPECTED_SOURCE["qml_source_inventory"]["styles_module_observation"]["source"],
        ],
        ["ui/qml"],
    ]
    ids = [
        "default_qml_entrypoint",
        "pyside_qml_root",
        "shared_qml_root",
        "styles_module",
        "windows_shared_qml_import_path",
    ]
    return [
        {
            "matrix_row_id": ids[i],
            "source_paths": _copy_plain(paths[i]) if source_ok else [],
            "source_inventory_present": source_ok,
            "source_inventory_complete": source_ok,
            "source_inventory_preserved": source_ok,
            "matrix_evaluated": source_ok,
            "validation_required": True,
            "validation_performed": False,
            "unresolved_condition_present": i == 4,
            "approved_for_packaging_contract": False,
            "approved_for_build": False,
            "build_ready": False,
            "matrix_classification": (
                (
                    "windows_shared_qml_import_path_requires_resolution"
                    if i == 4
                    else "qml_source_observed_requires_packaging_contract_validation"
                )
                if source_ok
                else "qml_source_inventory_not_preserved"
            ),
            "matrix_result": (
                (
                    "shared_qml_root_platform_condition_observed_windows_bundle_contract_unresolved"
                    if i == 4
                    else "qml_source_observed_contract_validation_pending"
                )
                if source_ok
                else "qml_bundle_matrix_blocked_by_invalid_source"
            ),
        }
        for i in range(5)
    ]


def _dependency_rows(source_ok: bool) -> list[dict[str, Any]]:
    ids = [
        "project_dependency_declarations",
        "desktop_optional_dependency_declarations",
        "dependency_resolution",
        "desktop_build_tool_candidates",
    ]
    rows = []
    for i, rid in enumerate(ids):
        source_derived = i != 2
        blocked_source = source_derived and not source_ok
        rows.append(
            {
                "matrix_row_id": rid,
                "source_inventory_present": source_ok and source_derived,
                "source_inventory_preserved": source_ok and source_derived,
                "declaration_inventory_complete": source_ok and i in (0, 1),
                "resolution_required": i != 3,
                "resolution_performed": False,
                "selection_required": i == 3,
                "selection_performed": False,
                "validated": False,
                "approved_for_packaging_contract": False,
                "approved_for_build": False,
                "build_ready": False,
                "matrix_classification": "dependency_inventory_source_not_preserved"
                if blocked_source
                else (
                    "dependency_resolution_not_performed"
                    if i == 2
                    else (
                        "desktop_build_tool_candidates_observed_not_selected"
                        if i == 3
                        else "dependency_declarations_observed_resolution_pending"
                    )
                ),
                "matrix_result": "dependency_matrix_blocked_by_invalid_source"
                if blocked_source
                else "dependency_resolution_pending",
            }
        )
    return rows


def _metadata_rows(source_ok: bool) -> list[dict[str, Any]]:
    ids = [
        "setuptools_ui_package_discovery",
        "qml_package_data_declaration",
        "deploy_packaging_sources",
        "deployment_documentation",
    ]
    classes = [
        "setuptools_ui_package_discovery_missing",
        "qml_package_data_declaration_missing",
        "deployment_source_inventory_observed_requires_contract_selection",
        "deployment_source_inventory_observed_requires_contract_selection",
    ]
    return [
        {
            "matrix_row_id": rid,
            "source_inventory_present": source_ok,
            "required_declaration_present": (False if i < 2 else None) if source_ok else None,
            "inventory_complete": (source_ok if i >= 2 else None) if source_ok else False,
            "unresolved_condition_present": i < 2 if source_ok else True,
            "validation_required": i >= 2 if source_ok else False,
            "validation_performed": False,
            "approved_for_packaging_contract": False,
            "approved_for_build": False,
            "build_ready": False,
            "matrix_classification": classes[i]
            if source_ok
            else "packaging_metadata_source_not_preserved",
            "matrix_result": classes[i] + "_pending"
            if source_ok
            else "packaging_metadata_matrix_blocked_by_invalid_source",
        }
        for i, rid in enumerate(ids)
    ]


def _preview_rows(source_ok: bool) -> list[dict[str, Any]]:
    return [
        {
            "matrix_row_id": rid,
            "source_inventory_present": source_ok,
            "source_scope": "cli_preview" if source_ok else "",
            "targets_run_local_bot": source_ok,
            "targets_final_desktop_entrypoint": False,
            "final_desktop_profile_aligned": False,
            "reusable_as_final_desktop_contract": False,
            "profile_validation_performed": False,
            "approved_for_packaging_contract": False,
            "approved_for_build": False,
            "build_ready": False,
            "matrix_classification": "cli_preview_packaging_source_not_aligned_to_final_desktop_contract"
            if source_ok
            else "cli_preview_packaging_source_not_preserved",
            "matrix_result": "cli_preview_packaging_source_preserved_not_final_desktop_contract"
            if source_ok
            else "preview_packaging_matrix_blocked_by_invalid_source",
        }
        for rid in [
            "safe_exe_preview_build_plan",
            "windows_preview_profile",
            "linux_preview_profile",
            "macos_preview_profile",
        ]
    ]


def _finding_rows(source_ok: bool) -> list[dict[str, Any]]:
    mapping = {
        "desktop_build_tools_declared_as_optional_dependencies": "build_tool_candidates_observed_selection_pending"
    }
    default = {
        "desktop_module_launcher_observed": "entrypoint_candidate_requires_contract_selection",
        "desktop_application_main_observed": "entrypoint_candidate_requires_contract_selection",
        "default_qml_entrypoint_observed": "qml_entrypoint_requires_contract_validation",
        "two_qml_source_roots_observed": "qml_roots_require_bundle_contract_definition",
        "shared_qml_import_path_is_platform_conditional": "windows_qml_import_path_requires_resolution",
        "cli_preview_plan_targets_non_desktop_entrypoint": "cli_preview_packaging_not_final_desktop_contract",
        "ui_package_discovery_not_declared_in_current_setuptools_include": "required_packaging_metadata_missing",
        "qml_package_data_not_declared_in_current_setuptools_metadata": "required_qml_package_data_missing",
        "example_config_references_local_secret_paths": "secret_path_exclusion_contract_required",
        "example_config_contains_sensitive_field_reference": "sensitive_field_exclusion_contract_required",
    }
    out = []
    for f in EXPECTED_SOURCE["inventory_findings"] if source_ok else []:
        fid = f["finding_id"]
        blocker = fid != "desktop_build_tools_declared_as_optional_dependencies"
        out.append(
            {
                "finding_id": fid,
                "source_paths": _copy_plain(f["source_paths"]),
                "source_observation_preserved": True,
                "source_severity_classification": f["severity_classification"],
                "matrix_evaluated": True,
                "requires_packaging_contract_action": True,
                "contract_blocker_present": blocker,
                "resolved": False,
                "approved": False,
                "matrix_classification": mapping.get(
                    fid, default.get(fid, "contract_action_required")
                ),
                "matrix_result": "inventory_finding_evaluated_contract_action_pending",
            }
        )
    return out


def _scope_rows(
    *,
    entrypoint_rows_valid: bool,
    qml_inventory_valid: bool,
    python_dependency_inventory_valid: bool,
    preview_packaging_valid: bool,
    artifact_exclusion_policy_valid: bool,
    config_reference_rows_valid: bool,
    packaging_metadata_valid: bool,
    entry_row_ids: list[str],
    qml_row_ids: list[str],
    dependency_row_ids: list[str],
    metadata_row_ids: list[str],
    preview_row_ids: list[str],
    policy_row_ids: list[str],
) -> list[dict[str, Any]]:
    data = [
        (
            "desktop_application_entrypoint",
            _copy_plain(entry_row_ids[:2]),
            [
                "final_desktop_entrypoint_not_selected",
                "desktop_entrypoint_validation_not_performed",
            ],
            entrypoint_rows_valid,
            entrypoint_rows_valid,
            "entrypoint_inventory_complete_contract_conditions_unresolved",
        ),
        (
            "qt_qml_runtime_bundle",
            _copy_plain(qml_row_ids)
            + [
                row_id
                for row_id in metadata_row_ids
                if row_id in ("setuptools_ui_package_discovery", "qml_package_data_declaration")
            ],
            [
                "qml_bundle_validation_not_performed",
                "windows_shared_qml_import_path_unresolved",
                "qt_plugin_inventory_missing",
                "ui_package_discovery_missing",
                "qml_package_data_missing",
            ],
            qml_inventory_valid,
            qml_inventory_valid and packaging_metadata_valid,
            "qml_inventory_complete_runtime_bundle_contract_conditions_unresolved",
        ),
        (
            "windows_exe_artifact_pipeline",
            _copy_plain(dependency_row_ids)
            + [
                row_id
                for row_id in preview_row_ids
                if row_id in ("safe_exe_preview_build_plan", "windows_preview_profile")
            ]
            + _copy_plain(policy_row_ids),
            [
                "final_desktop_packaging_profile_not_aligned",
                "dependency_resolution_not_performed",
                "secret_and_local_data_exclusion_policy_not_validated",
                "windows_toolchain_not_confirmed",
                "future_explicit_build_execution_gate_missing",
            ],
            python_dependency_inventory_valid
            or preview_packaging_valid
            or artifact_exclusion_policy_valid
            or config_reference_rows_valid
            or packaging_metadata_valid,
            python_dependency_inventory_valid
            and preview_packaging_valid
            and artifact_exclusion_policy_valid
            and config_reference_rows_valid
            and packaging_metadata_valid,
            "packaging_source_inventory_complete_final_desktop_pipeline_unresolved",
        ),
    ]
    return [
        {
            "scope_id": sid,
            "source_inventory_artifact_present": source_present,
            "inventory_matrix_evaluated": evaluated,
            "supporting_matrix_row_ids": _copy_plain(support),
            "resolved_condition_count": 0,
            "unresolved_condition_ids": _copy_plain(unres),
            "unresolved_condition_count": len(unres),
            "ready_for_packaging_contract": False,
            "scope_ready": False,
            "scope_authorized": False,
            "build_ready": False,
            "failure_policy": "fail_closed",
            "matrix_classification": cls,
            "matrix_result": "scope_contract_conditions_unresolved",
        }
        for sid, support, unres, source_present, evaluated, cls in data
    ]


def _requirement_rows(
    *,
    entrypoint_rows_valid: bool,
    qml_inventory_valid: bool,
    config_reference_rows_valid: bool,
    python_dependency_inventory_valid: bool,
    packaging_metadata_valid: bool,
    preview_packaging_valid: bool,
    artifact_exclusion_policy_valid: bool,
) -> list[dict[str, Any]]:
    reqs = [
        "desktop_application_entrypoint_inventory",
        "qml_asset_inventory",
        "qt_plugin_inventory",
        "python_dependency_inventory",
        "packaging_profile_alignment",
        "secret_and_local_data_exclusion_policy",
        "windows_target_toolchain_confirmation",
        "future_explicit_build_execution_gate",
    ]
    observed = [
        entrypoint_rows_valid,
        qml_inventory_valid,
        False,
        python_dependency_inventory_valid,
        preview_packaging_valid,
        config_reference_rows_valid and artifact_exclusion_policy_valid,
        False,
        False,
    ]
    evaluated = [
        entrypoint_rows_valid,
        qml_inventory_valid,
        True,
        python_dependency_inventory_valid,
        preview_packaging_valid,
        config_reference_rows_valid or artifact_exclusion_policy_valid,
        True,
        True,
    ]
    unresolved = [
        ["final_desktop_entrypoint_not_selected", "desktop_entrypoint_validation_not_performed"],
        [
            "qml_bundle_validation_not_performed",
            "windows_shared_qml_import_path_unresolved",
            "ui_package_discovery_missing",
            "qml_package_data_missing",
        ],
        ["qt_plugin_inventory_missing"],
        ["dependency_resolution_not_performed"],
        ["final_desktop_packaging_profile_not_aligned"],
        ["secret_and_local_data_exclusion_policy_not_validated"],
        ["windows_toolchain_not_confirmed"],
        ["future_explicit_build_execution_gate_missing"],
    ]
    return [
        {
            "requirement_id": rid,
            "required": True,
            "source_inventory_observed": observed[i],
            "inventory_requirement_satisfied": observed[i],
            "matrix_evaluated": evaluated[i],
            "packaging_contract_requirement_satisfied": False,
            "build_requirement_satisfied": False,
            "missing_inventory": not observed[i],
            "unresolved_for_contract": True,
            "unresolved_condition_ids": _copy_plain(unresolved[i]),
            "requires_future_explicit_step": i == 7,
            "build_ready": False,
            "packaging_authorized": False,
            "build_authorized": False,
            "failure_policy": "fail_closed",
            "matrix_classification": "inventory_observed_contract_unresolved"
            if observed[i]
            else "inventory_missing_contract_unresolved",
            "matrix_result": "requirement_not_build_ready",
        }
        for i, rid in enumerate(reqs)
    ]


BLOCKER_RELATION_ROWS: Final[list[dict[str, Any]]] = [
    {
        "blocker_id": "final_desktop_entrypoint_not_selected",
        "source_finding_ids": [
            "desktop_module_launcher_observed",
            "desktop_application_main_observed",
        ],
        "affected_scope_ids": ["desktop_application_entrypoint"],
    },
    {
        "blocker_id": "desktop_entrypoint_validation_not_performed",
        "source_finding_ids": [
            "desktop_module_launcher_observed",
            "desktop_application_main_observed",
        ],
        "affected_scope_ids": ["desktop_application_entrypoint"],
    },
    {
        "blocker_id": "qml_bundle_validation_not_performed",
        "source_finding_ids": ["default_qml_entrypoint_observed", "two_qml_source_roots_observed"],
        "affected_scope_ids": ["qt_qml_runtime_bundle"],
    },
    {
        "blocker_id": "windows_shared_qml_import_path_unresolved",
        "source_finding_ids": ["shared_qml_import_path_is_platform_conditional"],
        "affected_scope_ids": ["qt_qml_runtime_bundle"],
    },
    {
        "blocker_id": "qt_plugin_inventory_missing",
        "source_finding_ids": [],
        "affected_scope_ids": ["qt_qml_runtime_bundle"],
    },
    {
        "blocker_id": "ui_package_discovery_missing",
        "source_finding_ids": ["ui_package_discovery_not_declared_in_current_setuptools_include"],
        "affected_scope_ids": ["qt_qml_runtime_bundle"],
    },
    {
        "blocker_id": "qml_package_data_missing",
        "source_finding_ids": ["qml_package_data_not_declared_in_current_setuptools_metadata"],
        "affected_scope_ids": ["qt_qml_runtime_bundle"],
    },
    {
        "blocker_id": "final_desktop_packaging_profile_not_aligned",
        "source_finding_ids": ["cli_preview_plan_targets_non_desktop_entrypoint"],
        "affected_scope_ids": ["windows_exe_artifact_pipeline"],
    },
    {
        "blocker_id": "dependency_resolution_not_performed",
        "source_finding_ids": ["desktop_build_tools_declared_as_optional_dependencies"],
        "affected_scope_ids": ["windows_exe_artifact_pipeline"],
    },
    {
        "blocker_id": "secret_and_local_data_exclusion_policy_not_validated",
        "source_finding_ids": [
            "example_config_references_local_secret_paths",
            "example_config_contains_sensitive_field_reference",
        ],
        "affected_scope_ids": ["windows_exe_artifact_pipeline"],
    },
    {
        "blocker_id": "windows_toolchain_not_confirmed",
        "source_finding_ids": ["desktop_build_tools_declared_as_optional_dependencies"],
        "affected_scope_ids": ["windows_exe_artifact_pipeline"],
    },
    {
        "blocker_id": "future_explicit_build_execution_gate_missing",
        "source_finding_ids": [],
        "affected_scope_ids": ["windows_exe_artifact_pipeline"],
    },
]


def _blocker_rows(inventory_findings_valid: bool) -> list[dict[str, Any]]:
    rows = BLOCKER_RELATION_ROWS
    return [
        {
            "blocker_id": row["blocker_id"],
            "source_finding_ids": _copy_plain(row["source_finding_ids"])
            if inventory_findings_valid
            else [],
            "affected_scope_ids": _copy_plain(row["affected_scope_ids"]),
            "blocker_present": True,
            "resolved": False,
            "requires_18_3_contract": True,
            "blocks_build_readiness": True,
            "blocks_packaging_authorization": True,
            "blocks_build_authorization": True,
            "failure_policy": "fail_closed",
            "classification": "unresolved_packaging_contract_condition",
            "result": "requires_18_3_contract_definition_not_resolution",
        }
        for row in rows
    ]


def _future_steps() -> list[dict[str, Any]]:
    return [
        {"step": step, "title": title, "source_only": True, "build_performed": False}
        for step, title in [
            ("18.3", "BLOCK P DESKTOP EXE PACKAGING CONTRACT"),
            ("18.4", "BLOCK P DESKTOP EXE PACKAGING READ MODEL"),
            ("18.5", "BLOCK P DESKTOP EXE BUILD READINESS MATRIX"),
            ("18.6", "BLOCK P DESKTOP EXE BUILD READINESS CONTRACT"),
            ("18.7", "BLOCK P DESKTOP EXE BUILD READINESS READ MODEL"),
            ("18.8", "BLOCK P CLOSURE AUDIT"),
        ]
    ]


def build_preview_block_p_desktop_exe_packaging_inventory_matrix() -> dict[str, Any]:
    source = build_preview_block_p_desktop_exe_packaging_source_inventory()
    raw_source = source if type(source) is dict else {}
    safe_source, all_top_level_keys_exact_str = _safe_top_level_source(raw_source)
    source_plain_bounded = _all_plain_json(raw_source, MAX_DIAGNOSTIC_CONTAINER_DEPTH)
    local_valid = {
        "identity_valid": _source_identity_valid(safe_source),
        "reference_valid": _section_valid(
            safe_source.get("block_p_desktop_exe_packaging_entry_contract_reference"),
            EXPECTED_SOURCE["block_p_desktop_exe_packaging_entry_contract_reference"],
        ),
        "summary_valid": _section_valid(
            safe_source.get("source_inventory_summary"), EXPECTED_SOURCE["source_inventory_summary"]
        ),
        "entrypoint_rows_valid": _section_valid(
            safe_source.get("desktop_entrypoint_inventory_rows"),
            EXPECTED_SOURCE["desktop_entrypoint_inventory_rows"],
        ),
        "qml_inventory_valid": _section_valid(
            safe_source.get("qml_source_inventory"), EXPECTED_SOURCE["qml_source_inventory"]
        ),
        "config_reference_rows_valid": _section_valid(
            safe_source.get("config_and_runtime_reference_inventory_rows"),
            EXPECTED_SOURCE["config_and_runtime_reference_inventory_rows"],
        ),
        "python_dependency_inventory_valid": _section_valid(
            safe_source.get("python_dependency_inventory"),
            EXPECTED_SOURCE["python_dependency_inventory"],
        ),
        "packaging_metadata_valid": _section_valid(
            safe_source.get("packaging_metadata_inventory"),
            EXPECTED_SOURCE["packaging_metadata_inventory"],
        ),
        "preview_packaging_valid": _section_valid(
            safe_source.get("existing_cli_preview_packaging_inventory"),
            EXPECTED_SOURCE["existing_cli_preview_packaging_inventory"],
        ),
        "artifact_exclusion_policy_valid": _section_valid(
            safe_source.get("artifact_exclusion_policy_inventory"),
            EXPECTED_SOURCE["artifact_exclusion_policy_inventory"],
        ),
        "inventory_findings_valid": _section_valid(
            safe_source.get("inventory_findings"), EXPECTED_SOURCE["inventory_findings"]
        ),
        "real_capability_valid": _section_valid(
            safe_source.get("real_capability_inventory_state"),
            EXPECTED_SOURCE["real_capability_inventory_state"],
        ),
        "fail_closed_valid": _section_valid(
            safe_source.get("fail_closed_inventory_decision"),
            EXPECTED_SOURCE["fail_closed_inventory_decision"],
        ),
        "evidence_valid": _section_valid(
            safe_source.get("non_execution_inventory_evidence"),
            EXPECTED_SOURCE["non_execution_inventory_evidence"],
        ),
        "inventory_boundaries_valid": _section_valid(
            safe_source.get("inventory_boundaries"), EXPECTED_SOURCE["inventory_boundaries"]
        ),
        "source_boundaries_valid": _section_valid(
            safe_source.get("source_boundaries"), EXPECTED_SOURCE["source_boundaries"]
        ),
        "future_steps_valid": _section_valid(
            safe_source.get("future_steps"), EXPECTED_SOURCE["future_steps"]
        ),
    }
    source_accepted = (
        source_plain_bounded
        and all_top_level_keys_exact_str
        and list(raw_source.keys()) == TOP_LEVEL_FIELDS_18_1
        and all(local_valid.values())
        and _no_shadowing(safe_source)
    )
    status = STATUS if source_accepted else BLOCKED_STATUS
    entry = _entrypoint_rows(local_valid["entrypoint_rows_valid"])
    qml = _qml_rows(local_valid["qml_inventory_valid"])
    deps = _dependency_rows(local_valid["python_dependency_inventory_valid"])
    meta = _metadata_rows(local_valid["packaging_metadata_valid"])
    preview = _preview_rows(local_valid["preview_packaging_valid"])
    findings = _finding_rows(local_valid["inventory_findings_valid"])
    policy = [
        {
            "matrix_row_id": "artifact_exclusion_policy",
            "policy_source": EXPECTED_SOURCE["artifact_exclusion_policy_inventory"]["policy_source"]
            if local_valid["artifact_exclusion_policy_valid"]
            else "",
            "policy_version": EXPECTED_SOURCE["artifact_exclusion_policy_inventory"][
                "policy_version"
            ]
            if local_valid["artifact_exclusion_policy_valid"]
            else "",
            "policy_observed": local_valid["artifact_exclusion_policy_valid"],
            "denied_patterns_inventory_preserved": local_valid["artifact_exclusion_policy_valid"],
            "policy_application_required": True,
            "policy_applied": False,
            "desktop_bundle_validation_required": True,
            "desktop_bundle_validation_performed": False,
            "approved_for_packaging_contract": False,
            "approved_for_build": False,
            "build_ready": False,
            "matrix_classification": "artifact_exclusion_policy_observed_not_validated_for_desktop_bundle"
            if local_valid["artifact_exclusion_policy_valid"]
            else "artifact_exclusion_policy_source_not_preserved",
            "matrix_result": "artifact_exclusion_policy_requires_contract_application_and_validation"
            if local_valid["artifact_exclusion_policy_valid"]
            else "artifact_exclusion_policy_matrix_blocked_by_invalid_source",
        }
    ]
    entry_row_ids = [row["matrix_row_id"] for row in entry]
    qml_row_ids = [row["matrix_row_id"] for row in qml]
    dependency_row_ids = [row["matrix_row_id"] for row in deps]
    metadata_row_ids = [row["matrix_row_id"] for row in meta]
    preview_row_ids = [row["matrix_row_id"] for row in preview]
    policy_row_ids = [row["matrix_row_id"] for row in policy]
    scopes = _scope_rows(
        entrypoint_rows_valid=local_valid["entrypoint_rows_valid"],
        qml_inventory_valid=local_valid["qml_inventory_valid"],
        python_dependency_inventory_valid=local_valid["python_dependency_inventory_valid"],
        preview_packaging_valid=local_valid["preview_packaging_valid"],
        artifact_exclusion_policy_valid=local_valid["artifact_exclusion_policy_valid"],
        config_reference_rows_valid=local_valid["config_reference_rows_valid"],
        packaging_metadata_valid=local_valid["packaging_metadata_valid"],
        entry_row_ids=entry_row_ids,
        qml_row_ids=qml_row_ids,
        dependency_row_ids=dependency_row_ids,
        metadata_row_ids=metadata_row_ids,
        preview_row_ids=preview_row_ids,
        policy_row_ids=policy_row_ids,
    )
    reqs = _requirement_rows(
        entrypoint_rows_valid=local_valid["entrypoint_rows_valid"],
        qml_inventory_valid=local_valid["qml_inventory_valid"],
        config_reference_rows_valid=local_valid["config_reference_rows_valid"],
        python_dependency_inventory_valid=local_valid["python_dependency_inventory_valid"],
        packaging_metadata_valid=local_valid["packaging_metadata_valid"],
        preview_packaging_valid=local_valid["preview_packaging_valid"],
        artifact_exclusion_policy_valid=local_valid["artifact_exclusion_policy_valid"],
    )
    blockers = _blocker_rows(local_valid["inventory_findings_valid"])
    all_source_approvals_false = (
        local_valid["summary_valid"]
        and local_valid["fail_closed_valid"]
        and EXPECTED_SOURCE["source_inventory_summary"]["desktop_entrypoint_approved"] is False
        and EXPECTED_SOURCE["source_inventory_summary"]["qml_bundle_approved"] is False
        and EXPECTED_SOURCE["source_inventory_summary"]["dependency_bundle_approved"] is False
        and EXPECTED_SOURCE["source_inventory_summary"]["packaging_profile_approved"] is False
        and EXPECTED_SOURCE["source_inventory_summary"]["build_tool_approved"] is False
    )
    all_source_build_runtime_order_flags_false = (
        local_valid["summary_valid"]
        and local_valid["fail_closed_valid"]
        and local_valid["inventory_boundaries_valid"]
        and EXPECTED_SOURCE["source_inventory_summary"]["build_ready"] is False
        and EXPECTED_SOURCE["source_inventory_summary"]["runtime_authorized"] is False
        and EXPECTED_SOURCE["source_inventory_summary"]["orders_authorized"] is False
        and EXPECTED_SOURCE["fail_closed_inventory_decision"]["build_executed_by_18_1"] is False
        and EXPECTED_SOURCE["fail_closed_inventory_decision"]["runtime_enabled_by_18_1"] is False
        and EXPECTED_SOURCE["fail_closed_inventory_decision"]["orders_enabled_by_18_1"] is False
        and EXPECTED_SOURCE["inventory_boundaries"]["runtime_started"] is False
        and EXPECTED_SOURCE["inventory_boundaries"]["orders_enabled"] is False
    )
    cap_source = (
        safe_source.get("real_capability_inventory_state", {})
        if local_valid["real_capability_valid"]
        else {}
    )
    inherited_18_0 = (
        cap_source.get("inherited_18_0_capabilities", {}) if type(cap_source) is dict else {}
    )
    inherited_o = (
        _copy_plain(inherited_18_0.get("inherited_block_o_capabilities", {}))
        if type(inherited_18_0) is dict
        else {}
    )
    inherited_p = (
        _copy_plain(inherited_18_0.get("block_p_capabilities", {}))
        if type(inherited_18_0) is dict
        else {}
    )
    inventory_caps = (
        _copy_plain(cap_source.get("inventory_capabilities", {}))
        if type(cap_source) is dict
        else {}
    )
    matrix_caps = _matrix_capabilities()
    all_blocked = (
        bool(inherited_o)
        and bool(inherited_p)
        and bool(inventory_caps)
        and bool(matrix_caps)
        and all(
            v == "blocked"
            for m in [inherited_o, inherited_p, inventory_caps, matrix_caps]
            for v in m.values()
        )
    )
    op_false = {
        "inventory_observations_modified_by_18_2": False,
        "desktop_entrypoint_selected_by_18_2": False,
        "desktop_entrypoint_approved_by_18_2": False,
        "qml_bundle_validated_by_18_2": False,
        "qt_plugins_discovered_by_18_2": False,
        "dependency_resolution_performed_by_18_2": False,
        "packaging_metadata_modified_by_18_2": False,
        "packaging_profile_selected_by_18_2": False,
        "build_tool_selected_by_18_2": False,
        "packaging_contract_approved_by_18_2": False,
        "build_ready_by_18_2": False,
        "packaging_authorized_by_18_2": False,
        "build_authorized_by_18_2": False,
        "build_executed_by_18_2": False,
        "artifact_created_by_18_2": False,
        "release_authorized_by_18_2": False,
        "runtime_enabled_by_18_2": False,
        "orders_enabled_by_18_2": False,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "block_p_desktop_exe_packaging_inventory_matrix_kind": KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_p_desktop_exe_packaging_inventory_matrix_status": INVENTORY_MATRIX_STATUS,
        "block_p_desktop_exe_packaging_inventory_matrix_decision": INVENTORY_MATRIX_DECISION,
        "inventory_matrix_artifact_complete": source_accepted,
        "ready_for_block_p_3": source_accepted,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_p_desktop_exe_packaging_source_inventory_reference": {
            **_scalar_reference(safe_source),
            "source_block_p_desktop_exe_packaging_source_inventory_step": "FUNCTIONAL-PREVIEW-18.1",
            "source_inventory_read_by_18_2": True,
            "source_inventory_available_before_matrix": True,
            "static_inventory_matrix_only": True,
            "inventory_matrix_built_by_18_2": True,
            "inventory_matrix_artifact_complete_by_18_2": source_accepted,
            "ready_for_functional_preview_18_3": source_accepted,
            "runtime_filesystem_scan": False,
            "repo_rescan": False,
            "environment_scan": False,
            "secret_file_read": False,
            "dependency_import": False,
            "dependency_resolution": False,
            "pyside_import": False,
            "qml_load": False,
            "qt_plugin_discovery": False,
            "packaging_profile_validation": False,
            "build_tool_selection": False,
            "build_tool_execution": False,
            "spec_file_creation": False,
            "build_command_creation": False,
            "build_command_execution": False,
            "packaging": False,
            "artifact_creation": False,
            "artifact_scan": False,
            "artifact_signing": False,
            "installer_creation": False,
            "release": False,
            "runtime": False,
            "orders": False,
            "network": False,
            "credentials_read": False,
        },
        "inventory_matrix_summary": {
            "source_18_1_accepted": source_accepted,
            "source_inventory_artifact_preserved": source_accepted,
            "source_only": True,
            "plain_data": True,
            "static_matrix": True,
            "inventory_matrix_artifact_complete": source_accepted,
            "inventory_matrix_evaluated": source_accepted,
            "desktop_entrypoint_matrix_row_count": len(entry),
            "qml_bundle_matrix_row_count": len(qml),
            "python_dependency_matrix_row_count": len(deps),
            "packaging_metadata_matrix_row_count": len(meta),
            "existing_preview_packaging_matrix_row_count": len(preview),
            "artifact_exclusion_policy_matrix_row_count": 1,
            "inventory_finding_matrix_row_count": len(findings),
            "packaging_scope_matrix_row_count": len(scopes),
            "packaging_requirement_matrix_row_count": len(reqs),
            "unresolved_contract_blocker_count": len(blockers),
            "all_matrix_rows_evaluated": source_accepted,
            "unresolved_contract_blockers_present": True,
            "packaging_contract_conditions_satisfied": False,
            "desktop_entrypoint_selected": False,
            "desktop_entrypoint_approved": False,
            "qml_bundle_validated": False,
            "qt_plugin_inventory_complete": False,
            "dependency_bundle_validated": False,
            "packaging_profile_aligned": False,
            "artifact_exclusion_policy_validated": False,
            "windows_toolchain_confirmed": False,
            "future_explicit_build_execution_gate_present": False,
            "build_ready": False,
            "packaging_authorized": False,
            "build_authorized": False,
            "artifact_creation_authorized": False,
            "release_authorized": False,
            "runtime_authorized": False,
            "orders_authorized": False,
            "only_source_only_18_3_handoff_allowed": source_accepted,
        },
        "source_inventory_preservation": {
            "source_inventory_identity": _scalar_reference(safe_source),
            "desktop_entrypoint_row_count": 4 if local_valid["entrypoint_rows_valid"] else 0,
            "pyside_qml_file_count": 24 if local_valid["qml_inventory_valid"] else 0,
            "shared_qml_file_count": 107 if local_valid["qml_inventory_valid"] else 0,
            "additional_qml_support_asset_count": 0 if local_valid["qml_inventory_valid"] else 0,
            "deploy_packaging_source_file_count": EXPECTED_SOURCE["packaging_metadata_inventory"][
                "deploy_packaging_source_file_count"
            ]
            if local_valid["packaging_metadata_valid"]
            else 0,
            "deployment_documentation_file_count": EXPECTED_SOURCE["packaging_metadata_inventory"][
                "deployment_documentation_file_count"
            ]
            if local_valid["packaging_metadata_valid"]
            else 0,
            "config_reference_row_count": 6 if local_valid["config_reference_rows_valid"] else 0,
            "project_dependency_count": EXPECTED_SOURCE["python_dependency_inventory"][
                "project_dependency_count"
            ]
            if local_valid["python_dependency_inventory_valid"]
            else 0,
            "desktop_optional_dependency_count": EXPECTED_SOURCE["python_dependency_inventory"][
                "desktop_optional_dependency_count"
            ]
            if local_valid["python_dependency_inventory_valid"]
            else 0,
            "inventory_finding_count": 11 if local_valid["inventory_findings_valid"] else 0,
            "cli_preview_remains_separate": local_valid["preview_packaging_valid"],
            "all_source_approvals_false": all_source_approvals_false,
            "all_source_build_runtime_order_flags_false": all_source_build_runtime_order_flags_false,
            "source_inventory_preserved": source_accepted,
            "source_inventory_recalculated": False,
            "repo_rescanned": False,
            "inventory_paths_modified": False,
            "inventory_findings_modified": False,
        },
        "desktop_entrypoint_matrix_rows": entry,
        "qml_bundle_matrix_rows": qml,
        "python_dependency_matrix_rows": deps,
        "packaging_metadata_matrix_rows": meta,
        "existing_preview_packaging_matrix_rows": preview,
        "artifact_exclusion_policy_matrix_rows": policy,
        "inventory_finding_matrix_rows": findings,
        "packaging_scope_matrix_rows": scopes,
        "packaging_requirement_matrix_rows": reqs,
        "unresolved_contract_blocker_rows": blockers,
        "real_capability_matrix_state": {
            "inherited_block_o_capabilities": inherited_o,
            "inherited_block_p_capabilities": inherited_p,
            "source_inventory_capabilities": inventory_caps,
            "inventory_matrix_capabilities": matrix_caps,
            "inherited_block_o_capabilities_known_blocked": bool(inherited_o),
            "inherited_block_p_capabilities_known_blocked": bool(inherited_p),
            "source_inventory_capabilities_known_blocked": bool(inventory_caps),
            "matrix_capabilities_known_blocked": True,
            "all_real_capabilities_blocked_at_18_2": all_blocked,
        },
        "fail_closed_matrix_decision": {
            "block_p_source_inventory_in_18_1": "preserved" if source_accepted else "not_preserved",
            "block_p_inventory_matrix_in_18_2": "complete" if source_accepted else "blocked",
            "block_p_packaging_contract_in_18_3": "allowed" if source_accepted else "blocked",
            "only_source_only_18_3_handoff_allowed": source_accepted,
            **op_false,
        },
        "non_execution_matrix_evidence": {
            "source_builder_called": True,
            "source_builder_call_count": 1,
            "source_accepted": source_accepted,
            **local_valid,
            "exact_source_counts_preserved": source_accepted,
            "desktop_entrypoint_matrix_row_count": len(entry),
            "qml_bundle_matrix_row_count": len(qml),
            "python_dependency_matrix_row_count": len(deps),
            "packaging_metadata_matrix_row_count": len(meta),
            "existing_preview_packaging_matrix_row_count": len(preview),
            "artifact_exclusion_policy_matrix_row_count": 1,
            "inventory_finding_matrix_row_count": len(findings),
            "packaging_scope_matrix_row_count": len(scopes),
            "packaging_requirement_matrix_row_count": len(reqs),
            "unresolved_contract_blocker_count": len(blockers),
            "matrix_artifact_complete": source_accepted,
            **op_false,
        },
        "matrix_boundaries": {
            "reads_18_1_only": True,
            "source_only": True,
            "plain_data": True,
            "static": True,
            "repo_rescan": False,
            "filesystem_inventory": False,
            "environment_scan": False,
            "secret_file_read": False,
            "dependency_import": False,
            "dependency_resolution": False,
            "pyside_import": False,
            "qml_load": False,
            "qt_plugin_discovery": False,
            "packaging_metadata_mutation": False,
            "packaging_profile_validation": False,
            "build_tool_selection": False,
            "build_tool_execution": False,
            "spec_file_creation": False,
            "build_command_creation": False,
            "build_command_execution": False,
            "packaging_performed": False,
            "artifact_created": False,
            "artifact_scanned": False,
            "artifact_signed": False,
            "installer_created": False,
            "release_performed": False,
            "runtime_started": False,
            "orders_enabled": False,
            "network_opened": False,
            "credentials_read": False,
            "qml_bridge_gateway_controller_changed": False,
            "can_feed_only_18_3_packaging_contract": source_accepted,
        },
        "source_boundaries": {
            "source_block_p_desktop_exe_packaging_source_inventory": "FUNCTIONAL-PREVIEW-18.1",
            "source_inventory_preserved": source_accepted,
            "can_build_desktop_exe_packaging_inventory_matrix": source_accepted,
            "inventory_matrix_artifact_complete": source_accepted,
            "can_build_desktop_exe_packaging_contract": source_accepted,
            "can_feed_18_3": source_accepted,
        },
        "future_steps": _future_steps(),
        "status": status,
    }
    return _copy_plain(result)
