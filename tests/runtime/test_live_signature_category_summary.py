from __future__ import annotations

import tests._pathbootstrap  # noqa: F401  # pylint: disable=unused-import

from bot_core.runtime.live_signature_categories import summarize_live_categories_from_documents


def test_summarize_live_categories_distinguishes_detected_from_ok() -> None:
    categories_ok, categories_detected = summarize_live_categories_from_documents(
        {
            "kyc_packet": {
                "name": "kyc_packet",
                "path": "compliance/live/binance/kyc_packet.pdf",
                "signed_by": ("compliance",),
                "status": "invalid",
            },
            "risk_profile_alignment": {
                "name": "risk_profile_alignment",
                "path": "risk/live/binance/risk_profile_alignment.pdf",
                "signed_by": ("risk",),
                "status": "ok",
            },
            "penetration_report": {
                "name": "penetration_report",
                "path": "security/live/binance/penetration_report.pdf",
                "signed_by": ("security",),
                "status": "invalid",
            },
        }
    )

    assert categories_ok == {"compliance": False, "risk": True, "penetration": False}
    assert categories_detected == {"compliance": True, "risk": True, "penetration": True}


def test_summarize_live_categories_ignores_unclassified_entries() -> None:
    categories_ok, categories_detected = summarize_live_categories_from_documents(
        {
            "audit_note": {
                "name": "audit_note",
                "path": "docs/audit_note.txt",
                "signed_by": (),
                "status": "ok",
            }
        }
    )

    assert categories_ok == {"compliance": False, "risk": False, "penetration": False}
    assert categories_detected == {"compliance": False, "risk": False, "penetration": False}
