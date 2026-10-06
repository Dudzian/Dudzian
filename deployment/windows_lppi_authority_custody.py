"""Native CertifyCreation collection for the exact LPPI successor key.

Only borrowed PCP TBS contexts and virtual handles are used. This module never
creates an AK, flushes PCP handles, or issues a verified custody capability.
"""

from __future__ import annotations

from bot_core.licensing.lppi_authority_custody import (
    LPPIAuthorityKeyCustodyEvidenceV1,
    lppi_authority_custody_binding,
    lppi_authority_custody_qualifying_data,
    parse_lppi_authority_public,
)
from bot_core.licensing.production_tpm_custody import (
    _verify_creation,
    parse_production_creation_attestation,
    parse_production_ecc_public,
)
from deployment import windows_cng_custody_bridge as bridge


def collect_lppi_authority_key_custody(
    *, accepted: object, key: object
) -> LPPIAuthorityKeyCustodyEvidenceV1:
    from bot_core.licensing.lppi_package_acceptance import require_verified_lppi_package_acceptance
    from deployment.windows_lppi_authority_key import require_verified_production_lppi_authority_key

    trusted = require_verified_lppi_package_acceptance(accepted)
    qualified = require_verified_production_lppi_authority_key(key)
    native = qualified._native
    context = bridge._platform_handle(native, qualified._provider, provider=True)
    subject_handle = bridge._platform_handle(native, qualified._key, provider=False)
    creation_hash = bridge._creation_hash(native.property(qualified._key, "PCP_KEY_CREATIONHASH"))
    creation_ticket = bridge._creation_ticket(
        native.property(qualified._key, "PCP_KEY_CREATIONTICKET")
    )
    tbs = bridge._load_tbs_native()
    if type(tbs) is not bridge._TBSCustodyAPI:
        raise bridge.WindowsCNGCustodyBridgeError("EXACT_NATIVE_TBS_BOUNDARY_REQUIRED")
    tbs.require_tpm20()
    ak = bridge._open_retained_ak(native, qualified._provider, trusted.ak_key_name)
    try:
        bridge._require_ak_profile(native, ak, trusted.ak_key_name)
        ak_handle = bridge._platform_handle(native, ak, provider=False)
        if ak_handle == subject_handle:
            raise bridge.WindowsCNGCustodyBridgeError("DISTINCT_RETAINED_AK_REQUIRED")
        ak_raw, ak_name, ak_qualified_name = bridge._read_public(tbs, context, ak_handle)
        ak_public = parse_production_ecc_public(ak_raw, role="ak")
        projection = trusted.projection.document
        if (
            ak_public.name != ak_name
            or ak_raw.hex() != projection["ak"]["public_area"]["hex"]
            or ak_name.hex() != projection["ak"]["name"]
        ):
            raise bridge.WindowsCNGCustodyBridgeError("RETAINED_AK_IDENTITY_OR_AUTH_MISMATCH")
        subject_raw, subject_name, _qualified_name = bridge._read_public(
            tbs, context, subject_handle
        )
        subject = parse_lppi_authority_public(subject_raw)
        if subject.sec1 != qualified.public_key_bytes or subject.name != subject_name:
            raise bridge.WindowsCNGCustodyBridgeError("PCP_TPM_PUBLIC_IDENTITY_MISMATCH")
        binding = lppi_authority_custody_binding(
            accepted=trusted,
            key=qualified,
            subject_tpmt_public=subject_raw,
            subject_name=subject_name,
            creation_hash=creation_hash,
            creation_ticket=creation_ticket,
            retained_ak_public=ak_raw,
            retained_ak_name=ak_name,
            retained_ak_qualified_name=ak_qualified_name,
        )
        qualifying = lppi_authority_custody_qualifying_data(binding)
        parameters = (
            bridge._tpm2b(qualifying)
            + bridge._tpm2b(creation_hash)
            + b"\x00\x18\x00\x0b"
            + creation_ticket
        )
        result = tbs.submit(
            context,
            bridge._packet(
                bridge.TPM_CC_CERTIFY_CREATION, (ak_handle, subject_handle), parameters, auth=True
            ),
            auth=True,
        )
        attest, offset = bridge._read_2b(result, 0)
        signature = result[offset:]
        parsed = parse_production_creation_attestation(attest)
        if parsed.qualified_signer != ak_qualified_name:
            raise bridge.WindowsCNGCustodyBridgeError(
                "CERTIFY_CREATION_RETAINED_AK_QUALIFIED_NAME_MISMATCH"
            )
        _verify_creation(
            attest=attest,
            signature=signature,
            ak=ak_public,
            expected_name=subject_name,
            expected_creation_hash=creation_hash,
            expected_qualifier=qualifying,
        )
        import hashlib

        return LPPIAuthorityKeyCustodyEvidenceV1.from_mapping(
            {
                **binding,
                "certify_creation_attest_hex": attest.hex(),
                "certify_creation_signature_hex": signature.hex(),
                "tpm_creation_attestation_sha256": hashlib.sha256(attest).hexdigest(),
            }
        )
    finally:
        native.free(ak)
