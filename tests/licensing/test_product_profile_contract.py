from bot_core.licensing.external_provisioning import (
    ProductionProvisioningPackageVerifier,
)
from bot_core.licensing.product_profile import (
    PRODUCT_NAME,
    PRODUCTION_PRODUCT_PROFILE,
)


def test_production_profile_reuses_canonical_product_identity() -> None:
    assert PRODUCTION_PRODUCT_PROFILE == PRODUCT_NAME == "CryptoHunter"
    source = ProductionProvisioningPackageVerifier.__init__.__code__
    assert "CRYPTOHUNTER_PRODUCTION" not in source.co_consts
