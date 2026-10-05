# Marketplace signing trust boundary

Marketplace signatures prove both cryptographic validity and authorization. A
signature document is untrusted input: its `public_key`, when retained for
legacy interoperability, is metadata only. Verification starts with `key_id`,
looks that identifier up in an independently provisioned
`MarketplaceTrustPolicy`, checks the authorized issuer and exact trust domain,
and only then verifies the signature. Unknown keys, malformed metadata, and a
missing production trust policy fail closed.

Trust domains are `development`, `test`, and `production`. Keys explicitly
authorized for one domain cannot be used in another. The committed
`dev-presets-ed25519` fixture and `marketplace-ci` issuer are test credentials;
the signing policy and release bundle builder reject them for production.

## Production CI provisioning

The release job requires all of the following configuration and has no
development-key fallback:

* `MARKETPLACE_PRODUCTION_ED25519_PRIVATE_KEY` GitHub secret (external private
  signing material),
* `MARKETPLACE_PRODUCTION_ED25519_PUBLIC_KEY` GitHub secret (independently
  provisioned public trust anchor),
* `MARKETPLACE_PRODUCTION_HMAC_KEY` GitHub secret,
* `MARKETPLACE_PRODUCTION_KEY_ID` GitHub variable, and
* `MARKETPLACE_PRODUCTION_ISSUER` GitHub variable.

The private key is written with a restrictive umask to the ephemeral runner
directory and is never committed. Missing configuration stops catalog signing
or bundle construction. Production catalog signatures bind `key_id`, `issuer`,
and `environment=production`; the bundle builder checks all three before it
copies any asset into an installer.

Existing development fixtures remain usable when callers explicitly select the
test/development domain and configure the matching fixture as an external trust
anchor. Legacy signatures without an environment are never accepted in the
production domain.
