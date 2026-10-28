# Stage 9 observer contract fixtures

Every file is synthetic structural test data. No artifact has been verified,
no signature is cryptographically approved, and no host or runtime is authenticated.
Hashes refer to the explicitly synthetic byte strings used by these fixtures,
except for the launch-policy and scope-binding digests, which bind the supplied
canonical fixture documents. Acceptance signatures deliberately have valid syntax
without constituting approval.

`refused_manifest.json`, `refused_policy.json`, and `refused_result.json` are
**deliberately refused incomplete schema examples**, not valid V1 documents.
Their empty unspecified nested records must never be accepted as opaque data.
They document the approved specification gaps and fail-closed behavior pending
review of the proposed completion. Positive fixtures never qualify hardware.

The complete **PROPOSED_UNAPPROVED** profile uses pure fixture builders
`proposed_manifest_launch_fixture` and `proposed_bundle_documents` in
`tests/test_stage9_observer_contract.py`. They derive mutually bound document
hashes from synthetic bytes instead of pinning copied placeholder digests.
They are accepted only by the explicit review APIs (`parse_proposed_*`), always
retain `authorization=UNVERIFIED` and all qualification blocks, and are still
refused by the standard manifest/policy/result/report entrypoints. They do not
represent approved build materials, real signatures, artifacts or host evidence.
