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
