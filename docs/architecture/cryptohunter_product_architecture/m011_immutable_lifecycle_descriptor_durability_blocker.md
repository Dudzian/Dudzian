# M0.11 immutable lifecycle descriptor durability blocker

## Original decision

The proposed persistence of the **exact** `MigrationRecord` as an immutable-history
carrier is blocked by the existing M0.11 hash dependency contract.  The canonical
artifact and its freeze remain unchanged.

## Exact cycle

The existing meaning of `MigrationRecord.post_state_fingerprint_sha256` is the
fingerprint of the complete candidate target current/history projection.  The
proposed genesis membership would add the exact `MigrationRecord` carrier to that
immutable-history projection.

That produces this dependency cycle:

```text
MigrationRecord.post_state_fingerprint_sha256
  -> MigrationRecord payload
  -> Migration immutable descriptor PersistenceRecord
  -> candidate immutable-history projection
  -> candidate StateStore state_fingerprint_sha256
  -> MigrationRecord.post_state_fingerprint_sha256
```

The value therefore cannot be known before constructing its carrier, while the
carrier must already exist to calculate the value.  This contradicts the frozen
rule that lifecycle carrier values are known before the enclosing write and that
the candidate current/history projection follows those carriers in the DAG.

The baseline fields `pre_state_fingerprint_sha256`,
`transaction_fingerprint_sha256`, and `protected_freshness_generation` do not
cause this cycle: they refer to the already verified source snapshot.  The cycle
is specifically caused by `post_state_fingerprint_sha256` and the requirement to
persist the exact descriptor inside the projection that it fingerprints.

## SecretHandoff finding

No equivalent intrinsic cycle was found for `SecretHandoffRecord`.  Its inner
fingerprints cover reconciliation metadata and the identity-independent operation
projection.  Its old and new values are opaque secure-store references, not raw
secret payloads.  It could receive an immutable carrier after the shared amendment
model is made coherent, but amending only one of the two requested descriptors
would not close the requested amendment.

## Resolution: asymmetric descriptor model

M0.11 resolves the blocker without changing the StateStore hash projection:

- an exact `MigrationRecord` `PersistenceRecord` carrier is permanently rejected;
- `MigrationRecord` is a snapshot-bound, ephemeral runtime candidate and has no
  durable or restore authority;
- restart uses the durable migration lifecycle identity to resolve the sealed
  `MigrationDefinition`, then combines it with one fresh verified StateStore
  snapshot and deterministic target derivation to reconstruct a new runtime
  binding; and
- `SecretHandoffRecord`, which has no equivalent intrinsic cycle, receives the
  registered durable `SecretHandoff immutable descriptor` carrier.

The exact operation data required to reconcile a secret handoff cannot be
reconstructed from transition hashes.  Its immutable carrier is therefore written
at lifecycle genesis and recovered as candidate evidence, never restore authority.
