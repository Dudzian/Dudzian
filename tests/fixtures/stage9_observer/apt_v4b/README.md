# Offline APT V4b synthetic cryptographic fixtures

Every key here has the UID `SYNTHETIC TEST ONLY`. These keys, fingerprints,
keyrings and signatures are disposable test data. They are not accepted source
authorities, production roots, acceptance decisions or freshness checkpoints.
No production policy references them. No private key is retained here.

`Release` is inert metadata for cryptographic tests. `allowed.asc` signs its
exact bytes with the first Ed25519 signing subkey of a certification-only primary
key. The public `allowed-keyring.gpg` also contains a second signing subkey;
`wrong-subkey.asc` demonstrates that a valid signature by that second key still
requires a separate exact signer policy decision. `unknown.asc` uses a distinct
primary/signing pair absent from the allowed keyring. `weak-hash.asc` uses SHA1,
`expired.asc` expired after one day, and `revoked-keyring.gpg` carries the allowed
primary key's revocation. `manifest.json` records exact fingerprints and byte
hashes for fixture integrity, without independently approving those identities.

`regenerate.py` creates fresh random TEST keys in a temporary private GNUPGHOME,
signs inert bytes using an explicit local `gpg`, exports only public material,
kills the temporary agent and discards private key state. It does not import
into a global key home, contact keyservers, install packages or modify any
system keyring. Regeneration changes the public key and signature hashes.
The runtime Observer never calls this helper or `gpg`.

The real cryptographic tests require a preprovisioned local `gpgv`. Set
`STAGE9_TEST_GPGV` to its absolute path or provide it on PATH. Tests select and
hash that executable explicitly; they never acquire tools or keys over the
network. Disposable dynamic signatures additionally need local `gpg` and use
only pytest temporary directories. Missing tools cause an explicit skipped
validation, not a successful cryptographic result.

For this workspace's validation, a separate test tool was provisioned under
`work/tools/` without system installation: Debian `gpgv` version
`2.4.7-21+deb13u1+b5`, from
`https://snapshot.debian.org/archive/debian/20260925T181338Z/pool/main/g/gnupg2/gpgv_2.4.7-21+deb13u1+b5_amd64.deb`.
Its package SHA256
`104e4c57b98f0b883aa94c68fbe7ec7b48bcdec7c0f9dedf2a51a5054aaeb5ae`
matched the exact snapshot's `dists/trixie/main/binary-amd64/Packages.xz` entry.
Only the verifier executable was extracted for test tool setup; the tested APT
package payloads were never installed or executed. Its executable SHA256 was
`3f29dddc10e4089aeac5b2675313f4e5fe822abe5ab3bf7898d238c32d758304`.
This HTTPS/metadata comparison is a test reproducibility record. It does not
establish independent tool approval, complete dynamic-library closure or an
accepted OS sandbox. Those acceptance requirements remain UNVERIFIED.
