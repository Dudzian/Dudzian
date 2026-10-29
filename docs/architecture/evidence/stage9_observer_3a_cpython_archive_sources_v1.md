# Ubuntu CPython exact archive material verification V1

Date: **2026-10-29 UTC**. This records an independent cloud read of the current
Ubuntu archive snapshot, separate from the historical operator report. Status:
`VERIFIED_FROM_EXACT_BUILD` for the exact source/binary package membership and
hash/size checks below, conditional on the stated archive-key identity. Formal
project trust-root acceptance and qualification remain pending. This label does
not prove source-to-binary reproducibility or that these bytes were measured on
the Dell.

## Acquisition and trust boundary

The base URL is `https://archive.ubuntu.com/ubuntu/`. Retained URLs are:

- `dists/noble-updates/InRelease`
- `dists/noble-updates/main/binary-amd64/Packages.xz`
- `dists/noble-updates/main/source/Sources.xz`
- `pool/main/u/ubuntu-keyring/ubuntu-keyring_2023.11.28.1_all.deb`
- Each exact `filename` recorded in the adjacent [verification JSON](stage9_observer_3a_cpython_archive_verification_v1.json).

Downloads used HTTPS, inherited proxy/CA trust and `curl --fail --location
--silent --show-error`; no package was installed. `dpkg-deb -x` extracted the
keyring into the scratch directory. The keyring member
`usr/share/keyrings/ubuntu-archive-keyring.gpg` has SHA-256
`80a36b0a6de2f69f49d2df75ef473ccde121e9e190b9ea01d20a4f63778d5c31`.
That equals the historical operator-reported identifier; equality does not
replay or authenticate the historical collection.

The full signing fingerprint is
`F6ECB3762474EDA9D21B7022871920D1991BC93C`. It matches the published
**Ubuntu Archive Automatic Signing Key (2018)** identity in the existing
[official-source review](stage9_observer_3a_ubuntu_official_sources_v1.md),
including the linked [Ubuntu integrity documentation](https://documentation.ubuntu.com/security/software-integrity/image-verification/#step-2-verify-public-key-authenticity).
This is an externally corroborated key identity, not a project approval. The
keyring DEB was obtained over official HTTPS, not authenticated through a
pre-existing project-accepted root. A reviewer must explicitly accept the
bootstrap channel/key if using this chain as authorization evidence.

`gpgv` was unavailable. GnuPG **2.4.7**, libgcrypt **1.11.0**, verified the
clearsigned metadata in an isolated empty home, without a default keyring:

```sh
mkdir -m 700 gnupg
gpg --batch --homedir gnupg --no-default-keyring \
  --keyring ./keyring/usr/share/keyrings/ubuntu-archive-keyring.gpg \
  --status-fd 1 --verify noble-updates.InRelease
```

Exit code **0**, `GOODSIG` and `VALIDSIG` were observed for the full fingerprint.
The signature timestamp is **2026-10-29 07:16:08 UTC**, hash algorithm ID 10
(SHA-512). GnuPG reported `TRUST_UNDEFINED` and the expected warning that the
key is not certified with a trusted signature. The fingerprint match is the
explicit identity check; ownertrust was not set. No `.dsc` uploader signature
verification is claimed: its bytes are authenticated through the signed
`Sources` index hash chain.

## Exact hash chain

The machine-readable JSON retains the signature status, exact package/source
stanzas, their individual SHA-256 and byte lengths, and all verified records.
The InRelease SHA-256 is
`89ad2f985a8959b207f598f833feb5f852d145be5bd4efcb97c4934ecb45723e`
(**126125 bytes**).

| InRelease SHA256 member | Size | SHA-256 | Result |
| --- | ---: | --- | --- |
| `main/binary-amd64/Packages.xz` | 1406036 | `8965447433642cb1c1b73a90d7bdd0cb353153583572bd9bed641fb716827c49` | PASS |
| `main/binary-amd64/Packages` | 7607567 | `6178eea98a941888639f49ccc9a88f6088d5efbed70b605bc3701fa894d611d3` | PASS |
| `main/source/Sources.xz` | 579736 | `89c14c026e564f4b2a116bc168df1fd1c854b01a27603789408f660598dcd50e` | PASS |
| `main/source/Sources` | 2936200 | `937214ffc2c76d64feb26478bbe6ee6ee334328a2dce007064b4a954d3161783` | PASS |

Both downloaded compressed index representations and decompressed index bytes
matched their separate signed SHA-256/size records: **4/4 PASS**. Exact
Package+Version+Architecture stanzas were selected from the verified full
binary index. Exact Package+Version source stanza was selected from the
verified full source index. No package-control dependency was treated as an
ELF dependency or as evidence of required Observer imports.

| Downloaded exact binary archive | Size | SHA-256 | Result |
| --- | ---: | --- | --- |
| `libpython3.12-dev_3.12.3-1ubuntu0.17_amd64.deb` | 5683746 | `c123ddab7763e45199e21b4ac8545e705f20a0315b9ada4c236b52fb088c4e16` | PASS |
| `libpython3.12-minimal_3.12.3-1ubuntu0.17_amd64.deb` | 838536 | `d646ad7112b5adec21ba0e1af015f04ae8ca7f5efdc619622547dd6673c4c14b` | PASS |
| `libpython3.12-stdlib_3.12.3-1ubuntu0.17_amd64.deb` | 2070530 | `45d3f530ba1f9d6e879ad46b92046fabab13fe50a82450e4f65557a3bbad1489` | PASS |
| `python3.12-minimal_3.12.3-1ubuntu0.17_amd64.deb` | 2334634 | `d452689b9660845345a4c3e05e4ad82c082d5474e04031b7aa47f1a6d5610a6e` | PASS |

| Downloaded exact source archive/control | Size | SHA-256 | Result |
| --- | ---: | --- | --- |
| `python3.12_3.12.3.orig.tar.xz` | 20625068 | `56bfef1fdfc1221ce6720e43a661e3eb41785dd914ce99698d8c7896af4bdaa1` | PASS |
| `python3.12_3.12.3-1ubuntu0.17.debian.tar.xz` | 302472 | `17b15629cb10825924cf7b85055255080ec773133ae0f8b8ce5ea2eec90dd200` | PASS |
| `python3.12_3.12.3-1ubuntu0.17.dsc` | 3916 | `941700b999737d869b80c3b5a0adc0996a0fd760b982e18551bd2104da779384` | PASS |

Every row was compared against the authenticated index record, including byte
length. Result: **4/4 DEBs, 3/3 source files PASS**. The three runtime
package digests equal the earlier operator report; the developer package is
additional analysis material and is not a runtime requirement.

## Snapshot time and replay limits

The signed metadata says `Origin: Ubuntu`, `Suite: noble-updates`,
`Codename: noble`, `Date: Thu, 29 Oct 2026  7:15:18 UTC`. No `Valid-Until`
field occurs in this captured InRelease. Cloud acquisition/verification was
recorded at **2026-10-29 11:21:52 UTC**. That is about four hours after the
signed Date; it uses the cloud clock and is not independently attested trusted
time. A successful signature and a recent date do not establish the project's
accepted freshness/replay policy.

These current metadata hashes differ from the historical operator snapshot.
This verification does not retroactively accept its dates, keyring, retrieval
context or freshness, and does not overwrite historical evidence. Full signed
metadata and compressed indexes remain in the analysis workspace; the retained
JSON gives the exact digests needed to identify them. Standalone stanza text
cannot independently prove its inclusion in an index without that full index.
No target interpreter, shared object, Observer, package maintainer script or
loader diagnostic was executed. `BUILD_SPECIFIC_UNVERIFIED` remains appropriate
for omitted build logs/`.buildinfo`, reproducible source-to-binary linkage and
any unmeasured target installation details.
