# Official Ubuntu source review — 2026-10-28

Researcher: Codex, using the connected web research tool. All sources below
were opened and read on 2026-10-28 UTC. This is a review note with short excerpts
and paraphrases, not an archived copy or cryptographic authentication of the
publisher's web pages. The sources are independent of the operator's supplied
local results; they share Canonical/Ubuntu's publisher trust domain.

## Source matrix

| ID | Official source and location | Short exact excerpt | Supported conclusion |
| --- | --- | --- | --- |
| U1 | [Ubuntu security documentation: image integrity verification, Step 2](https://documentation.ubuntu.com/security/software-integrity/image-verification/#step-2-verify-public-key-authenticity), displayed lines 153–155; page updated 2026-01-27 | `F6ECB3762474EDA9D21B7022871920D1991BC93C`; `Ubuntu Archive Automatic Signing Key (2018)` | The publisher prints this full fingerprint next to the 2018 archive signing key identity and rsa4096 creation date 2018-09-17. The same section describes bootstrapping key authenticity through a previously trusted channel. |
| U2 | [Chisel documentation: chisel.yaml example](https://ubuntu.com/chisel/docs/latest/reference/chisel-releases/chisel.yaml/#example), displayed lines 255–271; page updated 2026-08-19 | `rsa4096/f6ecb3762474eda9d21b7022871920d1991bc93c`; `id: "871920D1991BC93C"` | The official Noble 24.04 example associates noble, noble-security and noble-updates with ubuntu-archive-key-2018 and the same full fingerprint. |
| U3 | [Ubuntu Noble apt.conf(5), THE ACQUIRE GROUP](https://manpages.ubuntu.com/manpages/noble/man5/apt.conf.5.html), displayed lines 186–201; provided by apt 2.8.3 | `The default value is 0 which stands for "valid forever".` | Check-Date and Check-Valid-Until default to true; disabling Check-Date disables time checks. Expiry checks depend on a correct clock. Max-ValidTime defaults to zero; when a signed Valid-Until exists and a maximum age is configured, the earlier limit applies. Per-source options can change policy. These documented defaults do not establish the Dell's effective configuration. |
| U4 | [Ubuntu Noble sources.list(5), source options](https://manpages.ubuntu.com/manpages/noble/man5/sources.list.5.html), displayed lines 142–148 | `Valid-Until-Min`; `Valid-Until-Max`; `Check-Date` | Source-specific options can alter freshness checks. Valid-Until-Max can impose an age bound when Release has no Valid-Until. Check-Date disables time checks if turned off. Effective global, source and command-line settings must be collected. |
| U5 | [Ubuntu Noble apt-secure(8), SIGNED REPOSITORIES](https://manpages.ubuntu.com/manpages/noble/man8/apt-secure.8.html), displayed lines 67–78 | `apt-secure does not review signatures at a package level.` | The archive signs Release metadata; Release authenticates Packages checksums; Packages records authenticate downloaded DEB checksums. This is not an individual GPG signature on each DEB, and does not establish that package contents are free of malicious code. |
| U6 | [Ubuntu security documentation: archive integrity verification, Trust anchor](https://documentation.ubuntu.com/security/software-integrity/archive-verification/#trust-anchor), displayed lines 129–142 | `each configured APT source is validated against a trust anchor` | APT uses configured GPG keyrings; Signed-By may restrict a source, otherwise global trusted keys are used. Configuration and protection of those keys are separate from mathematical signature verification. |

## Narrow conclusion about the signing key

The exact fingerprint and published identity are `EXTERNALLY_CORROBORATED` by
U1 and U2. U1 identifies it as Ubuntu Archive Automatic Signing Key (2018),
`<ftpmaster@ubuntu.com>`, rsa4096, created 2018-09-17. U2 connects it to the
official Noble example. This corroboration does not authenticate the local
keyring bytes or reported keyring SHA-256, the supplied InRelease files,
signatures, Packages records, downloaded DEBs, or the Dell laptop.

Formal project acceptance remains `INDEPENDENT_ACCEPTANCE_PENDING`. An
accountable independent reviewer must record the chosen trusted channel, exact
accepted key bytes/fingerprint, authorization scope, review time, revocation and
rotation policy, and an explicit acceptance decision. Multiple pages published
by Canonical are not multiple independent institutional trust roots.

## Separate freshness assessment

Against report date 2026-10-28, the operator-reported noble-updates date
2026-09-14 is 44 calendar days earlier; noble-security 2026-10-27 is one
calendar day earlier. These are calendar-date differences, not exact elapsed
ages: full signed Date fields and collection timestamps were not supplied.

The older updates date warrants investigation. Its age alone proves neither a
security failure nor expiration under an accepted policy. Successful gpgv
verification proves neither currentness nor replay protection. Both pockets
remain `UNVERIFIED` for freshness until their evidence and applicable policy
are assessed. No current mirror contents or historical archive bytes were
downloaded or authenticated in this research.

Required evidence for that assessment:

1. Exact signed Date, Valid-Until value or demonstrated absence, and signature
   creation time for each supplied InRelease.
2. Collection/retrieval timestamps, archive or mirror identity and a trustworthy
   UTC clock source for the collection and validation context.
3. Effective APT configuration: global and per-source Check-Date,
   Check-Valid-Until, Max-/Min-ValidTime and corresponding source options,
   future-time bounds, APT_CONFIG and invocation overrides.
4. An accepted project maximum metadata age, replay/rollback rules and policy
   for intentionally retained historical snapshots.
5. Independent authenticated contemporaneous archive comparison where that
   policy requires it, without overwriting the historical evidence snapshot.

These requirements do not authorize running Observer, Stage 3B, physical Dell
tests, package installation, or a qualification-status change.
