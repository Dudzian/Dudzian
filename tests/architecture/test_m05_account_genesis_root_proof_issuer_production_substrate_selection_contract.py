"""Wykonywalny, profilowy kontrakt substrate Root-Proof Issuer."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).parents[2]; DOCS=ROOT/'docs/architecture/cryptohunter_product_architecture'
STEM='m05_account_genesis_root_proof_issuer_production_substrate_selection_contract'
MACHINE=DOCS/f'{STEM}.json'; MARKDOWN=DOCS/f'{STEM}.md'
REQUESTER='ROOT_PROOF_REQUESTER'; CLAIMANT='ROOT_PROOF_CLAIMANT'; ISSUER='ROOT_PROOF_ISSUER_SIGNING'
FRESHNESS='ACCOUNT_GENESIS_FRESHNESS_PROPOSER'; CATALOG='CATALOG_AUTHORITY'; STORAGE='STORAGE_SECURITY_KEY'
ROLES={REQUESTER,CLAIMANT,ISSUER,FRESHNESS,CATALOG,STORAGE}
REQUIRED_DISTINCT_PAIRS={frozenset(x) for x in [(REQUESTER,CLAIMANT),(REQUESTER,ISSUER),(REQUESTER,FRESHNESS),(REQUESTER,CATALOG),(REQUESTER,STORAGE),(CLAIMANT,ISSUER),(ISSUER,FRESHNESS),(ISSUER,CATALOG)]}
ALIAS_MUTATION_PAIRS={
 'requester_key_reused_as_issuer_signing_key':frozenset((REQUESTER,ISSUER)),
 'claimant_key_reused_as_requester_key':frozenset((CLAIMANT,REQUESTER)),
 'requester_key_reused_as_freshness_proposer_key':frozenset((REQUESTER,FRESHNESS)),
 'requester_key_reused_as_Catalog_authority_key':frozenset((REQUESTER,CATALOG)),
 'requester_key_reused_as_storage_security_key':frozenset((REQUESTER,STORAGE)),
 'issuer_signing_key_reused_as_freshness_proposer_key':frozenset((ISSUER,FRESHNESS)),
 'issuer_signing_key_reused_as_Catalog_authority_key':frozenset((ISSUER,CATALOG)),
 'claimant_key_reused_as_issuer_signing_key':frozenset((CLAIMANT,ISSUER)),
}
PROFILE_MUTATIONS={'production_local_requires_HSM','production_local_requires_remote_checkpoint','local_provider_marked_server_ready_by_name_only','local_checkpoint_claims_independent_rollback_domain','local_software_key_claims_hardware_backed','TEST_key_authorizes_PRODUCTION_LOCAL','TEST_key_authorizes_PRODUCTION_SERVER_READY','PRODUCTION_LOCAL_key_reused_as_SERVER_READY_key','local_to_server_ready_automatic_promotion','local_history_retroactively_marked_server_ready','insecure_runtime_override_allows_server_ready','server_ready_checkpoint_same_rollback_domain_as_registry','server_ready_signing_plaintext_export_allowed','production_local_marked_deprecated','production_server_ready_replaces_local','production_local_full_product_functionality_false','vendor_name_required_by_core','provider_capabilities_caller_asserted_without_qualification','local_checkpoint_missing_monotonic_head_binding','production_local_credential_role_aliasing'}
LIFECYCLE_MUTATIONS={'compromised_issuer_key_becomes_VERIFY_ONLY','compromised_requester_key_becomes_VERIFY_ONLY','compromised_claimant_key_becomes_VERIFY_ONLY','compromised_deployment_root_becomes_VERIFY_ONLY','compromised_checkpoint_key_becomes_VERIFY_ONLY','REVOKED_to_VERIFY_ONLY_allowed','REVOKED_to_ACTIVE_allowed_after_compromise','compromised_key_signature_alone_establishes_historical_acceptance','compromised_checkpoint_self_attests_recovery','restored_REVOKED_to_VERIFY_ONLY_state_accepted'}
SECURITY_MUTATIONS={'CHA_directly_mutates_entitlement_registry','issuer_directly_commits_account','ordinary_DB_durability_claimed_as_anti_rollback','root_proof_signing_key_plaintext_config','TEST_namespace_reused_for_PRODUCTION','process_local_CAS_selected_for_multi_host_registry','last_write_wins_selected_as_BIND','checkpoint_unavailable_but_issuance_allowed','registry_unavailable_but_signing_only_fallback_allowed','NOT_FOUND_used_as_authoritative_UNBOUND','restored_BOUND_to_UNBOUND_state_accepted','restored_REVOKED_to_ACTIVE_state_accepted','CHA_attempt_store_shares_authority_write_role_with_issuer_registry','unknown_schema_auto_migrated','caller_selects_entitlement_record','local_timestamp_used_to_resolve_split_brain'}
ALL_MUTATIONS=PROFILE_MUTATIONS|LIFECYCLE_MUTATIONS|SECURITY_MUTATIONS|set(ALIAS_MUTATION_PAIRS)
ACTIONS={'ISSUE','RECOVER','REPLACE','PREPARED'}

def load(): return json.loads(MACHINE.read_text(encoding='utf-8'))
def render(v):
 body=json.dumps(v,indent=2,ensure_ascii=False)+'\n'
 return f"# M0.5 — profilowy wybór production substrate dla Root-Proof Issuer\n\n> Deterministyczna projekcja pliku `{STEM}.json`. JSON jest źródłem prawdy.\n\n## Wynik\n\n**{v['principal_result']}**\n\nWybrano trwały, czteroprofilowy i niezależny od vendora model capability/provider. PRODUCTION_LOCAL ma pełny lokalny substrate produkcyjny, a PRODUCTION_SERVER_READY ma zamrożone silniejsze wymagania capability; brak jego konkretnego wdrożenia nie blokuje pozostałych profili i nie jest deklaracją implementacji.\n\n## Projekcja maszynowa\n\n```json\n{body}```\n"

def credential_distinct_pairs(v):
 pairs=v['credential_role_non_aliasing']['required_distinct_pairs']
 assert all(isinstance(p,list) and len(p)==2 for p in pairs)
 normalized={frozenset(p) for p in pairs}
 assert all(len(p)==2 for p in normalized)
 assert len(normalized)==len(pairs)
 return normalized

def validate_non_aliasing_exact(v):
 m=v['credential_role_non_aliasing']
 assert set(m['roles'])==ROLES
 assert credential_distinct_pairs(v)==REQUIRED_DISTINCT_PAIRS
 assert m['requester_role']['role_id']==REQUESTER and m['requester_role']['scope']=='dedicated pre-account credential'
 assert m['issuer_signing_role']['role_id']==ISSUER
 assert m['pair_semantics']=='unordered; absence of a pair never implies equality'
 assert set(m['distinct_dimensions'])=={'semantic role','credential identity','cryptographic key material or non-exportable key handle','custody role','lifecycle namespace'}
 assert m['logical_ACLs_on_same_key_material_satisfy_separation'] is False
 assert m['future_exception_protocol_exists'] is False
 assert set(m['forbidden_alias_forms'])=={'same private key bytes with two key IDs','same non-exportable key handle under two semantic aliases','same KMS/HSM key version used for distinct forbidden roles','same credential generation re-labelled between roles','same physical secret copied into multiple role namespaces'}
 assert set(m['implementation_readiness_identity_check'])=={'role credential ID','provider namespace','key version or key handle identity'}
 assert m['issuer_may_authenticate_requester_with_proof_signing_credential'] is False
 assert m['CHA_requester_obtains_issuer_signing_authority'] is False
 assert set(v['credential_role_separation'])=={f'{p[0]} != {p[1]}' for p in m['required_distinct_pairs']}

def validate_lifecycle_and_compromise(v):
 l=v['credential_lifecycle']; assert set(l['states'])=={'ACTIVE','VERIFY_ONLY','REVOKED'}
 assert l['planned_retirement_transition']=='ACTIVE -> VERIFY_ONLY'
 assert set(l['compromise_transitions'])=={'ACTIVE -> REVOKED','VERIFY_ONLY -> REVOKED'}
 assert l['compromise_may_result_in_VERIFY_ONLY'] is False and l['REVOKED_is_terminal'] is True
 assert l['REVOKED_to_VERIFY_ONLY_allowed'] is False and l['REVOKED_to_ACTIVE_allowed'] is False
 h=l['history']; assert h['preserve_states']==['ACTIVE','VERIFY_ONLY','REVOKED']
 assert h['preserve_compromised_generation_identity'] is True and h['preserve_REVOKED_transition'] is True
 assert h['delete_compromised_generation'] is False and h['reinterpret_REVOKED_as_VERIFY_ONLY'] is False and h['retaining_REVOKED_equals_trusting_its_signature'] is False
 p=l['historical_proof_for_REVOKED']; assert p['independent_trusted_evidence_required'] is True
 assert set(p['requirements'])=={'currently trusted authority root or state','exact same historical credential identity, generation and record','independently verifiable retained authenticated history','no compromised-key-only ultimate trust root','no self-corroboration by the compromised key','missing trusted root or evidence fails closed'}
 assert p['compromised_key_signature_alone_sufficient'] is False and p['compromised_key_self_corroboration_allowed'] is False
 assert p['wall_clock_establishes_direction'] is False and p['unavailable_evidence_result']=='FAIL_CLOSED / UNAVAILABLE' and p['acceptance_changes_lifecycle_state'] is False
 c=l['checkpoint_interaction']; assert c['checkpoint_makes_compromised_key_trustworthy_again'] is False
 assert c['compromised_checkpoint_self_attests_recovery'] is False and c['checkpoint_compromise_recovery_requires']=='independently trusted checkpoint/root lineage'
 assert l['roles']==v['key_compromise']; assert set(l['roles'])=={'claimant','requester','issuer_signing','deployment_trust_root','checkpoint'}
 for role in l['roles'].values():
  assert role['compromise_transition']=='ACTIVE|VERIFY_ONLY -> REVOKED' and role['resulting_state']=='REVOKED'
  assert role['compromised_key_signature_alone_sufficient'] is False and role['retain_generation_and_revocation_in_history'] is True
 r=v['restore']; assert r['REVOKED_to_VERIFY_ONLY_restore_allowed'] is False and r['REVOKED_to_ACTIVE_restore_allowed'] is False
 assert r['restored_VERIFY_ONLY_after_later_REVOKED']=='ROLLBACK_OR_TAMPER / FAIL_CLOSED'

def validate_profile_model(v):
 assert v['principal_result']=='ROOT_PROOF_ISSUER_PRODUCTION_SUBSTRATE_PROFILE_MODEL_CAN_BE_FROZEN'
 assert set(v['profiles'])=={'DEVELOPMENT','TEST','PRODUCTION_LOCAL','PRODUCTION_SERVER_READY'}
 assert all(x['architecture_supported'] and x['design_allowed'] and x['permanently_supported'] for x in v['profiles'].values())
 assert all(not x['implemented'] and not x['deployment_available_now'] for x in v['profiles'].values())
 p=v['product_profile_policy']; assert p=={'four_profiles_permanently_supported':True,'production_local_permanently_supported':True,'production_local_is_development_fallback':False,'production_local_deprecation_planned':False,'production_server_ready_replaces_local':False,'production_local_full_product_functionality':True}
 local=v['profiles']['PRODUCTION_LOCAL']; assert local['requires_HSM'] is False and local['requires_remote_checkpoint'] is False and local['full_host_rollback_resistance_claimed'] is False and local['full_product_functionality'] is True
 s=v['production_local_substrate']; assert s['registry']['engine']=='PostgreSQL' and s['signing']['algorithm']=='Ed25519' and s['signing']['provider_kind']=='LOCAL_SOFTWARE'
 assert s['signing']['hardware_backed'] is False and s['signing']['plaintext_private_key_in_config'] is False
 cp=s['checkpoint']; assert cp['authenticated'] and cp['monotonic_sequence'] and cp['exact_history_head_binding'] and not cp['independent_rollback_domain'] and not cp['coordinated_full_host_rollback_detection']
 server=v['production_server_ready_capabilities']; assert server['root_proof_signing']['hardware_backed_or_equivalent_secure_custody'] and not server['root_proof_signing']['plaintext_private_key_export']
 assert server['checkpoint']['independent_rollback_domain'] and server['checkpoint']['independent_admin_or_security_domain'] and server['checkpoint']['retained_authenticated_history'] and not server['checkpoint']['wall_clock_arbitration']
 assert v['status']['architecture_selection_complete'] and not v['status']['architecture_implementation_currently_exists']

def validate_provider_qualification(v):
 c=v['core_provider_model']; assert c['provider_agnostic'] and c['semantic_core_shared_by_all_profiles']
 assert not c['vendor_name_required'] and not c['caller_asserted_capabilities_accepted'] and not c['provider_name_or_environment_grants_eligibility']
 assert not c['concrete_HSM_vendor_frozen'] and not c['concrete_cloud_or_VPS_provider_frozen']
 q=v['provider_qualification']; assert not q['name_only_qualification_allowed'] and not q['insecure_runtime_override_exists']
 assert q['local_signing_eligible_profiles']==['DEVELOPMENT','TEST','PRODUCTION_LOCAL'] and q['local_checkpoint_eligible_profiles']==['DEVELOPMENT','TEST','PRODUCTION_LOCAL']
 i=v['profile_isolation']; assert not i['TEST_credential_authorizes_PRODUCTION_LOCAL'] and not i['TEST_credential_authorizes_PRODUCTION_SERVER_READY']
 assert not i['PRODUCTION_credentials_used_in_TEST'] and len(i['trust_domains_and_credential_namespaces_distinct'])==4

def validate_profile_migration(v):
 m=v['profile_migration']; assert m['protocol']=='REVIEWED_PRODUCTION_PROFILE_MIGRATION'
 for k in ['new_trust_domain_or_security_epoch_required','new_server_ready_signing_credentials_required','first_independent_server_ready_checkpoint_required','preserve_local_history_provenance','exact_guarantee_boundary_required']: assert m[k] is True
 for k in ['automatic_promotion_allowed','local_key_relabel_allowed','retroactive_server_ready_guarantee']: assert m[k] is False

def validate_registry_schema_and_CAS(v):
 r=v['production_local_substrate']['registry']; assert r['engine']=='PostgreSQL' and r['issuer_only_semantic_BIND_owner'] and not r['CHA_registry_write_authority'] and r['unknown_schema']=='FAIL_CLOSED'
 c=v['production_local_substrate']['serialization']; assert c['isolation']=='SERIALIZABLE' and c['immutable_authority_identity'] and c['lifecycle_generation'] and c['cas_revision']
 assert c['condition']=='state=UNBOUND AND generation=? AND cas_revision=?' and c['winner']=='exactly one affected row' and c['unique_authoritative_decision_constraint'] and not c['last_write_wins']
 b=v['registry_schema_boundary']; assert b['primary_authority_key']=='issuer-generated opaque authority_record_id; never caller selected'
 assert b['generation_key']=='(environment, trust_domain, entitlement_id, lifecycle_generation)'
 assert set(b['uniqueness'])=={'authority_record_id','(environment,trust_domain,entitlement_id,generation)','one immutable BOUND decision per generation'}
 assert b['current_state_pointer']=='authority-owned pointer with cas_revision' and 'EntitlementBoundDecision' in b['records']
 assert set(b['required_fields'])=={'environment','trust_domain','entitlement_id','cas_revision'}

def validate_CHA_attempt_store(v):
 a=v['cha_attempt_store']; assert a['selection_status']=='SELECTED' and a['database']=='dedicated SQLite database' and a['separate_from_issuer_registry']
 assert a['transaction_begin']=='BEGIN IMMEDIATE' and a['journal_mode']=='WAL' and a['synchronous']=='FULL'
 assert a['immutable_attempt_rows'] and a['fenced_current_pointer'] and a['path_namespace']=='separate per environment/trust_domain' and a['OS_filesystem_ACL'] and a['supported_local_filesystem_locking_required']
 assert not a['process_memory_allowed'] and not a['shared_issuer_schema_allowed'] and not a['network_filesystem_without_proven_SQLite_locking_allowed'] and not a['issuer_write_authority'] and not a['independent_anti_rollback']
 assert a['authority_owner']=='CryptoHunterAccountAuthority' and a['persistence_owner']=='CHA local installation'
 assert a['transactions']=={'A_initial_reservation':'BEGIN IMMEDIATE; insert immutable reservation + current fenced pointer atomically','B_immutable_finalization':'insert immutable finalized attempt + CAS/fenced pointer update in one transaction commit','C_replacement_switch':'insert replacement reservation/evidence relation + CAS pointer old fence -> new fence in one transaction','D_local_recovery':'append authenticated resolution reference + CAS current pointer; never infer UNBOUND from NOT_FOUND'}

def validate_outage_and_restore(v):
 expected={'registry DB unavailable','CAS timeout','signing custody unavailable','claimant registry unavailable','requester registry unavailable','checkpoint unavailable','checkpoint stale','checkpoint ahead','history gap','trust-root store unavailable','reconciliation source unavailable','CHA attempt store unavailable','network partition','restore detected','one region/host stale'}
 outage={x['condition']:x for x in v['outage_matrix']}; assert set(outage)==expected
 for row in outage.values(): assert all(row[a] is False for a in ACTIONS) and row['behavior'].startswith('fail closed')
 assert v['restore']['ambiguous_policy']=={a:False for a in ACTIONS}

def validate_authority_separation_and_topology(v):
 a=v['authority_separation']; assert a['rule']=='PHYSICAL COLOCATION != AUTHORITY REUSE' and not a['CHA_direct_registry_write'] and not a['issuer_direct_account_commit'] and not a['Issuer_to_CHA_AttemptStore_write'] and not a['CHA_to_Issuer_Registry_semantic_BIND_write']
 assert set(a['required_separation'])=={'schemas/namespaces','credentials','write permissions','authority APIs','transaction ownership','lifecycle','TEST/PRODUCTION profile boundaries'}
 edges={(e['from'],e['to']):e for e in v['topology']['edges']}; assert edges[('Issuer','CHA Attempt Store')]['write'] is False and edges[('CHA','Issuer')]['write'] is False
 l=v['least_privilege']; assert 'never bind registry' in l['CHA'] and 'never commit account' in l['Issuer'] and 'never issue root proof' in l['CheckpointAuthority'] and 'no runtime CHA impersonation' in l['SecurityAuthority']

def validate_reconciliation_semantics(v):
 r=v['reconciliation_semantics']; assert r['status']=='FROZEN_PROFILED' and r['mechanism']=='read-only authenticated RootProofIssuer API over authoritative registry'
 assert r['positive_only']=='AUTHORITATIVELY_UNBOUND authenticated/signed authoritative evidence only' and r['NOT_FOUND']=='OUTCOME_UNKNOWN' and not r['NOT_FOUND_means_AUTHORITATIVELY_UNBOUND']
 assert set(r['binding'])=={'exact decision revision','canonical evidence bytes or immutable reference','digest','checkpoint/history head','environment','trust_domain','issuer authority identity','entitlement_id and lifecycle_generation','operation/account/request/attempt identity'}
 assert not (v['status']['architecture_selection_complete'] and r['status']=='DECISION_REQUIRED')

def validate_split_brain_and_migrations(v):
 s=v['split_brain']; assert 'global DB CAS' in s['two_issuer_processes'] and 'no local winner' in s['two_issuer_hosts'] and 'SQLite fenced pointer' in s['two_CHA_processes'] and s['timestamp_arbitration'] is False and s['custody_without_registry'].startswith('never sign')
 m=v['migrations']; assert m['unknown_schema']=='fail closed; never implicit upgrade'
 assert set(m['preserve'])=={'IDs','signature bytes','canonical digest interpretation','entitlement generations','reconciliation evidence','environment/trust_domain namespaces','historical key references'}

def validate(v):
 for fn in [validate_profile_model,validate_provider_qualification,validate_profile_migration,validate_non_aliasing_exact,validate_lifecycle_and_compromise,validate_registry_schema_and_CAS,validate_CHA_attempt_store,validate_outage_and_restore,validate_authority_separation_and_topology,validate_reconciliation_semantics,validate_split_brain_and_migrations]: fn(v)
 assert set(v['redteam_mutations'])==ALL_MUTATIONS
 assert v['provenance']=={'classification':'UNKNOWN','finding_scope':'CURRENT_TREE_ONLY','formal_project_advancement':'WITHHELD'}

def mutate(v,name):
 r=deepcopy(v); local=r['profiles']['PRODUCTION_LOCAL']; ps=r['production_local_substrate']; server=r['production_server_ready_capabilities']; iso=r['profile_isolation']; mig=r['profile_migration']
 if name=='production_local_requires_HSM': local['requires_HSM']=True
 elif name=='production_local_requires_remote_checkpoint': local['requires_remote_checkpoint']=True
 elif name=='local_provider_marked_server_ready_by_name_only': r['provider_qualification']['name_only_qualification_allowed']=True
 elif name=='local_checkpoint_claims_independent_rollback_domain': ps['checkpoint']['independent_rollback_domain']=True
 elif name=='local_software_key_claims_hardware_backed': ps['signing']['hardware_backed']=True
 elif name=='TEST_key_authorizes_PRODUCTION_LOCAL': iso['TEST_credential_authorizes_PRODUCTION_LOCAL']=True
 elif name=='TEST_key_authorizes_PRODUCTION_SERVER_READY': iso['TEST_credential_authorizes_PRODUCTION_SERVER_READY']=True
 elif name=='PRODUCTION_LOCAL_key_reused_as_SERVER_READY_key': mig['local_key_relabel_allowed']=True
 elif name=='local_to_server_ready_automatic_promotion': mig['automatic_promotion_allowed']=True
 elif name=='local_history_retroactively_marked_server_ready': mig['retroactive_server_ready_guarantee']=True
 elif name=='insecure_runtime_override_allows_server_ready': r['provider_qualification']['insecure_runtime_override_exists']=True
 elif name=='server_ready_checkpoint_same_rollback_domain_as_registry': server['checkpoint']['independent_rollback_domain']=False
 elif name=='server_ready_signing_plaintext_export_allowed': server['root_proof_signing']['plaintext_private_key_export']=True
 elif name=='production_local_marked_deprecated': r['product_profile_policy']['production_local_deprecation_planned']=True
 elif name=='production_server_ready_replaces_local': r['product_profile_policy']['production_server_ready_replaces_local']=True
 elif name=='production_local_full_product_functionality_false': r['product_profile_policy']['production_local_full_product_functionality']=False
 elif name=='vendor_name_required_by_core': r['core_provider_model']['vendor_name_required']=True
 elif name=='provider_capabilities_caller_asserted_without_qualification': r['core_provider_model']['caller_asserted_capabilities_accepted']=True
 elif name=='local_checkpoint_missing_monotonic_head_binding': ps['checkpoint']['exact_history_head_binding']=False
 elif name in ALIAS_MUTATION_PAIRS or name=='production_local_credential_role_aliasing':
  pair=ALIAS_MUTATION_PAIRS.get(name,frozenset((REQUESTER,ISSUER))); pairs=r['credential_role_non_aliasing']['required_distinct_pairs']; removed=next(x for x in pairs if frozenset(x)==pair); pairs.remove(removed); r['credential_role_separation'].remove(f'{removed[0]} != {removed[1]}')
 elif name.startswith('compromised_') and name.endswith('_becomes_VERIFY_ONLY'):
  role={'compromised_issuer_key_becomes_VERIFY_ONLY':'issuer_signing','compromised_requester_key_becomes_VERIFY_ONLY':'requester','compromised_claimant_key_becomes_VERIFY_ONLY':'claimant','compromised_deployment_root_becomes_VERIFY_ONLY':'deployment_trust_root','compromised_checkpoint_key_becomes_VERIFY_ONLY':'checkpoint'}[name]; r['credential_lifecycle']['roles'][role]['resulting_state']='VERIFY_ONLY'
 elif name=='REVOKED_to_VERIFY_ONLY_allowed': r['credential_lifecycle']['REVOKED_to_VERIFY_ONLY_allowed']=True
 elif name=='REVOKED_to_ACTIVE_allowed_after_compromise': r['credential_lifecycle']['REVOKED_to_ACTIVE_allowed']=True
 elif name=='compromised_key_signature_alone_establishes_historical_acceptance': r['credential_lifecycle']['historical_proof_for_REVOKED']['compromised_key_signature_alone_sufficient']=True
 elif name=='compromised_checkpoint_self_attests_recovery': r['credential_lifecycle']['checkpoint_interaction']['compromised_checkpoint_self_attests_recovery']=True
 elif name=='restored_REVOKED_to_VERIFY_ONLY_state_accepted': r['restore']['REVOKED_to_VERIFY_ONLY_restore_allowed']=True
 elif name=='CHA_directly_mutates_entitlement_registry': r['authority_separation']['CHA_direct_registry_write']=True
 elif name=='issuer_directly_commits_account': r['authority_separation']['issuer_direct_account_commit']=True
 elif name=='ordinary_DB_durability_claimed_as_anti_rollback': r['cha_attempt_store']['independent_anti_rollback']=True
 elif name=='root_proof_signing_key_plaintext_config': ps['signing']['plaintext_private_key_in_config']=True
 elif name=='TEST_namespace_reused_for_PRODUCTION': iso['TEST_credential_authorizes_PRODUCTION_LOCAL']=True
 elif name=='process_local_CAS_selected_for_multi_host_registry': ps['serialization']['isolation']='process mutex'
 elif name=='last_write_wins_selected_as_BIND': ps['serialization']['last_write_wins']=True
 elif name=='checkpoint_unavailable_but_issuance_allowed': next(x for x in r['outage_matrix'] if x['condition']=='checkpoint unavailable')['ISSUE']=True
 elif name=='registry_unavailable_but_signing_only_fallback_allowed': next(x for x in r['outage_matrix'] if x['condition']=='registry DB unavailable')['ISSUE']=True
 elif name=='NOT_FOUND_used_as_authoritative_UNBOUND': r['reconciliation_semantics']['NOT_FOUND']='AUTHORITATIVELY_UNBOUND'
 elif name=='restored_BOUND_to_UNBOUND_state_accepted': r['restore']['ambiguous_policy']['ISSUE']=True
 elif name=='restored_REVOKED_to_ACTIVE_state_accepted': r['restore']['ambiguous_policy']['PREPARED']=True
 elif name=='CHA_attempt_store_shares_authority_write_role_with_issuer_registry': r['cha_attempt_store']['issuer_write_authority']=True
 elif name=='unknown_schema_auto_migrated': r['migrations']['unknown_schema']='implicit auto-upgrade'
 elif name=='caller_selects_entitlement_record': r['registry_schema_boundary']['primary_authority_key']='caller selected'
 elif name=='local_timestamp_used_to_resolve_split_brain': r['split_brain']['timestamp_arbitration']=True
 else: raise AssertionError(name)
 return r

SPECIALIZED={**{x:validate_profile_model for x in PROFILE_MUTATIONS},**{x:validate_lifecycle_and_compromise for x in LIFECYCLE_MUTATIONS},**{x:validate_non_aliasing_exact for x in ALIAS_MUTATION_PAIRS},
 'production_local_credential_role_aliasing':validate_non_aliasing_exact,'CHA_directly_mutates_entitlement_registry':validate_authority_separation_and_topology,'issuer_directly_commits_account':validate_authority_separation_and_topology,'ordinary_DB_durability_claimed_as_anti_rollback':validate_CHA_attempt_store,'root_proof_signing_key_plaintext_config':validate_profile_model,'TEST_namespace_reused_for_PRODUCTION':validate_provider_qualification,'process_local_CAS_selected_for_multi_host_registry':validate_registry_schema_and_CAS,'last_write_wins_selected_as_BIND':validate_registry_schema_and_CAS,'checkpoint_unavailable_but_issuance_allowed':validate_outage_and_restore,'registry_unavailable_but_signing_only_fallback_allowed':validate_outage_and_restore,'NOT_FOUND_used_as_authoritative_UNBOUND':validate_reconciliation_semantics,'restored_BOUND_to_UNBOUND_state_accepted':validate_outage_and_restore,'restored_REVOKED_to_ACTIVE_state_accepted':validate_outage_and_restore,'CHA_attempt_store_shares_authority_write_role_with_issuer_registry':validate_CHA_attempt_store,'unknown_schema_auto_migrated':validate_split_brain_and_migrations,'caller_selects_entitlement_record':validate_registry_schema_and_CAS,'local_timestamp_used_to_resolve_split_brain':validate_split_brain_and_migrations}
# Profile mutations whose exact guard lives outside profile validator.
SPECIALIZED.update({'local_provider_marked_server_ready_by_name_only':validate_provider_qualification,'TEST_key_authorizes_PRODUCTION_LOCAL':validate_provider_qualification,'TEST_key_authorizes_PRODUCTION_SERVER_READY':validate_provider_qualification,'insecure_runtime_override_allows_server_ready':validate_provider_qualification,'vendor_name_required_by_core':validate_provider_qualification,'provider_capabilities_caller_asserted_without_qualification':validate_provider_qualification,'PRODUCTION_LOCAL_key_reused_as_SERVER_READY_key':validate_profile_migration,'local_to_server_ready_automatic_promotion':validate_profile_migration,'local_history_retroactively_marked_server_ready':validate_profile_migration})

def test_contract_and_deterministic_projection():
 v=load(); validate(v); assert MARKDOWN.read_text(encoding='utf-8')==render(v)
@pytest.mark.parametrize('mutation',sorted(ALL_MUTATIONS))
def test_each_mutation_reaches_specialized_validator(mutation):
 original=load(); changed=mutate(original,mutation); assert changed!=original
 with pytest.raises(AssertionError): SPECIALIZED[mutation](changed)
@pytest.mark.parametrize('pair',sorted(REQUIRED_DISTINCT_PAIRS,key=lambda x:sorted(x)))
def test_removing_each_required_pair_is_rejected(pair):
 v=load(); item=next(x for x in v['credential_role_non_aliasing']['required_distinct_pairs'] if frozenset(x)==pair); v['credential_role_non_aliasing']['required_distinct_pairs'].remove(item); v['credential_role_separation'].remove(f'{item[0]} != {item[1]}')
 with pytest.raises(AssertionError): validate_non_aliasing_exact(v)
def test_duplicate_unordered_pair_rejected():
 v=load(); a,b=v['credential_role_non_aliasing']['required_distinct_pairs'][0]; v['credential_role_non_aliasing']['required_distinct_pairs'].append([b,a])
 with pytest.raises(AssertionError): credential_distinct_pairs(v)
def test_self_pair_rejected():
 v=load(); v['credential_role_non_aliasing']['required_distinct_pairs'].append([REQUESTER,REQUESTER])
 with pytest.raises(AssertionError): credential_distinct_pairs(v)
@pytest.mark.parametrize('mutation',sorted(ALIAS_MUTATION_PAIRS))
def test_alias_mutation_changes_exactly_named_pair(mutation):
 original=credential_distinct_pairs(load()); changed=credential_distinct_pairs(mutate(load(),mutation)); assert original-changed=={ALIAS_MUTATION_PAIRS[mutation]}; assert changed-original==set()
def test_requester_issuer_mutation_preserves_claimant_issuer_pair(): assert frozenset((CLAIMANT,ISSUER)) in credential_distinct_pairs(mutate(load(),'requester_key_reused_as_issuer_signing_key'))
def test_planned_retirement_transition_mutation_rejected():
 v=load(); v['credential_lifecycle']['planned_retirement_transition']='ACTIVE -> REVOKED'
 with pytest.raises(AssertionError): validate_lifecycle_and_compromise(v)
def test_key_compromise_matrix_mismatch_rejected():
 v=load(); v['key_compromise']['claimant']['resulting_state']='VERIFY_ONLY'
 with pytest.raises(AssertionError): validate_lifecycle_and_compromise(v)
@pytest.mark.parametrize('field', ['transaction_begin','journal_mode','synchronous','fenced_current_pointer'])
def test_attempt_store_required_substrate_field_rejected(field):
 v=load(); v['cha_attempt_store'][field]=False
 with pytest.raises(AssertionError): validate_CHA_attempt_store(v)
def test_reconciliation_decision_required_rejected_after_selection():
 v=load(); v['reconciliation_semantics']['status']='DECISION_REQUIRED'
 with pytest.raises(AssertionError): validate_reconciliation_semantics(v)
@pytest.mark.parametrize('condition',sorted({x['condition'] for x in load()['outage_matrix']}))
@pytest.mark.parametrize('action',sorted(ACTIONS))
def test_every_ambiguous_outage_action_is_fail_closed(condition,action):
 v=load(); next(x for x in v['outage_matrix'] if x['condition']==condition)[action]=True
 with pytest.raises(AssertionError): validate_outage_and_restore(v)
