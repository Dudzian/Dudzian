"""Real PostgreSQL security proofs for the low-level FreshnessAuthority core."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import base64, hashlib, json, os, subprocess
import psycopg
from psycopg.errors import InsufficientPrivilege
import pytest

from bot_core.postgresql_freshness_authority import (
    DOCUMENT_DOMAIN, FINALIZATION_ROLE, PREPARATION_FIELDS, PROPOSER_ROLE,
    FreshnessAuthorityQualificationError, PostgreSQLConnectionConfig,
    canonical_json_bytes, complete_semantic_head_digest,
    normalize_complete_semantic_head_set, parse_canonical_json,
    provision_postgresql_freshness_authority, qualify_postgresql_freshness_authority,
)
DSN=os.environ.get("DUDZIAN_TEST_POSTGRES_DSN","host=127.0.0.1 port=55432 dbname=postgres user=postgres")
CFG=PostgreSQLConnectionConfig(DSN)
ROLES=("freshness_reader","freshness_runtime","freshness_crypto_verifier","freshness_admin","freshness_function_owner","freshness_schema_owner")
S0=[{"domain":"catalog","digest":"0"*64}]
S1=[{"domain":"entitlement","digest":"2"*64},{"domain":"catalog","digest":"1"*64}]
H0=complete_semantic_head_digest(S0)
H1=complete_semantic_head_digest(S1)

def _super(sql,args=(),fetch=False):
    with psycopg.connect(DSN,autocommit=True) as c:
        q=c.execute(sql,args); return q.fetchall() if fetch else None

def _admin(sql,args=()):
    with psycopg.connect(DSN,autocommit=True) as c:
        c.execute("SET SESSION AUTHORIZATION freshness_admin")
        return c.execute(sql,args).fetchone()

def _role(role): return psycopg.connect(DSN.replace("user=postgres",f"user={role}"))

def _cleanup():
    _super("DROP SCHEMA IF EXISTS freshness_authority CASCADE")
    for role in ROLES:
        if _super("SELECT 1 FROM pg_roles WHERE rolname=%s",(role,),True):
            _super(f'DROP OWNED BY "{role}" CASCADE')
            _super(f'DROP ROLE "{role}"')

@pytest.fixture(scope="module",autouse=True)
def authority():
    _cleanup(); provision_postgresql_freshness_authority(CFG); yield; _cleanup()

def _credential(credential,identity,role,key_id,octet):
    return {"security_profile":"PRODUCTION_LOCAL","environment":"PRODUCTION","trust_domain":"td","authority_id":"auth","credential_id":credential,"semantic_identity":identity,"semantic_role":role,"key_id":key_id,"key_version":1,"public_key_hex":(bytes([octet])*32).hex()}

def _seed(scope="auth",suffix=""):
    _admin("select freshness_authority.provision_authority(%s,%s,%s,%s,%s)",("PRODUCTION","td",scope,"a"*64,H0))
    for x in (_credential("prop"+suffix,"identity-A",PROPOSER_ROLE,"pk",1),_credential("fin"+suffix,"identity-F",FINALIZATION_ROLE,"fk",2)):
        x["authority_id"]=scope; _admin("select freshness_authority.provision_credential(%s)",(json.dumps(x),))

def _candidate(pid="prep",authority="auth",prop="prop",fin="fin",request="req",*,generation=1,predecessor_digest="a"*64,predecessor_head=H0,head_set=S1,proposer_key_id="pk",proposer_octet=1,finalization_key_id="fk",finalization_octet=2):
    normalized_heads=normalize_complete_semantic_head_set(head_set)
    payload={"schema_version":1,"environment":"PRODUCTION","trust_domain":"td","authority_id":authority,"generation":generation,"predecessor_generation":generation-1,"predecessor_document_digest":predecessor_digest,"complete_semantic_head_set":normalized_heads,"freshness_authority_key_id":finalization_key_id,"freshness_authority_key_version":1,"finalization_request_id":request}
    digest=hashlib.sha256(DOCUMENT_DOMAIN+canonical_json_bytes(payload)).hexdigest(); sig=base64.urlsafe_b64encode(b"x"*64).rstrip(b"=").decode()
    doc={"payload":payload,"document_digest":digest,"authentication_tag_or_signature":sig}
    receipt={"schema_version":1,"environment":"PRODUCTION","trust_domain":"td","authority_id":authority,"exact_predecessor_generation":generation-1,"exact_predecessor_document_digest":predecessor_digest,"accepted_generation":generation,"accepted_document_digest":digest,"complete_semantic_head_digest":complete_semantic_head_digest(normalized_heads),"finalization_request_id":request,"receipt_id":"receipt"+pid,"freshness_authority_key_id":finalization_key_id,"freshness_authority_key_version":1,"authentication_tag_or_signature":sig}
    prep={"schema_version":1,"security_profile":"PRODUCTION_LOCAL","environment":"PRODUCTION","trust_domain":"td","authority_id":authority,"operation_type":"FULL_AUTHORITATIVE_DOCUMENT","expected_predecessor_generation":generation-1,"expected_predecessor_document_digest":predecessor_digest,"expected_predecessor_complete_semantic_head_digest":predecessor_head,"proposed_document_digest":digest,"original_decision_identity":"odi"+pid,"proposer_identity":"identity-A","proposer_credential_role_identity":prop,"proposer_key_version":1,"proposer_lifecycle_generation_observed_for_key_binding_only":1,"proposer_public_key_material_identity":hashlib.sha256(bytes([proposer_octet])*32).hexdigest(),"proposer_authentication_digest":"c"*64,"finalization_credential_role_identity":fin,"finalization_key_version":1,"finalization_lifecycle_generation_observed_for_key_binding_only":1,"finalization_public_key_material_identity":hashlib.sha256(bytes([finalization_octet])*32).hexdigest(),"authoritative_document_authentication_digest":"d"*64,"receipt_id":receipt["receipt_id"],"receipt_canonical_digest":hashlib.sha256(canonical_json_bytes(receipt)).hexdigest(),"finalization_request_id":request,"preparation_id":pid,"verifier_authority_identity":"ver","verifier_authority_version":1}
    assert set(prep)==PREPARATION_FIELDS
    return prep,doc,receipt

def _prepare(values):
    p,d,r=values
    with _role("freshness_crypto_verifier") as c:
        return c.execute("select freshness_authority.prepare_verified_freshness_candidate(%s,%s,%s)",(canonical_json_bytes(p),canonical_json_bytes(d),canonical_json_bytes(r))).fetchone()[0]

def _prepare_raw(preparation: bytes, document: bytes, receipt: bytes):
    with _role("freshness_crypto_verifier") as c:
        return c.execute("select freshness_authority.prepare_verified_freshness_candidate(%s,%s,%s)",(preparation,document,receipt)).fetchone()[0]

def _rebind(values):
    p,d,r=values
    document_digest=hashlib.sha256(DOCUMENT_DOMAIN+canonical_json_bytes(d["payload"])).hexdigest()
    d["document_digest"]=document_digest
    p["proposed_document_digest"]=document_digest
    r["accepted_document_digest"]=document_digest
    if type(d["payload"].get("complete_semantic_head_set")) is list:
        r["complete_semantic_head_digest"]=complete_semantic_head_digest(d["payload"]["complete_semantic_head_set"])
    p["receipt_canonical_digest"]=hashlib.sha256(canonical_json_bytes(r)).hexdigest()
    return p,d,r

def _cas(values,isolation="serializable"):
    p,d,r=values
    with _role("freshness_runtime") as c:
        c.execute(f"set transaction isolation level {isolation}")
        row=c.execute("select * from freshness_authority.compare_and_advance(%s,%s,%s,%s)",(p["preparation_id"],json.dumps(p),canonical_json_bytes(d),canonical_json_bytes(r))).fetchone(); c.commit(); return row

def test_fresh_provisioning_qualification_roles_owner_oids_and_exact_acl():
    qualify_postgresql_freshness_authority(CFG)
    rows=_super("SELECT rolname,rolcanlogin,rolinherit,rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls,oid FROM pg_roles WHERE rolname=ANY(%s)",(list(ROLES),),True)
    assert {r[0] for r in rows if r[1]}=={"freshness_crypto_verifier","freshness_runtime"}; assert all(not any(r[2:8]) for r in rows); assert len({r[8] for r in rows})==6

def test_raw_dml_denied_runtime_verifier_admin_and_offline_admin_boundary():
    _seed()
    for role in ("freshness_runtime","freshness_crypto_verifier"):
        with _role(role) as c, pytest.raises(InsufficientPrivilege): c.execute("insert into freshness_authority.authority_lineages values('PRODUCTION','x','x',0,%s,%s,0)",("a"*64,"b"*64))
    with psycopg.connect(DSN) as c:
        c.execute("SET SESSION AUTHORIZATION freshness_admin")
        with pytest.raises(InsufficientPrivilege): c.execute("delete from freshness_authority.credentials")
    for role in ("freshness_runtime","freshness_crypto_verifier"):
        with _role(role) as c, pytest.raises(InsufficientPrivilege): c.execute("select freshness_authority.transition_credential('PRODUCTION','td','auth','prop',1,'REVOKED','{}')")

def test_verifier_has_only_reviewed_read_functions_and_no_raw_select():
    _seed("read-boundary")
    with _role("freshness_crypto_verifier") as c:
        credential=c.execute(
            "select * from freshness_authority.resolve_verification_credential(%s,%s,%s,%s,%s,%s)",
            ("PRODUCTION","td","read-boundary",PROPOSER_ROLE,"pk",1),
        ).fetchone()
        assert credential[:3]==("prop","identity-A",1) and bytes(credential[3])==bytes([1])*32
        assert c.execute(
            "select freshness_authority.resolve_predecessor_head(%s,%s,%s,%s,%s)",
            ("PRODUCTION","td","read-boundary",0,"a"*64),
        ).fetchone()==(H0,)
        with pytest.raises(InsufficientPrivilege): c.execute("select * from freshness_authority.credentials")
        c.rollback()
        with pytest.raises(InsufficientPrivilege): c.execute("select * from freshness_authority.authority_generation_heads")
    for role in ("freshness_crypto_verifier","freshness_runtime"):
        with _role(role) as c, pytest.raises(InsufficientPrivilege):
            c.execute("delete from freshness_authority.authority_generation_heads")
    with psycopg.connect(DSN) as c:
        c.execute("SET SESSION AUTHORIZATION freshness_admin")
        with pytest.raises(InsufficientPrivilege):
            c.execute("update freshness_authority.authority_generation_heads set complete_head_digest=%s",("f"*64,))

def test_preparation_exact_retry_is_idempotent_and_conflict_is_fail_closed():
    _seed("prepare-replay")
    exact=_candidate("prepare-replay",authority="prepare-replay",prop="prop",fin="fin")
    assert _prepare(exact)==_prepare(exact)=="prepare-replay"
    assert _super("select count(*) from freshness_authority.prepared_verifications where preparation_id='prepare-replay'",fetch=True)==[(1,)]
    conflicting=deepcopy(exact)
    conflicting[0]["verifier_authority_identity"]="other-verifier"
    with pytest.raises(psycopg.errors.UniqueViolation,match="preparation identity conflict"):
        _prepare(conflicting)
    assert _super("select binding->>'verifier_authority_identity' from freshness_authority.prepared_verifications where preparation_id='prepare-replay'",fetch=True)==[("ver",)]

def test_two_concurrent_exact_preparations_return_one_physical_row():
    _seed("prepare-race")
    exact=_candidate("prepare-race",authority="prepare-race",prop="prop",fin="fin")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _: _prepare(exact),range(2)))
    assert results==["prepare-race","prepare-race"]
    assert _super("select count(*) from freshness_authority.prepared_verifications where preparation_id='prepare-race'",fetch=True)==[(1,)]

def test_canonical_codec_duplicate_unicode_float_bool_and_safe_integer():
    with pytest.raises(ValueError): parse_canonical_json('{"x":1,"x":2}')
    with pytest.raises((ValueError,UnicodeError)): canonical_json_bytes({"x":"\ud800"})
    for v in (1.0,2**53):
        with pytest.raises(ValueError): canonical_json_bytes({"x":v})
    assert canonical_json_bytes({"x":True})!=canonical_json_bytes({"x":1})

def test_rfc8785_utf16_property_order_and_postgresql_byte_equality():
    vector={"\u20ac":"Euro Sign","\r":"Carriage Return","\ufb33":"Hebrew Letter Dalet With Dagesh","1":"One","😀":"Emoji: Grinning Face","\u0080":"Control","ö":"Latin Small Letter O With Diaeresis"}
    expected='{"\\r":"Carriage Return","1":"One","\u0080":"Control","ö":"Latin Small Letter O With Diaeresis","€":"Euro Sign","😀":"Emoji: Grinning Face","דּ":"Hebrew Letter Dalet With Dagesh"}'.encode()
    assert canonical_json_bytes(vector)==expected
    assert canonical_json_bytes({"😀":1,"\ue000":2})==b'{"\xf0\x9f\x98\x80":1,"\xee\x80\x80":2}'
    database=_super("SELECT freshness_authority.canonical_jsonb(%s)",(json.dumps(vector,ensure_ascii=False),),True)[0][0]
    assert bytes(database)==expected

def test_set_like_complete_heads_have_deterministic_normalization_and_sql_rejects_unsorted():
    reverse=list(reversed(S1))
    assert normalize_complete_semantic_head_set(S1)==normalize_complete_semantic_head_set(reverse)
    assert complete_semantic_head_digest(S1)==complete_semantic_head_digest(reverse)==H1
    p,d,r=_candidate("unsorted")
    d["payload"]["complete_semantic_head_set"]=S1
    d["document_digest"]=hashlib.sha256(DOCUMENT_DOMAIN+canonical_json_bytes(d["payload"])).hexdigest()
    p["proposed_document_digest"]=d["document_digest"]
    r["accepted_document_digest"]=d["document_digest"]
    r["receipt_canonical_digest"]=hashlib.sha256(canonical_json_bytes(r)).hexdigest()
    p["receipt_canonical_digest"]=r["receipt_canonical_digest"]
    with pytest.raises(psycopg.Error): _prepare((p,d,r))
    with pytest.raises(ValueError): normalize_complete_semantic_head_set([S0[0],S0[0]])

@pytest.mark.parametrize("mutation",["missing","extra","bool","scope","role","launder"])
def test_preparation_closed_schema_identity_and_scope_negatives(mutation):
    p,d,r=_candidate("closed"+mutation)
    if mutation=="missing": p.pop("verifier_authority_version")
    elif mutation=="extra": p["verified"]=True
    elif mutation=="bool": p["proposer_key_version"]=True
    elif mutation=="scope": p["environment"]="TEST"
    elif mutation=="role": p["proposer_credential_role_identity"]="fin"
    else: p["proposer_identity"]="identity-B"
    with pytest.raises(psycopg.Error): _prepare((p,d,r))

@pytest.mark.parametrize("field",["environment","trust_domain","authority_id","generation","predecessor_generation","predecessor_document_digest","freshness_authority_key_id","freshness_authority_key_version","finalization_request_id"])
def test_document_payload_required_null_is_rejected_after_exact_digest_rebinding(field):
    values=deepcopy(_candidate("null-document-"+field)); values[1]["payload"][field]=None; values=_rebind(values)
    with pytest.raises(psycopg.Error): _prepare(values)
    assert _super("SELECT count(*) FROM freshness_authority.prepared_verifications WHERE preparation_id=%s",(values[0]["preparation_id"],),True)==[(0,)]

@pytest.mark.parametrize("field",["schema_version","environment","trust_domain","authority_id","exact_predecessor_generation","exact_predecessor_document_digest","accepted_generation","accepted_document_digest","complete_semantic_head_digest","finalization_request_id","receipt_id","freshness_authority_key_id","freshness_authority_key_version","authentication_tag_or_signature"])
def test_receipt_v1_required_null_is_rejected_after_canonical_digest_rebinding(field):
    values=deepcopy(_candidate("null-receipt-"+field)); values[2][field]=None
    values[0]["receipt_canonical_digest"]=hashlib.sha256(canonical_json_bytes(values[2])).hexdigest()
    with pytest.raises(psycopg.Error): _prepare(values)
    assert _super("SELECT count(*) FROM freshness_authority.prepared_verifications WHERE preparation_id=%s",(values[0]["preparation_id"],),True)==[(0,)]

@pytest.mark.parametrize("field",["schema_version","proposer_key_version","finalization_key_version","proposer_lifecycle_generation_observed_for_key_binding_only","finalization_lifecycle_generation_observed_for_key_binding_only","expected_predecessor_generation","verifier_authority_version"])
def test_preparation_numeric_strings_are_not_integers(field):
    values=deepcopy(_candidate("prep-string-"+field)); values[0][field]=str(values[0][field])
    with pytest.raises(psycopg.Error): _prepare(values)

@pytest.mark.parametrize("field",["schema_version","generation","predecessor_generation","freshness_authority_key_version"])
def test_document_numeric_strings_are_rejected_with_recomputed_digests(field):
    values=deepcopy(_candidate("doc-string-"+field)); values[1]["payload"][field]=str(values[1]["payload"][field]); values=_rebind(values)
    with pytest.raises(psycopg.Error): _prepare(values)

@pytest.mark.parametrize("field",["schema_version","exact_predecessor_generation","accepted_generation","freshness_authority_key_version"])
def test_receipt_numeric_strings_are_rejected_with_recomputed_receipt_digest(field):
    values=deepcopy(_candidate("receipt-string-"+field)); values[2][field]=str(values[2][field]); values[0]["receipt_canonical_digest"]=hashlib.sha256(canonical_json_bytes(values[2])).hexdigest()
    with pytest.raises(psycopg.Error): _prepare(values)

def test_wrong_type_receipt_cannot_be_prepared_committed_or_recovered():
    values=deepcopy(_candidate("wrong-type-receipt")); values[2]["schema_version"]="1"
    values[0]["receipt_canonical_digest"]=hashlib.sha256(canonical_json_bytes(values[2])).hexdigest()
    with pytest.raises(psycopg.Error): _prepare(values)
    with pytest.raises(psycopg.Error): _cas(values)
    assert _super("SELECT count(*) FROM freshness_authority.decisions WHERE original_decision_identity=%s",(values[0]["original_decision_identity"],),True)==[(0,)]

@pytest.mark.parametrize("object_name,field",[("preparation","proposed_document_digest"),("preparation","receipt_canonical_digest"),("document","document_digest"),("receipt","accepted_document_digest"),("receipt","complete_semantic_head_digest")])
def test_digest_fields_require_exact_lowercase_64_hex(object_name,field):
    values=deepcopy(_candidate("lexical-"+field)); target=values[0] if object_name=="preparation" else values[1] if object_name=="document" else values[2]
    target[field]="A"*64
    if object_name=="receipt": values[0]["receipt_canonical_digest"]=hashlib.sha256(canonical_json_bytes(values[2])).hexdigest()
    with pytest.raises(psycopg.Error): _prepare(values)

@pytest.mark.parametrize("object_name,field,value",[("preparation","proposer_key_version",True),("document","generation",False),("receipt","accepted_generation",True)])
def test_bool_is_never_an_integer_across_all_cryptographic_objects(object_name,field,value):
    values=deepcopy(_candidate("bool-"+object_name))
    target=values[0] if object_name=="preparation" else values[1]["payload"] if object_name=="document" else values[2]
    target[field]=value
    if object_name=="document": values=_rebind(values)
    elif object_name=="receipt": values[0]["receipt_canonical_digest"]=hashlib.sha256(canonical_json_bytes(values[2])).hexdigest()
    with pytest.raises(psycopg.Error): _prepare(values)

@pytest.mark.parametrize("position",[0,1,2])
def test_float_numeric_representation_is_rejected_at_canonical_byte_boundary(position):
    values=_candidate("float-"+str(position)); raw=[canonical_json_bytes(x) for x in values]
    needle=[b'"proposer_key_version":1',b'"generation":1',b'"accepted_generation":1'][position]
    raw[position]=raw[position].replace(needle,needle[:-1]+b'1.0')
    with pytest.raises(psycopg.Error): _prepare_raw(*raw)

@pytest.mark.parametrize("document_key,document_version,receipt_key,receipt_version",[("bogus",1,"fk",1),("fk",2,"fk",1),("bogus",2,"bogus",2)])
def test_document_receipt_retained_finalization_selector_must_be_exact(document_key,document_version,receipt_key,receipt_version):
    values=deepcopy(_candidate("key-binding-"+document_key+str(document_version)+receipt_key+str(receipt_version)))
    values[1]["payload"]["freshness_authority_key_id"]=document_key; values[1]["payload"]["freshness_authority_key_version"]=document_version
    values[2]["freshness_authority_key_id"]=receipt_key; values[2]["freshness_authority_key_version"]=receipt_version
    values=_rebind(values)
    with pytest.raises(psycopg.Error): _prepare(values)

def test_preparation_and_cas_happy_path_exact_replay_and_lost_response_recovery():
    values=_candidate(); assert _prepare(values)=="prep"; accepted=_cas(values); assert accepted[0]=="CAS_ACCEPTED"; replay=_cas(values); assert replay[0]=="ALREADY_ACCEPTED_EXACT" and replay[1:]==accepted[1:]

def test_complete_head_h0_to_h1_then_n2_and_stale_h0_rejection():
    assert H0!=H1
    first_doc=_super("SELECT complete_head_digest,document_digest FROM freshness_authority.authoritative_documents WHERE authority_id='auth' AND generation=1",fetch=True)[0]
    lineage=_super("SELECT current_generation,current_complete_head_digest FROM freshness_authority.authority_lineages WHERE authority_id='auth'",fetch=True)[0]
    assert first_doc[0]==H1 and lineage==(1,H1)
    s2=[{"domain":"catalog","digest":"3"*64}]; h2=complete_semantic_head_digest(s2)
    second=_candidate("n2",generation=2,predecessor_digest=first_doc[1],predecessor_head=H1,head_set=s2)
    _prepare(second); assert _cas(second)[0]=="CAS_ACCEPTED"
    assert _super("SELECT current_generation,current_complete_head_digest FROM freshness_authority.authority_lineages WHERE authority_id='auth'",fetch=True)==[(2,h2)]
    stale=_candidate("staleh0",generation=3,predecessor_digest=second[0]["proposed_document_digest"],predecessor_head=H0,head_set=[])
    _prepare(stale)
    with pytest.raises(psycopg.Error): _cas(stale)

def test_retained_generation_heads_resolve_historical_not_current_projection():
    _seed("head-history")
    first=_candidate("head-history-1",authority="head-history",prop="prop",fin="fin")
    _prepare(first); assert _cas(first)[0]=="CAS_ACCEPTED"
    first_digest=first[0]["proposed_document_digest"]
    s2=[{"domain":"catalog","digest":"4"*64}]; h2=complete_semantic_head_digest(s2)
    second=_candidate("head-history-2",authority="head-history",prop="prop",fin="fin",
                      generation=2,predecessor_digest=first_digest,predecessor_head=H1,head_set=s2)
    _prepare(second); assert _cas(second)[0]=="CAS_ACCEPTED"
    previous_digest=second[0]["proposed_document_digest"]; previous_head=h2
    for generation in range(3,6):
        heads=[{"domain":"catalog","digest":str(generation)*64}]
        candidate=_candidate(f"head-history-{generation}",authority="head-history",prop="prop",fin="fin",
                             generation=generation,predecessor_digest=previous_digest,
                             predecessor_head=previous_head,head_set=heads)
        _prepare(candidate); assert _cas(candidate)[0]=="CAS_ACCEPTED"
        previous_digest=candidate[0]["proposed_document_digest"]
        previous_head=complete_semantic_head_digest(heads)
    with _role("freshness_crypto_verifier") as c:
        resolve=lambda generation,digest: c.execute(
            "select freshness_authority.resolve_predecessor_head(%s,%s,%s,%s,%s)",
            ("PRODUCTION","td","head-history",generation,digest)).fetchone()[0]
        assert resolve(0,"a"*64)==H0
        assert resolve(1,first_digest)==H1
        assert resolve(2,second[0]["proposed_document_digest"])==h2
        assert resolve(0,"a"*64)==H0  # still historical with current at N+5
        for generation,digest in ((0,"f"*64),(9,"a"*64)):
            with pytest.raises(psycopg.Error): resolve(generation,digest)
            c.rollback()
        with pytest.raises(psycopg.Error): c.execute(
            "select freshness_authority.resolve_predecessor_head('TEST','td','head-history',0,%s)",
            ("a"*64,))
        c.rollback()
        with pytest.raises(psycopg.Error): c.execute(
            "select freshness_authority.resolve_predecessor_head('PRODUCTION','wrong','head-history',0,%s)",
            ("a"*64,))

def test_current_projection_tamper_does_not_rewrite_history_and_fails_qualification():
    _seed("projection-tamper")
    retained=_super("select complete_head_digest from freshness_authority.authority_generation_heads where authority_id='projection-tamper' and generation=0",fetch=True)
    assert retained==[(H0,)]
    _super("update freshness_authority.authority_lineages set current_complete_head_digest=%s where authority_id='projection-tamper'",("f"*64,))
    with _role("freshness_crypto_verifier") as c:
        assert c.execute(
            "select freshness_authority.resolve_predecessor_head('PRODUCTION','td','projection-tamper',0,%s)",
            ("a"*64,)).fetchone()==(H0,)
    with pytest.raises(FreshnessAuthorityQualificationError,match="retained evidence integrity"):
        qualify_postgresql_freshness_authority(CFG)
    _super("update freshness_authority.authority_lineages set current_complete_head_digest=%s where authority_id='projection-tamper'",(H0,))

def test_arbitrary_receipt_complete_head_digest_is_rejected():
    values=_candidate("badhead"); p,d,r=deepcopy(values); r["complete_semantic_head_digest"]="f"*64
    p["receipt_canonical_digest"]=hashlib.sha256(canonical_json_bytes(r)).hexdigest()
    with pytest.raises(psycopg.Error): _prepare((p,d,r))

def test_exact_replay_requires_retained_document_and_receipt_bytes():
    p,d,r=_candidate("replaybytes",authority="isolation",prop="propi",fin="fini")
    # This authority is provisioned later; the substantive replay checks use auth's retained decision.
    original=_candidate(); op,od,orr=deepcopy(original)
    od["authentication_tag_or_signature"]=base64.urlsafe_b64encode(b"y"*64).rstrip(b"=").decode()
    with pytest.raises(psycopg.Error): _cas((op,od,orr))
    op,od,orr=deepcopy(original); orr["authentication_tag_or_signature"]=base64.urlsafe_b64encode(b"y"*64).rstrip(b"=").decode()
    with pytest.raises(psycopg.Error): _cas((op,od,orr))
    with _role("freshness_runtime") as c, pytest.raises(psycopg.Error):
        c.execute("set transaction isolation level serializable")
        c.execute("select * from freshness_authority.compare_and_advance(%s,%s,%s,%s)",("prep",json.dumps(original[0]),json.dumps(original[1]).encode(),canonical_json_bytes(original[2])))
    assert _super("SELECT count(*) FROM freshness_authority.decisions WHERE authority_id='auth' AND original_decision_identity='odiprep'",fetch=True)==[(1,)]

@pytest.mark.parametrize("signature",[
    base64.urlsafe_b64encode(b"x"*64).decode(),
    "+"+base64.urlsafe_b64encode(b"x"*64).rstrip(b"=").decode()[1:],
    "A", base64.urlsafe_b64encode(b"x"*63).rstrip(b"=").decode(),
    base64.urlsafe_b64encode(b"x"*65).rstrip(b"=").decode(),
    base64.urlsafe_b64encode(b"x"*64).rstrip(b"=").decode()[:-1]+"B",
])
def test_postgresql_rejects_noncanonical_base64url_signatures(signature):
    p,d,r=_candidate("b64"+str(len(signature))); r["authentication_tag_or_signature"]=signature
    p["receipt_canonical_digest"]=hashlib.sha256(canonical_json_bytes(r)).hexdigest()
    with pytest.raises(psycopg.Error): _prepare((p,d,r))

@pytest.mark.parametrize("target,field,value",[("document","payload",None),("document","document_digest","f"*64),("receipt","receipt_id","evil"),("receipt","finalization_request_id","evil"),("receipt","accepted_document_digest","f"*64),("receipt","freshness_authority_key_id","evil"),("receipt","freshness_authority_key_version",2),("receipt","authentication_tag_or_signature",base64.urlsafe_b64encode(b"z"*64).rstrip(b"=").decode()),("receipt","missing",None),("receipt","extra",1)])
def test_document_and_receipt_substitution_is_rejected(target,field,value):
    pid="sub"+field+str(value)[:3]; values=_candidate(pid); _prepare(values); p,d,r=deepcopy(values)
    if target=="document" and field=="payload": d["payload"]["authority_id"]="evil"
    elif target=="receipt" and field=="missing": r.pop("receipt_id")
    elif target=="receipt" and field=="extra": r["extra"]=value
    else: (d if target=="document" else r)[field]=value
    with pytest.raises(psycopg.Error): _cas((p,d,r))

def test_non_serializable_rejected_and_serializable_allowed_subject_to_preconditions():
    _seed("isolation","i"); v=_candidate("iso",authority="isolation",prop="propi",fin="fini"); _prepare(v)
    for level in ("read committed","repeatable read"):
        with pytest.raises(psycopg.Error): _cas(v,level)
    assert _cas(v)[0]=="CAS_ACCEPTED"

def test_cross_role_same_raw_key_sequential_and_concurrent_constraint():
    key=(bytes([9])*32).hex(); a=_credential("p9","p9",PROPOSER_ROLE,"p9",9); b=_credential("f9","f9",FINALIZATION_ROLE,"f9",9)
    _admin("select freshness_authority.provision_authority(%s,%s,%s,%s,%s)",("PRODUCTION","td","race","a"*64,H0)); a["authority_id"]=b["authority_id"]="race"
    def put(x):
        try: _admin("select freshness_authority.provision_credential(%s)",(json.dumps(x),)); return True
        except psycopg.Error: return False
    with ThreadPoolExecutor(2) as ex: result=list(ex.map(put,(a,b)))
    assert sum(result)==1
    assert _super("SELECT count(DISTINCT semantic_role) FROM freshness_authority.credentials WHERE authority_id='race'",fetch=True)==[(1,)]

@pytest.mark.parametrize("role,prefix",[(FINALIZATION_ROLE,"fin"),(PROPOSER_ROLE,"prop")])
def test_historical_authenticated_selector_cannot_be_reused_for_different_material(role,prefix):
    scope="selector"+prefix; _admin("select freshness_authority.provision_authority(%s,%s,%s,%s,%s)",("PRODUCTION","td",scope,"a"*64,H0))
    first=_credential(prefix+"A",prefix+"-A",role,"shared",11); first["authority_id"]=scope
    second=_credential(prefix+"B",prefix+"-B",role,"shared",12); second["authority_id"]=scope
    _admin("select freshness_authority.provision_credential(%s)",(json.dumps(first),))
    _admin("select freshness_authority.transition_credential('PRODUCTION','td',%s,%s,1,'VERIFY_ONLY','{}')",(scope,first["credential_id"]))
    with pytest.raises(psycopg.Error): _admin("select freshness_authority.provision_credential(%s)",(json.dumps(second),))
    assert _super("SELECT count(*) FROM freshness_authority.credentials WHERE authority_id=%s AND semantic_role=%s AND key_id='shared' AND key_version=1",(scope,role),True)==[(1,)]

@pytest.mark.parametrize("role,prefix",[(FINALIZATION_ROLE,"cf"),(PROPOSER_ROLE,"cp")])
def test_concurrent_duplicate_selector_is_resolved_by_storage_uniqueness(role,prefix):
    scope="concurrent"+prefix; _admin("select freshness_authority.provision_authority(%s,%s,%s,%s,%s)",("PRODUCTION","td",scope,"a"*64,H0))
    candidates=[]
    for suffix,octet in (("A",13),("B",14)):
        item=_credential(prefix+suffix,prefix+suffix,role,"selector",octet); item["authority_id"]=scope; candidates.append(item)
    def put(item):
        try: _admin("select freshness_authority.provision_credential(%s)",(json.dumps(item),)); return True
        except psycopg.Error: return False
    with ThreadPoolExecutor(2) as ex: outcomes=list(ex.map(put,candidates))
    assert sum(outcomes)==1
    assert _super("SELECT count(*) FROM freshness_authority.credentials WHERE authority_id=%s AND semantic_role=%s",(scope,role),True)==[(1,)]

def test_one_active_per_role_rotation_retains_history_and_old_cannot_authorize_cas():
    _seed("rotationcard","q")
    extra=_credential("fin-new","identity-new",FINALIZATION_ROLE,"fk2",4); extra["authority_id"]="rotationcard"
    with pytest.raises(psycopg.Error): _admin("select freshness_authority.provision_credential(%s)",(json.dumps(extra),))
    _admin("select freshness_authority.transition_credential('PRODUCTION','td','rotationcard','finq',1,'VERIFY_ONLY','{}')")
    _admin("select freshness_authority.provision_credential(%s)",(json.dumps(extra),))
    states=_super("SELECT credential_id,lifecycle_state FROM freshness_authority.credentials WHERE authority_id='rotationcard' AND semantic_role=%s ORDER BY credential_id",(FINALIZATION_ROLE,),True)
    assert states==[("fin-new","ACTIVE"),("finq","VERIFY_ONLY")]
    accepted=_candidate("newfin",authority="rotationcard",prop="propq",fin="fin-new",finalization_key_id="fk2",finalization_octet=4)
    _prepare(accepted); assert _cas(accepted)[0]=="CAS_ACCEPTED"
    old=_candidate("oldfin",authority="rotationcard",prop="propq",fin="finq",request="old",generation=2,predecessor_digest=accepted[0]["proposed_document_digest"],predecessor_head=H1,finalization_key_id="fk",finalization_octet=2)
    _prepare(old)
    with pytest.raises(psycopg.Error): _cas(old)
    _admin("select freshness_authority.transition_credential('PRODUCTION','td','rotationcard','finq',2,'REVOKED','{}')")
    with pytest.raises(psycopg.Error): _admin("select freshness_authority.transition_credential('PRODUCTION','td','rotationcard','finq',3,'VERIFY_ONLY','{}')")

def test_proposer_active_to_verify_only_then_new_active_retains_old_lineage():
    scope="proposerrotation"; _admin("select freshness_authority.provision_authority(%s,%s,%s,%s,%s)",("PRODUCTION","td",scope,"a"*64,H0))
    old=_credential("old-proposer","old",PROPOSER_ROLE,"pk-old",21); old["authority_id"]=scope
    new=_credential("new-proposer","new",PROPOSER_ROLE,"pk-new",22); new["authority_id"]=scope
    _admin("select freshness_authority.provision_credential(%s)",(json.dumps(old),))
    with pytest.raises(psycopg.Error): _admin("select freshness_authority.provision_credential(%s)",(json.dumps(new),))
    _admin("select freshness_authority.transition_credential('PRODUCTION','td',%s,'old-proposer',1,'VERIFY_ONLY','{}')",(scope,))
    _admin("select freshness_authority.provision_credential(%s)",(json.dumps(new),))
    assert _super("SELECT credential_id,lifecycle_state FROM freshness_authority.credentials WHERE authority_id=%s ORDER BY credential_id",(scope,),True)==[("new-proposer","ACTIVE"),("old-proposer","VERIFY_ONLY")]

def test_concurrent_same_role_provision_and_transition_never_leave_two_active():
    _seed("transitionrace","t")
    new=_credential("prop-new","identity-new",PROPOSER_ROLE,"pk2",5); new["authority_id"]="transitionrace"
    def transition():
        try: _admin("select freshness_authority.transition_credential('PRODUCTION','td','transitionrace','propt',1,'VERIFY_ONLY','{}')"); return True
        except psycopg.Error: return False
    def provision():
        try: _admin("select freshness_authority.provision_credential(%s)",(json.dumps(new),)); return True
        except psycopg.Error: return False
    with ThreadPoolExecutor(2) as ex: outcomes=[ex.submit(transition),ex.submit(provision)]; [x.result() for x in outcomes]
    assert _super("SELECT count(*) FROM freshness_authority.credentials WHERE authority_id='transitionrace' AND semantic_role=%s AND lifecycle_state='ACTIVE'",(PROPOSER_ROLE,),True)[0][0] in (0,1)

def test_latest_retained_lifecycle_head_is_authority_and_revoked_terminal():
    _seed("diverge","d"); v=_candidate("div",authority="diverge",prop="propd",fin="find"); _prepare(v)
    _super("INSERT INTO freshness_authority.key_lifecycle_history VALUES('PRODUCTION','td','diverge','propd',2,'REVOKED','{}')")
    with pytest.raises(psycopg.Error): _cas(v)
    with pytest.raises(psycopg.Error): _admin("select freshness_authority.transition_credential('PRODUCTION','td','diverge','propd',1,'VERIFY_ONLY','{}')")

def test_concurrent_different_successors_exact_replay_and_same_preparation_consumption():
    _seed("concurrent","c")
    left=_candidate("left",authority="concurrent",prop="propc",fin="finc",request="left")
    right=_candidate("right",authority="concurrent",prop="propc",fin="finc",request="right")
    _prepare(left); _prepare(right)
    def attempt(v):
        try: return _cas(v)[0]
        except psycopg.Error: return "REJECTED"
    with ThreadPoolExecutor(2) as ex: results=list(ex.map(attempt,(left,right)))
    assert results.count("CAS_ACCEPTED")==1 and results.count("REJECTED")==1
    assert _super("SELECT count(*) FROM freshness_authority.decisions WHERE authority_id='concurrent'",fetch=True)==[(1,)]
    winner=left if results[0]=="CAS_ACCEPTED" else right
    with ThreadPoolExecutor(2) as ex: replay=list(ex.map(attempt,(winner,winner)))
    assert replay==["ALREADY_ACCEPTED_EXACT","ALREADY_ACCEPTED_EXACT"]

def test_concurrent_same_preparation_has_one_physical_decision_and_recoverable_replay():
    _seed("consume","m"); v=_candidate("consume",authority="consume",prop="propm",fin="finm"); _prepare(v)
    def attempt():
        try: return _cas(v)[0]
        except psycopg.Error: return "RETRY"
    with ThreadPoolExecutor(2) as ex: outcomes=list(ex.map(lambda _:attempt(),range(2)))
    assert outcomes.count("CAS_ACCEPTED")==1
    assert _super("SELECT count(*) FROM freshness_authority.decisions WHERE authority_id='consume'",fetch=True)==[(1,)]
    assert _cas(v)[0]=="ALREADY_ACCEPTED_EXACT"

def test_revoke_before_cas_and_cas_before_revoke_serialization():
    _seed("revoke","r"); v=_candidate("rev",authority="revoke",prop="propr",fin="finr"); _prepare(v)
    _admin("select freshness_authority.transition_credential('PRODUCTION','td','revoke','propr',1,'REVOKED','{}')")
    with pytest.raises(psycopg.Error): _cas(v)
    _seed("casfirst","z"); winner=_candidate("first",authority="casfirst",prop="propz",fin="finz"); _prepare(winner)
    assert _cas(winner)[0]=="CAS_ACCEPTED"
    _admin("select freshness_authority.transition_credential('PRODUCTION','td','casfirst','propz',1,'REVOKED','{}')")

@pytest.mark.parametrize("credential,prefix",[("prop","rotation"),("fin","signer")])
def test_proposer_rotation_and_finalization_signer_transition_race(credential,prefix):
    suffix=prefix[0]; authority_id=prefix+"race"; _seed(authority_id,suffix)
    v=_candidate(prefix,authority=authority_id,prop="prop"+suffix,fin="fin"+suffix); _prepare(v)
    target=credential+suffix
    def transition():
        try: _admin("select freshness_authority.transition_credential('PRODUCTION','td',%s,%s,1,'VERIFY_ONLY','{}')",(authority_id,target)); return "TRANSITIONED"
        except psycopg.Error: return "RETRY"
    def cas():
        try: return _cas(v)[0]
        except psycopg.Error: return "REJECTED"
    with ThreadPoolExecutor(2) as ex:
        results=[ex.submit(transition),ex.submit(cas)]; outcomes=[x.result() for x in results]
    assert outcomes[0]=="TRANSITIONED" and outcomes[1] in {"CAS_ACCEPTED","REJECTED"}
    assert _super("SELECT count(*) FROM freshness_authority.decisions WHERE authority_id=%s",(authority_id,),True)[0][0] in (0,1)

def test_postgresql_restart_after_preparation_and_after_commit_exact_recovery():
    _seed("restart","s"); v=_candidate("restart",authority="restart",prop="props",fin="fins"); _prepare(v)
    subprocess.run(["pg_ctlcluster","16","main","restart"],check=True)
    accepted=_cas(v); assert accepted[0]=="CAS_ACCEPTED"
    subprocess.run(["pg_ctlcluster","16","main","restart"],check=True)
    recovered=_cas(v); assert recovered[0]=="ALREADY_ACCEPTED_EXACT" and recovered[1:]==accepted[1:]

def test_acl_function_source_manifest_role_owner_schema_and_relation_tamper_fail_closed():
    mutations=[
      ("GRANT EXECUTE ON FUNCTION freshness_authority.prepare_verified_freshness_candidate(bytea,bytea,bytea) TO freshness_runtime","REVOKE EXECUTE ON FUNCTION freshness_authority.prepare_verified_freshness_candidate(bytea,bytea,bytea) FROM freshness_runtime"),
      ("GRANT INSERT ON freshness_authority.credentials TO freshness_runtime","REVOKE INSERT ON freshness_authority.credentials FROM freshness_runtime"),
      ("CREATE TABLE freshness_authority.evil(x int)","DROP TABLE freshness_authority.evil"),
      ("ALTER ROLE freshness_function_owner LOGIN","ALTER ROLE freshness_function_owner NOLOGIN"),
      ("ALTER TABLE freshness_authority.credentials ENABLE ROW LEVEL SECURITY","ALTER TABLE freshness_authority.credentials DISABLE ROW LEVEL SECURITY"),
      ("ALTER TABLE freshness_authority.metadata ADD COLUMN evil text","ALTER TABLE freshness_authority.metadata DROP COLUMN evil"),
      ("CREATE INDEX evil_index ON freshness_authority.credentials(credential_id)","DROP INDEX freshness_authority.evil_index"),
      ("CREATE TRIGGER evil_trigger BEFORE UPDATE ON freshness_authority.metadata FOR EACH STATEMENT EXECUTE FUNCTION suppress_redundant_updates_trigger()","DROP TRIGGER evil_trigger ON freshness_authority.metadata"),
      ("CREATE RULE evil_rule AS ON DELETE TO freshness_authority.metadata DO INSTEAD NOTHING","DROP RULE evil_rule ON freshness_authority.metadata"),
      ("CREATE POLICY evil_policy ON freshness_authority.metadata USING (true)","DROP POLICY evil_policy ON freshness_authority.metadata"),
      ("GRANT freshness_runtime TO freshness_reader","REVOKE freshness_runtime FROM freshness_reader"),
    ]
    for bad,undo in mutations:
        _super(bad)
        with pytest.raises(FreshnessAuthorityQualificationError): qualify_postgresql_freshness_authority(CFG)
        _super(undo)
        qualify_postgresql_freshness_authority(CFG)

def test_selector_constraint_active_predicate_and_unexpected_object_inventory_tamper():
    _super("ALTER TABLE freshness_authority.credentials DROP CONSTRAINT credentials_authenticated_selector_key")
    with pytest.raises(FreshnessAuthorityQualificationError): qualify_postgresql_freshness_authority(CFG)
    _super("ALTER TABLE freshness_authority.credentials ADD CONSTRAINT credentials_authenticated_selector_key UNIQUE(environment,trust_domain,authority_id,semantic_role,key_id,key_version)")
    qualify_postgresql_freshness_authority(CFG)
    _super("DROP INDEX freshness_authority.credentials_one_active_role_idx")
    with pytest.raises(FreshnessAuthorityQualificationError): qualify_postgresql_freshness_authority(CFG)
    _super("CREATE UNIQUE INDEX credentials_one_active_role_idx ON freshness_authority.credentials(environment,trust_domain,authority_id,semantic_role) WHERE lifecycle_state='ACTIVE'")
    qualify_postgresql_freshness_authority(CFG)
    _super("CREATE VIEW freshness_authority.evil_view AS SELECT * FROM freshness_authority.credentials")
    _super("GRANT SELECT ON freshness_authority.evil_view TO freshness_runtime")
    with pytest.raises(FreshnessAuthorityQualificationError): qualify_postgresql_freshness_authority(CFG)
    _super("DROP VIEW freshness_authority.evil_view"); qualify_postgresql_freshness_authority(CFG)
    _super("CREATE MATERIALIZED VIEW freshness_authority.evil_materialized AS SELECT count(*) FROM freshness_authority.credentials")
    with pytest.raises(FreshnessAuthorityQualificationError): qualify_postgresql_freshness_authority(CFG)
    _super("DROP MATERIALIZED VIEW freshness_authority.evil_materialized"); qualify_postgresql_freshness_authority(CFG)

def test_unexpected_execute_grantees_public_and_grant_option_fail_closed():
    _super("CREATE ROLE freshness_outsider NOLOGIN")
    cases=[
      ("GRANT EXECUTE ON FUNCTION freshness_authority.compare_and_advance(text,jsonb,bytea,bytea) TO freshness_crypto_verifier","REVOKE EXECUTE ON FUNCTION freshness_authority.compare_and_advance(text,jsonb,bytea,bytea) FROM freshness_crypto_verifier"),
      ("GRANT EXECUTE ON FUNCTION freshness_authority.compare_and_advance(text,jsonb,bytea,bytea) TO freshness_outsider","REVOKE EXECUTE ON FUNCTION freshness_authority.compare_and_advance(text,jsonb,bytea,bytea) FROM freshness_outsider"),
      ("GRANT EXECUTE ON FUNCTION freshness_authority.compare_and_advance(text,jsonb,bytea,bytea) TO PUBLIC","REVOKE EXECUTE ON FUNCTION freshness_authority.compare_and_advance(text,jsonb,bytea,bytea) FROM PUBLIC"),
      ("GRANT EXECUTE ON FUNCTION freshness_authority.compare_and_advance(text,jsonb,bytea,bytea) TO freshness_runtime WITH GRANT OPTION","REVOKE GRANT OPTION FOR EXECUTE ON FUNCTION freshness_authority.compare_and_advance(text,jsonb,bytea,bytea) FROM freshness_runtime"),
    ]
    for bad,undo in cases:
        _super(bad)
        with pytest.raises(FreshnessAuthorityQualificationError): qualify_postgresql_freshness_authority(CFG)
        _super(undo); qualify_postgresql_freshness_authority(CFG)
    _super("DROP ROLE freshness_outsider")

def test_pg_proc_source_and_matching_mutable_manifest_cannot_bypass_reviewed_source():
    source=_super("SELECT prosrc FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace WHERE n.nspname='freshness_authority' AND p.proname='compare_and_advance'",fetch=True)[0][0]
    _super("UPDATE pg_proc SET prosrc=prosrc||E'\\n-- evil' WHERE oid='freshness_authority.compare_and_advance(text,jsonb,bytea,bytea)'::regprocedure")
    _super("UPDATE freshness_authority.metadata SET function_manifest=jsonb_set(function_manifest,'{compare_and_advance(text,jsonb,bytea,bytea)}',to_jsonb(encode(sha256(convert_to(%s||E'\\n-- evil','UTF8')),'hex')))",(source,))
    with pytest.raises(FreshnessAuthorityQualificationError): qualify_postgresql_freshness_authority(CFG)
    _super("UPDATE pg_proc SET prosrc=%s WHERE oid='freshness_authority.compare_and_advance(text,jsonb,bytea,bytea)'::regprocedure",(source,))
    # Restore the immutable code-side manifest through a fresh digest.
    digest=hashlib.sha256(source.encode()).hexdigest(); _super("UPDATE freshness_authority.metadata SET function_manifest=jsonb_set(function_manifest,'{compare_and_advance(text,jsonb,bytea,bytea)}',to_jsonb(%s::text))",(digest,)); qualify_postgresql_freshness_authority(CFG)

def test_retained_receipt_document_and_prepared_evidence_tamper_fail_qualification():
    rows=_super("SELECT receipt_id,canonical_receipt FROM freshness_authority.finalization_receipts LIMIT 1",fetch=True)
    assert rows
    receipt_id,receipt=rows[0]
    _super("UPDATE freshness_authority.finalization_receipts SET canonical_receipt=canonical_receipt||'{\"evil\":true}'::jsonb WHERE receipt_id=%s",(receipt_id,))
    with pytest.raises(FreshnessAuthorityQualificationError): qualify_postgresql_freshness_authority(CFG)
    _super("UPDATE freshness_authority.finalization_receipts SET canonical_receipt=%s WHERE receipt_id=%s",(json.dumps(receipt),receipt_id))
    qualify_postgresql_freshness_authority(CFG)

def test_credential_raw_key_material_identity_relationship_tamper_fails_qualification():
    original=_super("SELECT public_key FROM freshness_authority.credentials WHERE authority_id='auth' AND credential_id='prop'",fetch=True)[0][0]
    _super("UPDATE freshness_authority.credentials SET public_key=%s WHERE authority_id='auth' AND credential_id='prop'",(bytes([7])*32,))
    with pytest.raises(FreshnessAuthorityQualificationError): qualify_postgresql_freshness_authority(CFG)
    _super("UPDATE freshness_authority.credentials SET public_key=%s WHERE authority_id='auth' AND credential_id='prop'",(original,))
    qualify_postgresql_freshness_authority(CFG)
