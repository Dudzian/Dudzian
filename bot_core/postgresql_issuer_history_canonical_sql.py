"""Reviewed PostgreSQL lexer for the frozen exact canonical JSON byte domain.

Decoded string keys remain bytea: PostgreSQL text cannot contain U+0000, while
the existing history contract deliberately permits it. UTF-8 byte ordering of
valid scalar strings is the same as Python's Unicode code-point ordering.
Integers are scanned without converting them to a bounded SQL numeric type.
"""

from __future__ import annotations

import re

CANONICAL_FUNCTION_RETURNS = {
    "canonical_string(bytea,integer)": "TABLE(next_position integer,decoded_key bytea)",
    "canonical_value(bytea,integer)": "integer",
    "validate_canonical_json(bytea)": "void",
    "canonical_object_member(bytea,text)": "bytea",
}
_SAFE_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]{0,62}\Z")


def canonical_function_sources(schema: str) -> dict[str, str]:
    """Return exact function bodies for source-attested offline installation."""
    if type(schema) is not str or _SAFE_IDENTIFIER.fullmatch(schema) is None:
        raise ValueError("schema must be a safe lowercase PostgreSQL identifier")
    namespace = '"' + schema + '"'
    return {
        "canonical_string(bytea,integer)": r"""
DECLARE
  pos integer := $2;
  size integer := pg_catalog.octet_length($1);
  ch integer;
  escaped integer;
  hex_control text;
  control integer;
  decoded bytea := ''::bytea;
BEGIN
  IF $1 IS NULL OR $2 IS NULL OR pos < 0 OR pos >= size
     OR pg_catalog.get_byte($1,pos) <> 34 THEN
    RAISE EXCEPTION 'canonical JSON string required' USING ERRCODE='23514';
  END IF;
  pos := pos + 1;
  LOOP
    IF pos >= size THEN
      RAISE EXCEPTION 'unterminated canonical JSON string' USING ERRCODE='23514';
    END IF;
    ch := pg_catalog.get_byte($1,pos);
    IF ch = 34 THEN
      next_position := pos + 1;
      decoded_key := decoded;
      RETURN NEXT;
      RETURN;
    ELSIF ch = 92 THEN
      IF pos + 1 >= size THEN
        RAISE EXCEPTION 'invalid canonical JSON escape' USING ERRCODE='23514';
      END IF;
      escaped := pg_catalog.get_byte($1,pos+1);
      IF escaped IN (34,92) THEN
        decoded := decoded || pg_catalog.set_byte(pg_catalog.decode('00','hex'),0,escaped);
        pos := pos + 2;
      ELSIF escaped IN (98,102,110,114,116) THEN
        control := CASE escaped WHEN 98 THEN 8 WHEN 102 THEN 12 WHEN 110 THEN 10
                   WHEN 114 THEN 13 WHEN 116 THEN 9 END;
        decoded := decoded || pg_catalog.set_byte(pg_catalog.decode('00','hex'),0,control);
        pos := pos + 2;
      ELSIF escaped = 117 AND pos + 5 < size THEN
        hex_control := pg_catalog.convert_from(pg_catalog.substring($1,pos+3,4),'UTF8');
        IF hex_control !~ '^00[01][0-9a-f]$' THEN
          RAISE EXCEPTION 'noncanonical JSON unicode escape' USING ERRCODE='23514';
        END IF;
        control := pg_catalog.get_byte(pg_catalog.decode(
          pg_catalog.substring(hex_control,3,2),'hex'),0);
        IF control IN (8,9,10,12,13) THEN
          RAISE EXCEPTION 'noncanonical JSON control escape' USING ERRCODE='23514';
        END IF;
        decoded := decoded || pg_catalog.set_byte(pg_catalog.decode('00','hex'),0,control);
        pos := pos + 6;
      ELSE
        RAISE EXCEPTION 'noncanonical JSON string escape' USING ERRCODE='23514';
      END IF;
    ELSIF ch < 32 THEN
      RAISE EXCEPTION 'unescaped JSON control character' USING ERRCODE='23514';
    ELSE
      decoded := decoded || pg_catalog.substring($1,pos+1,1);
      pos := pos + 1;
    END IF;
  END LOOP;
END
""",
        "canonical_value(bytea,integer)": r"""
DECLARE
  pos integer := $2;
  size integer := pg_catalog.octet_length($1);
  ch integer;
  item record;
  depth integer := 0;
  states integer[] := ARRAY[]::integer[];
  prior_keys bytea[] := ARRAY[]::bytea[];
  need_value boolean := true;
  negative boolean;
BEGIN
  IF $1 IS NULL OR $2 IS NULL OR pos < 0 OR pos >= size THEN
    RAISE EXCEPTION 'canonical JSON value required' USING ERRCODE='23514';
  END IF;
  -- Container states: 1/2 first/required object key, 3 object delimiter,
  -- 4/5 first/required array value, 6 array delimiter. No SQL recursion.
  LOOP
    IF pos >= size THEN
      RAISE EXCEPTION 'unterminated canonical JSON value' USING ERRCODE='23514';
    END IF;
    ch := pg_catalog.get_byte($1,pos);
    IF need_value THEN
      IF ch = 34 THEN
        SELECT * INTO STRICT item FROM @SCHEMA@.canonical_string($1,pos);
        pos := item.next_position;
      ELSIF ch IN (123,91) THEN
        depth := depth + 1;
        states[depth] := CASE ch WHEN 123 THEN 1 ELSE 4 END;
        prior_keys[depth] := NULL;
        pos := pos + 1;
        need_value := false;
        CONTINUE;
      ELSIF ch IN (110,116,102) THEN
        IF pg_catalog.substring($1,pos+1,4) IN
          (pg_catalog.decode('6e756c6c','hex'),pg_catalog.decode('74727565','hex')) THEN
          pos := pos + 4;
        ELSIF pg_catalog.substring($1,pos+1,5) = pg_catalog.decode('66616c7365','hex') THEN
          pos := pos + 5;
        ELSE
          RAISE EXCEPTION 'invalid canonical JSON literal' USING ERRCODE='23514';
        END IF;
      ELSE
        negative := false;
        IF ch = 45 THEN
          negative := true;
          pos := pos + 1;
          IF pos >= size THEN
            RAISE EXCEPTION 'canonical JSON integer required' USING ERRCODE='23514';
          END IF;
          ch := pg_catalog.get_byte($1,pos);
        END IF;
        IF ch = 48 THEN
          IF negative THEN
            RAISE EXCEPTION 'noncanonical negative zero' USING ERRCODE='23514';
          END IF;
          pos := pos + 1;
        ELSIF ch BETWEEN 49 AND 57 THEN
          pos := pos + 1;
          WHILE pos < size LOOP
            ch := pg_catalog.get_byte($1,pos);
            EXIT WHEN ch < 48 OR ch > 57;
            pos := pos + 1;
          END LOOP;
        ELSE
          RAISE EXCEPTION 'noncanonical JSON value or number' USING ERRCODE='23514';
        END IF;
      END IF;
      IF depth = 0 THEN RETURN pos; END IF;
      states[depth] := CASE WHEN states[depth] <= 3 THEN 3 ELSE 6 END;
      need_value := false;
      CONTINUE;
    END IF;
    IF states[depth] IN (1,2) THEN
      IF NOT (states[depth] = 1 AND ch = 125) THEN
        SELECT * INTO STRICT item FROM @SCHEMA@.canonical_string($1,pos);
        IF pg_catalog.octet_length(item.decoded_key) = 0
           OR (prior_keys[depth] IS NOT NULL AND item.decoded_key <= prior_keys[depth]) THEN
          RAISE EXCEPTION 'noncanonical JSON object key order or duplicate' USING ERRCODE='23514';
        END IF;
        prior_keys[depth] := item.decoded_key;
        pos := item.next_position;
        IF pos >= size OR pg_catalog.get_byte($1,pos) <> 58 THEN
          RAISE EXCEPTION 'canonical JSON object colon required' USING ERRCODE='23514';
        END IF;
        pos := pos + 1;
        states[depth] := 3;
        need_value := true;
        CONTINUE;
      END IF;
    ELSIF states[depth] IN (4,5) THEN
      IF NOT (states[depth] = 4 AND ch = 93) THEN
        need_value := true;
        CONTINUE;
      END IF;
    ELSIF states[depth] IN (3,6) THEN
      IF ch = 44 THEN
        states[depth] := CASE states[depth] WHEN 3 THEN 2 ELSE 5 END;
        pos := pos + 1;
        CONTINUE;
      ELSIF ch <> (CASE states[depth] WHEN 3 THEN 125 ELSE 93 END) THEN
        RAISE EXCEPTION 'canonical JSON container separator required' USING ERRCODE='23514';
      END IF;
    ELSE
      RAISE EXCEPTION 'invalid canonical JSON lexer state' USING ERRCODE='23514';
    END IF;
    pos := pos + 1;
    depth := depth - 1;
    IF depth = 0 THEN RETURN pos; END IF;
    states[depth] := CASE WHEN states[depth] <= 3 THEN 3 ELSE 6 END;
  END LOOP;
END
""".replace("@SCHEMA@", namespace),
        "validate_canonical_json(bytea)": r"""
DECLARE
  size integer := pg_catalog.octet_length($1);
  finish integer;
BEGIN
  IF $1 IS NULL OR size < 2 OR pg_catalog.get_byte($1,0) <> 123 THEN
    RAISE EXCEPTION 'canonical JSON object root required' USING ERRCODE='23514';
  END IF;
  PERFORM pg_catalog.convert_from($1,'UTF8');
  finish := @SCHEMA@.canonical_value($1,0);
  IF finish <> size THEN
    RAISE EXCEPTION 'noncanonical trailing JSON bytes' USING ERRCODE='23514';
  END IF;
END
""".replace("@SCHEMA@", namespace),
        "canonical_object_member(bytea,text)": r"""
DECLARE
  pos integer := 1;
  finish integer;
  item record;
  wanted bytea;
BEGIN
  PERFORM @SCHEMA@.validate_canonical_json($1);
  IF $2 IS NULL THEN
    RAISE EXCEPTION 'canonical JSON member name required' USING ERRCODE='23514';
  END IF;
  wanted := pg_catalog.convert_to($2,'UTF8');
  IF pg_catalog.get_byte($1,pos) = 125 THEN RETURN NULL; END IF;
  LOOP
    SELECT * INTO STRICT item FROM @SCHEMA@.canonical_string($1,pos);
    pos := item.next_position + 1;
    finish := @SCHEMA@.canonical_value($1,pos);
    IF item.decoded_key = wanted THEN
      RETURN pg_catalog.substring($1,pos+1,finish-pos);
    END IF;
    IF pg_catalog.get_byte($1,finish) = 125 THEN RETURN NULL; END IF;
    pos := finish + 1;
  END LOOP;
END
""".replace("@SCHEMA@", namespace),
    }
