"""Dependency-free strict canonical JSON profile shared by security contracts."""
from __future__ import annotations
import json, math
from typing import Any

class CanonicalJSONError(ValueError): pass

def canonical_json_bytes(value: Any) -> bytes:
    def encode(item: Any) -> str:
        if item is None: return "null"
        if item is True: return "true"
        if item is False: return "false"
        if isinstance(item,str): return json.dumps(item,ensure_ascii=False,separators=(",",":"))
        if isinstance(item,int) and not isinstance(item,bool):
            if abs(item)>9_007_199_254_740_991: raise CanonicalJSONError("JSON integer exceeds the interoperable exact range")
            return str(item)
        if isinstance(item,float):
            if not math.isfinite(item): raise CanonicalJSONError("non-finite JSON numbers are forbidden")
            raise CanonicalJSONError("floating-point JSON numbers are forbidden")
        if isinstance(item,list): return "["+",".join(encode(x) for x in item)+"]"
        if isinstance(item,dict):
            if not all(isinstance(k,str) for k in item): raise CanonicalJSONError("JSON object keys must be strings")
            return "{"+",".join(encode(k)+":"+encode(item[k]) for k in sorted(item))+"}"
        raise CanonicalJSONError(f"unsupported canonical JSON type: {type(item).__name__}")
    return encode(value).encode("utf-8")
