"""Dynamic protobuf definitions for telemetry snapshots."""
from __future__ import annotations
from typing import Any  # added by patch

from typing import Any, Dict

try:
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
except ImportError:  # pragma: no cover - protobuf opcjonalny w środowisku testowym
    descriptor_pb2 = descriptor_pool = message_factory = None  # type: ignore
    PROTOBUF_AVAILABLE = False
else:
    PROTOBUF_AVAILABLE = True

_FILE_NAME = "telemetry.proto"
if PROTOBUF_AVAILABLE:
    _POOL = descriptor_pool.Default()

    try:
        _POOL.FindFileByName(_FILE_NAME)
    except KeyError:
        file_proto = descriptor_pb2.FileDescriptorProto()
        file_proto.name = _FILE_NAME
        file_proto.package = "telemetry"

        endpoint_msg = file_proto.message_type.add()
        endpoint_msg.name = "EndpointMetric"
        field = endpoint_msg.field.add()
        field.name = "endpoint"
        field.number = 1
        field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING

    field = endpoint_msg.field.add()
    field.name = "total_calls"
    field.number = 2
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_UINT64

    field = endpoint_msg.field.add()
    field.name = "total_errors"
    field.number = 3
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_UINT64

    field = endpoint_msg.field.add()
    field.name = "avg_latency_ms"
    field.number = 4
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE

    field = endpoint_msg.field.add()
    field.name = "max_latency_ms"
    field.number = 5
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE

    field = endpoint_msg.field.add()
    field.name = "last_latency_ms"
    field.number = 6
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE

    bucket_msg = file_proto.message_type.add()
    bucket_msg.name = "RateLimitBucket"

    field = bucket_msg.field.add()
    field.name = "name"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING

    field = bucket_msg.field.add()
    field.name = "capacity"
    field.number = 2
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_UINT32

    field = bucket_msg.field.add()
    field.name = "count"
    field.number = 3
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_UINT32

    field = bucket_msg.field.add()
    field.name = "usage"
    field.number = 4
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE

    field = bucket_msg.field.add()
    field.name = "max_usage"
    field.number = 5
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE

    field = bucket_msg.field.add()
    field.name = "reset_in_seconds"
    field.number = 6
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE

    field = bucket_msg.field.add()
    field.name = "window_seconds"
    field.number = 7
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE

    field = bucket_msg.field.add()
    field.name = "alert_active"
    field.number = 8
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_BOOL

    snapshot_msg = file_proto.message_type.add()
    snapshot_msg.name = "ApiTelemetrySnapshot"

    def _add_field(msg, name, number, field_type, label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL):
        fld = msg.field.add()
        fld.name = name
        fld.number = number
        fld.label = label
        fld.type = field_type
        return fld

    _add_field(snapshot_msg, "schema_version", 1, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
    _add_field(snapshot_msg, "timestamp_ns", 2, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)
    _add_field(snapshot_msg, "exchange", 3, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _add_field(snapshot_msg, "mode", 4, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _add_field(snapshot_msg, "total_calls", 5, descriptor_pb2.FieldDescriptorProto.TYPE_UINT64)
    _add_field(snapshot_msg, "total_errors", 6, descriptor_pb2.FieldDescriptorProto.TYPE_UINT64)
    _add_field(snapshot_msg, "avg_latency_ms", 7, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
    _add_field(snapshot_msg, "max_latency_ms", 8, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
    _add_field(snapshot_msg, "last_latency_ms", 9, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
    _add_field(snapshot_msg, "consecutive_errors", 10, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
    _add_field(snapshot_msg, "window_calls", 11, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
    _add_field(snapshot_msg, "window_errors", 12, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
    _add_field(snapshot_msg, "last_endpoint", 13, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    fld = _add_field(snapshot_msg, "endpoints", 14, descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE, descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED)
    fld.type_name = ".telemetry.EndpointMetric"
    fld = _add_field(snapshot_msg, "buckets", 15, descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE, descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED)
    fld.type_name = ".telemetry.RateLimitBucket"
    _add_field(snapshot_msg, "current_window_usage", 16, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
    _add_field(snapshot_msg, "rate_limit_window_seconds", 17, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)

    ack_msg = file_proto.message_type.add()
    ack_msg.name = "TelemetryAck"
    _add_field(ack_msg, "received", 1, descriptor_pb2.FieldDescriptorProto.TYPE_BOOL)

    service = file_proto.service.add()
    service.name = "TelemetryStream"
    method = service.method.add()
    method.name = "PushSnapshot"
    method.input_type = ".telemetry.ApiTelemetrySnapshot"
    method.output_type = ".telemetry.TelemetryAck"

    _POOL.Add(file_proto)

    _FACTORY = message_factory.MessageFactory(_POOL)

    EndpointMetric = _FACTORY.GetPrototype(_POOL.FindMessageTypeByName("telemetry.EndpointMetric"))
    RateLimitBucketMsg = _FACTORY.GetPrototype(_POOL.FindMessageTypeByName("telemetry.RateLimitBucket"))
    ApiTelemetrySnapshot = _FACTORY.GetPrototype(_POOL.FindMessageTypeByName("telemetry.ApiTelemetrySnapshot"))
    TelemetryAck = _FACTORY.GetPrototype(_POOL.FindMessageTypeByName("telemetry.TelemetryAck"))


    def build_snapshot_message(snapshot: Dict[str, Any], *, exchange: str, mode: str) -> Any:
        msg = ApiTelemetrySnapshot()
        msg.schema_version = int(snapshot.get("schema_version", 1))
        msg.timestamp_ns = int(snapshot.get("timestamp_ns", 0))
        msg.exchange = exchange
        msg.mode = mode
        msg.total_calls = int(snapshot.get("total_calls", 0))
        msg.total_errors = int(snapshot.get("total_errors", 0))
        msg.avg_latency_ms = float(snapshot.get("avg_latency_ms", 0.0))
        msg.max_latency_ms = float(snapshot.get("max_latency_ms", 0.0))
        msg.last_latency_ms = float(snapshot.get("last_latency_ms", 0.0))
        msg.consecutive_errors = int(snapshot.get("consecutive_errors", 0))
        msg.window_calls = int(snapshot.get("window_calls", 0))
        msg.window_errors = int(snapshot.get("window_errors", 0))
        if snapshot.get("last_endpoint"):
            msg.last_endpoint = str(snapshot["last_endpoint"])
        msg.current_window_usage = float(snapshot.get("current_window_usage") or 0.0)
        msg.rate_limit_window_seconds = float(snapshot.get("rate_limit_window_seconds", 0.0))

        for endpoint_name, data in snapshot.get("endpoints", {}).items():
            endpoint_msg = msg.endpoints.add()
            endpoint_msg.endpoint = str(endpoint_name)
            endpoint_msg.total_calls = int(data.get("total_calls", 0))
            endpoint_msg.total_errors = int(data.get("total_errors", 0))
            endpoint_msg.avg_latency_ms = float(data.get("avg_latency_ms", 0.0))
            endpoint_msg.max_latency_ms = float(data.get("max_latency_ms", 0.0))
            endpoint_msg.last_latency_ms = float(data.get("last_latency_ms", 0.0))

        for bucket in snapshot.get("rate_limit_buckets", []):
            bucket_msg = msg.buckets.add()
            bucket_msg.name = str(bucket.get("name", ""))
            bucket_msg.capacity = int(bucket.get("capacity", 0))
            bucket_msg.count = int(bucket.get("count", 0))
            bucket_msg.usage = float(bucket.get("usage", 0.0))
            bucket_msg.max_usage = float(bucket.get("max_usage", 0.0))
            bucket_msg.reset_in_seconds = float(bucket.get("reset_in_seconds", 0.0))
            bucket_msg.window_seconds = float(bucket.get("window_seconds", 0.0))
            bucket_msg.alert_active = bool(bucket.get("alert_active", False))

        return msg

else:  # pragma: no cover - fallback bez protobuf

    EndpointMetric = RateLimitBucketMsg = ApiTelemetrySnapshot = TelemetryAck = None  # type: ignore

    def build_snapshot_message(snapshot: Dict[str, Any], *, exchange: str, mode: str) -> Any:
        raise RuntimeError(
            "Obsługa protobuf jest niedostępna – zainstaluj pakiet 'protobuf' aby włączyć serializację telemetryjną."
        )


__all__ = [
    "ApiTelemetrySnapshot",
    "EndpointMetric",
    "RateLimitBucketMsg",
    "TelemetryAck",
    "build_snapshot_message",
]
