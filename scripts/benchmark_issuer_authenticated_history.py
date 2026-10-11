"""Manual, isolated PostgreSQL issuer-history latency and memory qualification.

Run from the repository root, outside ordinary CI::

    python -m scripts.benchmark_issuer_authenticated_history \
        --sizes 100 1000 10000 --samples 3 --output work/history-benchmark.json

The DSN comes from DUDZIAN_BENCHMARK_POSTGRES_DSN, or the isolated localhost:55432
test server by default. Only loopback hosts are accepted. Each size gets a fresh
child process, UUID-named schema and three unrelated UUID-named roles; cleanup
removes only those names. No existing CHA or entitlement data is accessed.

Fixture setup uses the privileged, offline connection to COPY legal records
built by the frozen build_record/canonical_json_bytes functions, and updates
the matching head in the same transaction. This avoids quadratic setup through
repeated full-chain append checks. It is exclusively a benchmark fixture:
runtime qualification and a complete authoritative read must verify the seeded
chain before measurements. Timed appends use the unchanged production adapter.

Latency samples are uninstrumented. Separate tracemalloc probes deliberately
report their instrumentation overhead. PostgreSQL statistics are observed, not
reset or reconfigured; schema relation counters provide isolated workload
evidence while global database/WAL deltas may include unrelated server work.
An additional append, after the comparable samples, uses an independent
pg_locks observer to bound the granted stream-relation transaction marker.
It does not supply exact row-lock telemetry or alter primary measurements.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from bot_core.authenticated_issuer_history import (
    NO_PREDECESSOR,
    HistoryEventIdentity,
    HistoryRecord,
    HistoryStreamIdentity,
    build_record,
    canonical_json_bytes,
)
from bot_core.postgresql_authenticated_issuer_history import PostgreSQLAuthenticatedIssuerHistory
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.postgresql_issuer_history_schema import (
    PostgreSQLIssuerHistoryProvisioning,
    provision_postgresql_issuer_history,
    provision_postgresql_issuer_history_stream,
)

_DEFAULT_DSN = "host=127.0.0.1 port=55432 dbname=postgres user=postgres"
_T = TypeVar("_T")


def _event(number: int) -> HistoryEventIdentity:
    return HistoryEventIdentity(
        f"bench-operation-{number}", f"bench-attempt-{number}", f"bench-proof-{number}"
    )


def _payload(number: int) -> dict[str, object]:
    return {
        "fixture": "benchmark-only-not-an-issuance-decision",
        "number": number,
        "unicode": "żółć",
    }


def _scalar(value: str) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _role_connection(dsn: str, role: str) -> PostgreSQLConnectionConfig:
    return PostgreSQLConnectionConfig(make_conninfo(dsn, user=role))


@contextmanager
def _fixture(
    dsn: str, size: int
) -> Iterator[tuple[PostgreSQLIssuerHistoryProvisioning, HistoryStreamIdentity]]:
    # Independent UUIDs intentionally prevent role-name substring normalization
    # from making a benchmark fixture depend on overlapping principal names.
    config = PostgreSQLIssuerHistoryProvisioning(
        "bench_hist_" + uuid.uuid4().hex,
        "bench_owner_" + uuid.uuid4().hex,
        "bench_run_" + uuid.uuid4().hex,
        "bench_admin_" + uuid.uuid4().hex,
        "isolated-benchmark-trust",
    )
    stream = HistoryStreamIdentity(
        f"benchmark-stream-{size}",
        "benchmark-authority",
        "PRODUCTION_LOCAL",
        "BENCHMARK",
        config.trust_domain,
        "benchmark-only-product",
        1,
    )
    installed = False
    try:
        provision_postgresql_issuer_history(PostgreSQLConnectionConfig(dsn), config)
        installed = True
        provision_postgresql_issuer_history_stream(
            _role_connection(dsn, config.admin_role), schema=config.schema, stream=stream
        )
        yield config, stream
    finally:
        if installed:
            with psycopg.connect(dsn, autocommit=True) as conn:
                conn.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                        sql.Identifier(config.schema)
                    )
                )
                for role in (config.runtime_role, config.admin_role, config.schema_owner_role):
                    conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))


def _seed(
    dsn: str, config: PostgreSQLIssuerHistoryProvisioning, stream: HistoryStreamIdentity, size: int
) -> HistoryRecord:
    stream_bytes = canonical_json_bytes(stream.material())
    predecessor = NO_PREDECESSOR
    tail: HistoryRecord | None = None
    with psycopg.connect(dsn, autocommit=True) as conn, conn.transaction():
        conn.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(config.schema_owner_role)))
        with conn.cursor().copy(
            sql.SQL(
                "COPY {}.records(stream_identity,sequence,predecessor_authenticated_digest,"
                "event_identity,logical_operation_id,issuance_attempt_id,root_proof_id,"
                "event_digest,authenticated_digest,canonical_record) FROM STDIN"
            ).format(sql.Identifier(config.schema))
        ) as copy:
            for number in range(1, size + 1):
                tail = build_record(stream, number, predecessor, _event(number), _payload(number))
                event = tail.event_identity
                copy.write_row(
                    (
                        stream_bytes,
                        number,
                        predecessor,
                        canonical_json_bytes(event.material()),
                        _scalar(event.logical_operation_id),
                        _scalar(event.issuance_attempt_id),
                        _scalar(event.root_proof_id),
                        tail.event_digest,
                        tail.authenticated_digest,
                        canonical_json_bytes(tail.unsigned_material()),
                    )
                )
                predecessor = tail.authenticated_digest
        conn.execute(
            sql.SQL(
                "UPDATE {}.streams SET head_sequence=%s,head_digest=%s WHERE stream_identity=%s"
            ).format(sql.Identifier(config.schema)),
            (size, predecessor, stream_bytes),
        )
        conn.execute(sql.SQL("ANALYZE {}.records").format(sql.Identifier(config.schema)))
    if tail is None:
        raise ValueError("benchmark size must be positive")
    return tail


def _rss_kib() -> int | None:
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return None


def _stats(conn: psycopg.Connection[Any], schema: str) -> dict[str, Any]:
    conn.execute("SELECT pg_catalog.pg_stat_clear_snapshot()")
    tables = conn.execute(
        "SELECT relname,seq_scan,seq_tup_read,idx_scan,idx_tup_fetch,n_tup_ins,n_tup_upd,n_tup_del "
        "FROM pg_catalog.pg_stat_user_tables WHERE schemaname=%s ORDER BY relname",
        (schema,),
    ).fetchall()
    table_io = conn.execute(
        "SELECT relname,heap_blks_read,heap_blks_hit,idx_blks_read,idx_blks_hit,"
        "toast_blks_read,toast_blks_hit "
        "FROM pg_catalog.pg_statio_user_tables WHERE schemaname=%s ORDER BY relname",
        (schema,),
    ).fetchall()
    database = conn.execute(
        "SELECT xact_commit,xact_rollback,blks_read,blks_hit,tup_returned,tup_fetched,"
        "tup_inserted,tup_updated,temp_files,temp_bytes,deadlocks "
        "FROM pg_catalog.pg_stat_database WHERE datname=current_database()"
    ).fetchone()
    lsn = conn.execute("SELECT pg_catalog.pg_current_wal_lsn()::text").fetchone()[0]
    return {
        "tables": {row[0]: list(row[1:]) for row in tables},
        "table_io": {row[0]: list(row[1:]) for row in table_io},
        "database": list(database),
        "wal_lsn": lsn,
    }


def _difference(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    def lsn_number(value: str) -> int:
        high, low = value.split("/")
        return (int(high, 16) << 32) + int(low, 16)

    return {
        "tables": {
            key: [a - b for a, b in zip(values, before["tables"][key], strict=True)]
            for key, values in after["tables"].items()
        },
        "table_io": {
            key: [a - b for a, b in zip(values, before["table_io"][key], strict=True)]
            for key, values in after["table_io"].items()
        },
        "database": [a - b for a, b in zip(after["database"], before["database"], strict=True)],
        "global_wal_bytes": lsn_number(after["wal_lsn"]) - lsn_number(before["wal_lsn"]),
    }


def _measure(
    operation: Callable[[], _T],
    observer: psycopg.Connection[Any],
    schema: str,
    *,
    trace: bool = False,
) -> tuple[_T, dict[str, Any]]:
    gc.collect()
    before_stats = _stats(observer, schema)
    before_usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_before = _rss_kib()
    if trace:
        tracemalloc.start()
    started = time.perf_counter()
    result = operation()
    elapsed = time.perf_counter() - started
    if trace:
        traced_current, traced_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    else:
        traced_current, traced_peak = None, None
    after_usage = resource.getrusage(resource.RUSAGE_SELF)
    after_stats = _stats(observer, schema)
    database_delta = _difference(after_stats, before_stats)
    record_stats = database_delta["tables"]["records"]
    return result, {
        "elapsed_seconds": elapsed,
        "python_user_cpu_seconds": after_usage.ru_utime - before_usage.ru_utime,
        "python_system_cpu_seconds": after_usage.ru_stime - before_usage.ru_stime,
        "rss_before_kib": rss_before,
        "rss_after_kib": _rss_kib(),
        "process_peak_rss_kib": after_usage.ru_maxrss,
        "python_tracemalloc_current_bytes": traced_current,
        "python_tracemalloc_peak_bytes": traced_peak,
        "tracemalloc_enabled": trace,
        "db_delta": database_delta,
        "record_tuples_visited": record_stats[1] + record_stats[3],
        "host_load_average": list(os.getloadavg()),
    }


def _lock_probe(
    dsn: str,
    config: PostgreSQLIssuerHistoryProvisioning,
    stream: HistoryStreamIdentity,
    predecessor: HistoryRecord,
    *,
    poll_seconds: float = 0.01,
) -> dict[str, Any]:
    """Observe one separate append; never instrument primary latency samples.

    PostgreSQL does not expose every uncontended row lock in pg_locks. The
    granted RowShareLock on the exact streams relation identifies SELECT FOR
    UPDATE entering its lock-holding transaction. Its observed duration is a
    transaction marker, not exact row-lock telemetry. The isolated fixture has
    one stream and one writer, with no lock contention.
    """
    application_name = "issuer-history-lock-probe-" + uuid.uuid4().hex
    runtime = PostgreSQLAuthenticatedIssuerHistory(
        PostgreSQLConnectionConfig(
            make_conninfo(dsn, user=config.runtime_role, application_name=application_name)
        ),
        schema=config.schema,
        stream=stream,
    )
    number = predecessor.sequence + 1

    def append() -> tuple[HistoryRecord, bool, float, float]:
        started = time.perf_counter()
        record, replayed = runtime.append(
            expected_digest=predecessor.authenticated_digest,
            event_identity=_event(number),
            payload=_payload(number),
        )
        return record, replayed, started, time.perf_counter()

    with psycopg.connect(dsn, autocommit=True) as observer:
        relation_oid = observer.execute(
            "SELECT pg_catalog.to_regclass(%s)::oid", (f"{config.schema}.streams",)
        ).fetchone()[0]
        if relation_oid is None:
            raise RuntimeError("lock probe streams relation is missing")
        started = time.perf_counter()
        last_absent: tuple[float, float] | None = None
        first_present: tuple[float, float] | None = None
        last_present: tuple[float, float] | None = None
        first_absent_after: tuple[float, float] | None = None
        polls = 0
        present_polls = 0
        observed_pid: int | None = None
        maximum_observation_gap = 0.0
        previous_query_end = started

        def observe() -> tuple[bool, tuple[float, float]]:
            nonlocal polls, observed_pid, maximum_observation_gap, previous_query_end
            query_start = time.perf_counter()
            rows = observer.execute(
                "SELECT DISTINCT l.pid FROM pg_catalog.pg_locks l "
                "JOIN pg_catalog.pg_stat_activity a ON a.pid=l.pid "
                "WHERE l.locktype='relation' AND l.relation=%s "
                "AND l.mode='RowShareLock' AND l.granted "
                "AND a.usename=%s AND a.application_name=%s",
                (relation_oid, config.runtime_role, application_name),
            ).fetchall()
            query_end = time.perf_counter()
            maximum_observation_gap = max(maximum_observation_gap, query_start - previous_query_end)
            previous_query_end = query_end
            polls += 1
            if rows:
                if len(rows) != 1 or (observed_pid is not None and observed_pid != rows[0][0]):
                    raise RuntimeError("lock probe did not identify one exact runtime backend")
                observed_pid = rows[0][0]
            return bool(rows), (query_start - started, query_end - started)

        present, last_absent = observe()
        if present:
            raise RuntimeError("lock probe unexpectedly observed a lock before its append")
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(append)
            while True:
                present, interval = observe()
                if present:
                    present_polls += 1
                    if first_present is None:
                        first_present = interval
                    last_present = interval
                elif first_present is not None:
                    first_absent_after = interval
                    break
                else:
                    last_absent = interval
                    if pending.done():
                        # Propagate an append exception instead of inventing a
                        # lock duration for an operation that failed.
                        pending.result()
                        raise RuntimeError("append lock interval was not observed")
                time.sleep(poll_seconds)
            record, replayed, append_started, append_finished = pending.result()
        if replayed or record.sequence != number:
            raise RuntimeError("lock probe did not commit one exact successor")
        if last_absent is None or first_present is None or last_present is None:
            raise RuntimeError("lock probe acquisition bounds are missing")
        if first_absent_after is None:
            raise RuntimeError("lock probe release bounds are missing")
        # Each query observes pg_locks sometime between its start and end.
        # These conservative bounds include the query's own uncertainty.
        acquisition_bounds = (last_absent[0], first_present[1])
        release_bounds = (last_present[0], first_absent_after[1])
        verification_started = time.perf_counter()
        if runtime.current_head() != record:
            raise RuntimeError("lock probe append lacks an exact authoritative retained head")
        verification_elapsed = time.perf_counter() - verification_started
    return {
        "retained_before": predecessor.sequence,
        "verified_sequence_after": number,
        "measurement": "Independent pg_locks observer; granted streams RowShareLock marker.",
        "row_lock_telemetry": False,
        "uncontended_single_stream_single_writer": True,
        "excluded_from_primary_latency_and_memory_samples": True,
        "poll_interval_seconds": poll_seconds,
        "poll_count": polls,
        "present_poll_count": present_polls,
        "maximum_gap_between_observation_queries_seconds": maximum_observation_gap,
        "acquisition_bounds_relative_seconds": list(acquisition_bounds),
        "release_bounds_relative_seconds": list(release_bounds),
        "lock_marker_duration_lower_seconds": max(0.0, release_bounds[0] - acquisition_bounds[1]),
        "lock_marker_duration_upper_seconds": release_bounds[1] - acquisition_bounds[0],
        "separate_append_wall_seconds": append_finished - append_started,
        "post_probe_authoritative_read_seconds": verification_elapsed,
        "runtime_backend_pid": observed_pid,
        "limitations": (
            "Relation RowShareLock begins before the row lock is acquired and lasts until "
            "transaction end. Bounds describe that granted transaction marker; they do not "
            "claim exact row-lock acquisition or waiting time. Observer queries can perturb "
            "this separate probe and are absent from all primary latency samples."
        ),
    }


def _worker(size: int, samples: int) -> dict[str, Any]:
    dsn = os.environ.get("DUDZIAN_BENCHMARK_POSTGRES_DSN", _DEFAULT_DSN)
    connection_options = conninfo_to_dict(dsn)
    if connection_options.get("service") is not None or os.environ.get("PGSERVICE"):
        raise ValueError("manual benchmark forbids implicit PostgreSQL service connection settings")
    if (
        connection_options.get("host") not in {"localhost", "127.0.0.1", "::1"}
        or (
            connection_options.get("hostaddr") is not None
            and connection_options["hostaddr"] not in {"127.0.0.1", "::1"}
        )
        or (
            os.environ.get("PGHOSTADDR") is not None
            and os.environ["PGHOSTADDR"] not in {"127.0.0.1", "::1"}
        )
    ):
        raise ValueError("manual benchmark requires an isolated loopback PostgreSQL server")
    report: dict[str, Any] = {
        "retained_size": size,
        "pid": os.getpid(),
        "read": [],
        "append": [],
        "replay": [],
    }
    with _fixture(dsn, size) as (config, stream), psycopg.connect(dsn, autocommit=True) as observer:
        setup_started = time.perf_counter()
        tail = _seed(dsn, config, stream, size)
        runtime = PostgreSQLAuthenticatedIssuerHistory(
            _role_connection(dsn, config.runtime_role), schema=config.schema, stream=stream
        )
        if runtime.current_head() != tail:
            raise RuntimeError(
                "offline seeded fixture did not pass authoritative full-chain verification"
            )
        runtime.verify()
        report["fixture_setup_and_authoritative_verification_seconds"] = (
            time.perf_counter() - setup_started
        )
        print(
            f"retained={size}: fixture verified; timing unchanged adapter",
            file=sys.stderr,
            flush=True,
        )
        report["server"] = dict(
            zip(
                (
                    "version",
                    "shared_preload_libraries",
                    "track_io_timing",
                    "track_functions",
                    "fsync",
                    "synchronous_commit",
                ),
                observer.execute(
                    "SELECT version(),current_setting('shared_preload_libraries'),"
                    "current_setting('track_io_timing'),"
                    "current_setting('track_functions'),current_setting('fsync'),current_setting('synchronous_commit')"
                ).fetchone(),
                strict=True,
            )
        )
        report["pg_stat_statements"] = {
            "available": bool(
                observer.execute(
                    "SELECT 1 FROM pg_catalog.pg_extension WHERE extname='pg_stat_statements'"
                ).fetchone()
            ),
            "collected": False,
            "reason": (
                "No extension installed or benchmark uses portable relation counters; "
                "server configuration unchanged."
            ),
        }
        report["storage_bytes"] = observer.execute(
            "SELECT pg_catalog.pg_total_relation_size(%s::regclass)", (f"{config.schema}.records",)
        ).fetchone()[0]
        report["fixture_payload_canonical_bytes"] = len(canonical_json_bytes(_payload(size)))
        report["fixture_tail_canonical_record_bytes"] = len(
            canonical_json_bytes(tail.unsigned_material())
        )
        for number in range(samples):
            current, measurement = _measure(runtime.current_head, observer, config.schema)
            if current != tail:
                raise RuntimeError("timed authority read differs from exact retained head")
            measurement["retained_before"] = size
            report["read"].append(measurement)
            print(
                f"retained={size} current_head[{number}]={measurement['elapsed_seconds']:.3f}s",
                file=sys.stderr,
                flush=True,
            )
        for number in range(samples):
            next_number = size + number + 1
            expected = tail.authenticated_digest
            result, measurement = _measure(
                partial(
                    runtime.append,
                    expected_digest=expected,
                    event_identity=_event(next_number),
                    payload=_payload(next_number),
                ),
                observer,
                config.schema,
            )
            tail, replayed = result
            if replayed or tail.sequence != next_number:
                raise RuntimeError("timed append did not publish exactly one legal successor")
            measurement["retained_before"] = next_number - 1
            report["append"].append(measurement)
            print(
                f"retained={next_number - 1} append[{number}]: "
                f"{measurement['elapsed_seconds']:.3f}s",
                file=sys.stderr,
                flush=True,
            )
        for number in range(samples):
            result, measurement = _measure(
                lambda: runtime.append(
                    expected_digest=NO_PREDECESSOR,
                    event_identity=tail.event_identity,
                    payload=_payload(tail.sequence),
                ),
                observer,
                config.schema,
            )
            if result != (tail, True):
                raise RuntimeError("exact replay did not resolve the original committed identity")
            measurement["retained_before"] = tail.sequence
            report["replay"].append(measurement)
            print(
                f"retained={tail.sequence} replay[{number}]={measurement['elapsed_seconds']:.3f}s",
                file=sys.stderr,
                flush=True,
            )
        _, report["read_memory_probe"] = _measure(
            runtime.current_head, observer, config.schema, trace=True
        )
        report["read_memory_probe"]["retained_before"] = tail.sequence
        next_number = tail.sequence + 1
        expected = tail.authenticated_digest
        result, report["append_memory_probe"] = _measure(
            lambda: runtime.append(
                expected_digest=expected,
                event_identity=_event(next_number),
                payload=_payload(next_number),
            ),
            observer,
            config.schema,
            trace=True,
        )
        tail, replayed = result
        report["append_memory_probe"]["retained_before"] = next_number - 1
        if replayed:
            raise RuntimeError("memory append probe unexpectedly replayed")
        print(f"retained={size}: separate memory probes complete", file=sys.stderr, flush=True)
        with (
            psycopg.connect(
                _role_connection(dsn, config.runtime_role).dsn, autocommit=True
            ) as conn,
            conn.transaction(),
        ):
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            conn.execute("SET LOCAL search_path=pg_catalog")
            runtime._qualify(conn)
            report["server_read_retained_size"] = tail.sequence
            report["server_read_explain"] = conn.execute(
                sql.SQL(
                    "EXPLAIN (ANALYZE,BUFFERS,WAL,TIMING OFF,FORMAT JSON) "
                    "SELECT {}.read_stream(%s,false)"
                ).format(sql.Identifier(config.schema)),
                (canonical_json_bytes(stream.material()),),
            ).fetchone()[0]
        report["final_verified_sequence"] = runtime.current_head().sequence
        report["latency_summary"] = {
            operation: {
                "minimum_seconds": min(item["elapsed_seconds"] for item in report[operation]),
                "median_seconds": statistics.median(
                    item["elapsed_seconds"] for item in report[operation]
                ),
                "maximum_seconds": max(item["elapsed_seconds"] for item in report[operation]),
            }
            for operation in ("read", "append", "replay")
        }
        report["append_lock_probe"] = _lock_probe(dsn, config, stream, tail)
        print(
            f"retained={size}: separate observed lock probe complete", file=sys.stderr, flush=True
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 1000, 10000])
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("work/history-benchmark.json"))
    parser.add_argument("--worker-size", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        args.samples < 1
        or any(size < 1 for size in args.sizes)
        or (args.worker_size is not None and args.worker_size < 1)
    ):
        parser.error("sizes and samples must be positive")
    if args.worker_size is not None:
        print(json.dumps(_worker(args.worker_size, args.samples), ensure_ascii=False))
        return
    report = {
        "schema": "CryptoHunter.IssuerHistoryManualBenchmarkV1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "samples_per_operation": args.samples,
        "validation_work_source_review": {
            "authoritative_read": {"complete_sql_chain_passes": 1, "python_snapshots": 1},
            "new_append": {"complete_sql_chain_passes": 4, "python_snapshots": 2},
            "exact_replay": {"complete_sql_chain_passes": 1, "python_snapshots": 1},
            "evidence": (
                "Reviewed control flow in the source-hashed adapter/schema; not runtime "
                "function-call telemetry. Relation tuple visits are observed separately."
            ),
        },
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, stdout=subprocess.PIPE, text=True
        ).stdout.strip(),
        "git_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.strip()
        ),
        "runtime_source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "bot_core/postgresql_authenticated_issuer_history.py",
                "bot_core/postgresql_issuer_history_schema.py",
                "bot_core/postgresql_issuer_history_canonical_sql.py",
            )
        },
        "measurement_source": (
            "Runtime source hashes identify the measured working copy; "
            "git_commit identifies its base revision."
        ),
        "cgroup_limits": {
            name: path.read_text().strip()
            for name, path in {
                "cpu.max": Path("/sys/fs/cgroup/cpu.max"),
                "memory.max": Path("/sys/fs/cgroup/memory.max"),
            }.items()
            if path.exists()
        },
        "method": (
            "Fresh child per size; offline canonical COPY fixture then authoritative "
            "qualification/verification; unchanged production operations; warm loopback "
            "measurements; separate traced memory probes."
        ),
        "statistics_columns": {
            "tables": [
                "seq_scan",
                "seq_tup_read",
                "idx_scan",
                "idx_tup_fetch",
                "n_tup_ins",
                "n_tup_upd",
                "n_tup_del",
            ],
            "table_io": [
                "heap_blks_read",
                "heap_blks_hit",
                "idx_blks_read",
                "idx_blks_hit",
                "toast_blks_read",
                "toast_blks_hit",
            ],
            "database": [
                "xact_commit",
                "xact_rollback",
                "blks_read",
                "blks_hit",
                "tup_returned",
                "tup_fetched",
                "tup_inserted",
                "tup_updated",
                "temp_files",
                "temp_bytes",
                "deadlocks",
            ],
        },
        "limitations": [
            "Single local host; payload approximately 100 canonical bytes; warm cache; "
            "no workload SLA or remote latency inference.",
            "Global database/WAL counters include observer/background/concurrent work; "
            "schema table counters are isolated.",
            "Linux ru_maxrss is the child process high-water RSS, including prior fixture "
            "and memory probes; tracemalloc excludes libpq and PostgreSQL backend allocations.",
            "EXPLAIN profiles server read_stream separately, excludes adapter qualification/"
            "Python decoding, and changes neither runtime code nor server configuration.",
            "Lock-duration bounds observe a granted stream-relation RowShareLock in a separate "
            "uncontended append after all comparable samples; they are not exact row-lock timing.",
        ],
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.benchmark_issuer_authenticated_history",
                "--worker-size",
                str(size),
                "--samples",
                str(args.samples),
            ],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        result = json.loads(completed.stdout)
        report["results"].append(result)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(
            json.dumps({"retained_size": size, "latency_summary": result["latency_summary"]}),
            flush=True,
        )
    print(f"Saved benchmark report to {args.output}", flush=True)


if __name__ == "__main__":
    main()
