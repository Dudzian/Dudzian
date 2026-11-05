"""Wyłącznie MODELED: żadna odmowa w tym pliku nie pochodzi od jądra."""

import struct
from dataclasses import replace

import pytest

from scripts.diagnostics import stage9_verifier_seal_model as model


@pytest.mark.parametrize(
    "name",
    [
        "execve",
        "execveat",
        "fork",
        "vfork",
        "clone",
        "clone3",
        "sendmsg",
        "sendmmsg",
        "recvmsg",
        "recvmmsg",
        "ptrace",
        "pidfd_getfd",
        "io_uring_setup",
        "io_uring_enter",
        "io_uring_register",
        "prctl",
        "seccomp",
    ],
)
def test_adversarial_syscalls_are_modeled_reject(name: str) -> None:
    assert model.syscall_decision(model.SYSCALLS[name]) == model.Decision("REJECT")


@pytest.mark.parametrize("number", [2, 85, 257, 437, 105, 106, 117, 119, 272, 308, 9999])
def test_default_deny_covers_file_loads_credentials_namespaces_and_unknown(number: int) -> None:
    assert model.syscall_decision(number).result == "REJECT"


@pytest.mark.parametrize("number", [0, 1, 41, 59, 288])
@pytest.mark.parametrize("architecture", [0x40000003, 0xC00000B7, 0])
def test_other_abis_are_killed_only_in_model(number: int, architecture: int) -> None:
    assert (
        model.evaluate(
            model.compile_filter(), model.seccomp_data(number, architecture=architecture)
        )
        == model.KILL
    )


@pytest.mark.parametrize("number", [0, 1, 41, 59, 512, 547])
def test_x32_cannot_reuse_an_allowed_lp64_number(number: int) -> None:
    assert (
        model.evaluate(model.compile_filter(), model.seccomp_data(number | model.X32_BIT))
        == model.KILL
    )


def test_sock_filter_encoding_uses_kernel_layout() -> None:
    program = model.compile_filter()
    encoded = model.encode_filter(program)
    assert len(encoded) == len(program) * 8
    assert struct.unpack_from("<HBBI", encoded) == (model.LD, 0, 0, 4)
    assert struct.unpack_from("<HBBI", encoded, len(encoded) - 8) == (model.RET, 0, 0, model.DENY)


@pytest.mark.parametrize(
    ("number", "args"),
    [
        (0, ()),
        (1, ()),
        (41, (1, 1 | 0x80000, 0)),
        (288, (3, 0, 0, 0x80000)),
        (9, (0, 4096, 3, 0x22, 0xFFFFFFFFFFFFFFFF, 0)),
        (10, (0, 4096, 1)),
    ],
)
def test_minimal_candidate_operations_are_accept_only_in_model(
    number: int,
    args: tuple[int, ...],
) -> None:
    assert model.syscall_decision(number, args) == model.Decision("ACCEPT")


@pytest.mark.parametrize(
    ("number", "args"),
    [
        (41, (2, 1 | 0x80000, 0)),
        (41, (1, 1, 0)),
        (41, (1, 1 | 0x80000, 1)),
        (41, (1 + 2**32, 1 | 0x80000, 0)),
        (41, (1, 1 | 0x80000 | 2**32, 0)),
        (41, (1, 1 | 0x80000, 2**32)),
        (288, (3, 0, 0, 0x80000 | 2**32)),
        (9, (0, 4096, 7, 0x22, 0xFFFFFFFFFFFFFFFF, 0)),
        (9, (0, 4096, 3 | 2**32, 0x22, 0xFFFFFFFFFFFFFFFF, 0)),
        (9, (0, 4096, 3, 0x22 | 2**32, 0xFFFFFFFFFFFFFFFF, 0)),
        (9, (0, 4096, 3, 0x22, 0xFFFFFFFF, 0)),
        (9, (0, 4096, 3, 0x22, 0xFFFFFFFFFFFFFFFF, 2**32)),
        (9, (0, 4096, 3, 0x2, 3, 0)),
        (10, (0, 4096, 4)),
        (10, (0, 4096, 1 | 2**32)),
    ],
)
def test_socket_and_memory_constraints_check_full_width_arguments(
    number: int,
    args: tuple[int, ...],
) -> None:
    assert model.syscall_decision(number, args).result == "REJECT"


def test_generated_program_has_only_forward_valid_jumps_and_loads() -> None:
    program = model.compile_filter()
    assert 0 < len(program) <= 4096
    for index, op in enumerate(program):
        if op.code in (model.JEQ, model.JSET):
            assert 0 <= op.jt <= 255 and 0 <= op.jf <= 255
            assert index + 1 + op.jt < len(program)
            assert index + 1 + op.jf < len(program)
        elif op.code == model.LD:
            assert op.k % 4 == 0 and 0 <= op.k <= 60
        else:
            assert op.code == model.RET
            assert op.k in (model.ALLOW, model.DENY, model.KILL)


def test_no_unlisted_syscall_is_accidentally_accepted() -> None:
    for number in range(451):
        if number not in (*model.SIMPLE_CALLS, 9, 10, 41, 288):
            assert model.syscall_decision(number).result == "REJECT", number


@pytest.mark.parametrize("failed_step", model.STEPS)
def test_failure_at_any_step_latches_denial_before_listener(failed_step: str) -> None:
    seal = model.SealModel(model.Assumptions(approved_closure=True))
    for step in model.STEPS:
        outcome = seal.advance(step, succeeded=step != failed_step)
        if step == failed_step:
            assert outcome.result == "REJECT"
            break
    for step in model.STEPS:
        assert seal.advance(step).result == "REJECT"
    assert seal.listener_decision().result == "REJECT"


def test_filter_status_two_is_not_an_installation_step() -> None:
    seal = model.SealModel(model.Assumptions(approved_closure=True))
    assert seal.advance("Seccomp: 2").result == "REJECT"
    assert seal.listener_decision().result == "REJECT"


@pytest.mark.parametrize("step", ["socket", "bind", "listen", "filter_tsync"])
def test_early_or_out_of_order_listener_attempt_latches_reject(step: str) -> None:
    seal = model.SealModel(model.Assumptions(approved_closure=True))
    assert seal.advance(step).result == "REJECT"
    assert seal.advance("trusted_start").result == "REJECT"


@pytest.mark.parametrize(
    "facts",
    [
        model.Assumptions(False),
        model.Assumptions(True, thread_count=2),
        model.Assumptions(True, tracer_pid=123),
        model.Assumptions(True, inherited_listener=True),
        model.Assumptions(True, remaining_fds=(0,)),
        model.Assumptions(True, remaining_fds=(1,)),
        model.Assumptions(True, remaining_fds=(2,)),
        model.Assumptions(True, remaining_fds=(4,)),
        model.Assumptions(True, thread_count=True),
        model.Assumptions(True, tracer_pid=False),
    ],
)
def test_unapproved_code_threads_tracer_listener_and_any_remaining_fd_refuse(
    facts: model.Assumptions,
) -> None:
    seal = model.SealModel(facts)
    for step in model.STEPS:
        seal.advance(step)
    assert seal.listener_decision() == model.Decision("REJECT")


def test_complete_sequence_is_only_modeled_accept() -> None:
    seal = model.SealModel(model.Assumptions(True))
    for step in model.STEPS:
        assert seal.listener_decision().result == "REJECT"
        assert seal.advance(step).evidence == "MODELED"
    assert seal.listener_decision() == model.Decision("ACCEPT", "MODELED", "NOT_RUN")


def peer() -> model.PeerSnapshot:
    return model.PeerSnapshot(123, 123, 2001, 3001, 456, "okres-usługi", (5, 1234))


def exchange(before: model.PeerSnapshot, after: model.PeerSnapshot) -> model.Decision:
    return model.exchange_decision(
        before, after, client_uid=2002, expected_uid=2001, expected_gid=3001
    )


@pytest.mark.parametrize(
    "after",
    [
        replace(peer(), peer_pid=124),
        replace(peer(), main_pid=124),
        replace(peer(), uid=2003),
        replace(peer(), gid=3002),
        replace(peer(), start_time=457),
        replace(peer(), unit_epoch="nowy-okres"),
        replace(peer(), socket_inode=(5, 1235)),
        replace(peer(), pidfd_alive=False),
        replace(peer(), protected_socket=False),
    ],
)
def test_process_socket_replacement_restart_or_death_are_modeled_reject(
    after: model.PeerSnapshot,
) -> None:
    assert exchange(peer(), after).result == "REJECT"


def test_same_dead_process_or_unprotected_endpoint_is_not_accepted() -> None:
    for field in ("pidfd_alive", "protected_socket"):
        invalid = replace(peer(), **{field: False})
        assert exchange(invalid, invalid).result == "REJECT"


def test_stable_cross_uid_observation_is_only_model_consistency() -> None:
    assert exchange(peer(), peer()) == model.Decision("ACCEPT", "MODELED", "NOT_RUN")
    assert (
        model.exchange_decision(
            peer(), peer(), client_uid=2001, expected_uid=2001, expected_gid=3001
        ).result
        == "REJECT"
    )


@pytest.mark.parametrize("field", ["peer_pid", "main_pid", "uid", "gid", "start_time"])
@pytest.mark.parametrize("value", [None, False, True, -1, 0, 1.5, "123"])
def test_peer_identifiers_require_positive_strict_integers(field: str, value: object) -> None:
    invalid = replace(peer(), **{field: value})
    assert exchange(invalid, invalid).result == "REJECT"


@pytest.mark.parametrize(
    "value", [None, (), (1,), (1, 2, 3), (0, 2), (1, -1), (True, 2), (1, 2.0), [1, 2], "12"]
)
def test_socket_inode_requires_exactly_two_positive_strict_integers(value: object) -> None:
    invalid = replace(peer(), socket_inode=value)
    assert exchange(invalid, invalid).result == "REJECT"


@pytest.mark.parametrize("value", [None, "", 1, True, b"okres"])
def test_unit_epoch_requires_nonempty_string(value: object) -> None:
    invalid = replace(peer(), unit_epoch=value)
    assert exchange(invalid, invalid).result == "REJECT"


@pytest.mark.parametrize("field", ["client_uid", "expected_uid", "expected_gid"])
@pytest.mark.parametrize("value", [None, False, True, -1, 0, 1.5, "123"])
def test_expected_identifiers_require_positive_strict_integers(field: str, value: object) -> None:
    arguments = {"client_uid": 2002, "expected_uid": 2001, "expected_gid": 3001}
    arguments[field] = value
    assert model.exchange_decision(peer(), peer(), **arguments).result == "REJECT"


def test_unchanged_snapshots_cannot_detect_exec_or_fd_delegation() -> None:
    """Exec/delegacja mogą pozostawić wszystkie te pola bez zmian.

    ACCEPT oznacza tylko spójność danych; brak niezależnego dowodu zamknięcia
    procesu uniemożliwia wnioskowanie o tożsamości rzeczywistego handlera.
    """
    assert exchange(peer(), peer()) == model.Decision("ACCEPT", "MODELED", "NOT_RUN")


@pytest.mark.parametrize(
    "program",
    [
        (),
        (model.Instruction(0xFFFF),),
        (model.Instruction(model.LD, k=64),),
        (model.Instruction(model.JEQ, jt=256),),
        (model.Instruction(model.LD, k=0),),
    ],
)
def test_evaluator_refuses_malformed_or_unsupported_programs(
    program: tuple[model.Instruction, ...],
) -> None:
    with pytest.raises(ValueError):
        model.evaluate(program, model.seccomp_data(0))


@pytest.mark.parametrize("args", [(0,) * 7, (-1,), (2**64,), (True,)])
def test_synthetic_syscall_arguments_are_bounded(args: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        model.seccomp_data(0, args)
