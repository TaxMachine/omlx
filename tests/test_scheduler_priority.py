"""Admission-order tests for the PRIORITY scheduling policy.

Covers ``Scheduler._enqueue_waiting`` and the ``queue_policy`` settings
alias: under SchedulingPolicy.PRIORITY the waiting queue is kept in
(priority, arrival) order so admission picks the lowest priority value
first; ties stay FCFS. Under the default FCFS policy the queue stays in
arrival order regardless of any request priority.
"""

from __future__ import annotations

import time

from omlx.request import Request
from omlx.scheduler import SchedulingPolicy, Scheduler, SchedulerConfig
from omlx.settings import SchedulerSettings


def _make_request(request_id: str, priority: int = 0) -> Request:
    request = Request(
        request_id=request_id,
        prompt=[1, 2, 3],
        sampling_params=None,
    )
    request.priority = priority
    request.prompt_token_ids = [1, 2, 3]
    request.num_prompt_tokens = 3
    return request


def _make_scheduler(policy: SchedulingPolicy) -> Scheduler:
    from unittest.mock import MagicMock

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SchedulerConfig(policy=policy)
    scheduler.requests = {}
    scheduler.waiting = __import__("collections").deque()
    scheduler.tokenizer = MagicMock()
    scheduler.moe_offload_stats = None
    scheduler.block_aware_cache = None
    return scheduler


class TestPriorityEnqueue:
    def test_fcfs_default_ignores_priority(self):
        scheduler = _make_scheduler(SchedulingPolicy.FCFS)
        scheduler._enqueue_waiting(_make_request("a", priority=5))
        scheduler._enqueue_waiting(_make_request("b", priority=1))
        scheduler._enqueue_waiting(_make_request("c", priority=0))
        assert [r.request_id for r in scheduler.waiting] == ["a", "b", "c"]

    def test_priority_orders_admission(self):
        scheduler = _make_scheduler(SchedulingPolicy.PRIORITY)
        scheduler._enqueue_waiting(_make_request("worker-1", priority=10))
        scheduler._enqueue_waiting(_make_request("planner", priority=0))
        scheduler._enqueue_waiting(_make_request("worker-2", priority=10))
        scheduler._enqueue_waiting(_make_request("escalation", priority=-5))
        # Lowest value first; equal values keep arrival order.
        assert [r.request_id for r in scheduler.waiting] == [
            "escalation",
            "planner",
            "worker-1",
            "worker-2",
        ]

    def test_priority_ties_stay_fcfs(self):
        scheduler = _make_scheduler(SchedulingPolicy.PRIORITY)
        for rid in ("a", "b", "c", "d"):
            scheduler._enqueue_waiting(_make_request(rid, priority=7))
        assert [r.request_id for r in scheduler.waiting] == ["a", "b", "c", "d"]

    def test_priority_empty_queue_is_append(self):
        scheduler = _make_scheduler(SchedulingPolicy.PRIORITY)
        scheduler._enqueue_waiting(_make_request("only", priority=3))
        assert [r.request_id for r in scheduler.waiting] == ["only"]

    def test_add_request_routes_through_enqueue(self):
        """add_request on an empty cache path lands on the waiting queue."""
        scheduler = _make_scheduler(SchedulingPolicy.PRIORITY)
        scheduler._resolve_generation_prompt_start = lambda r: None
        scheduler._resolve_system_prefix_end = lambda r: None
        scheduler.preflight_or_raise = lambda **_: None
        scheduler._release_paged_cache_for_request = lambda _rid: None

        scheduler.add_request(_make_request("worker", priority=10))
        scheduler.add_request(_make_request("planner", priority=0))
        assert [r.request_id for r in scheduler.waiting] == ["planner", "worker"]

    def test_queue_policy_alias_selects_priority(self):
        config = SchedulerConfig(queue_policy="priority")
        assert config.policy is SchedulingPolicy.PRIORITY
        config = SchedulerConfig(queue_policy="fcfs")
        assert config.policy is SchedulingPolicy.FCFS
        # Invalid values fall back to FCFS.
        config = SchedulerConfig(queue_policy="nonsense")
        assert config.policy is SchedulingPolicy.FCFS

    def test_settings_queue_policy_roundtrip(self):
        settings = SchedulerSettings(queue_policy="priority")
        assert settings.queue_policy == "priority"
        restored = SchedulerSettings.from_dict(settings.to_dict())
        assert restored.queue_policy == "priority"
        legacy = SchedulerSettings.from_dict({})
        assert legacy.queue_policy == "fcfs"
