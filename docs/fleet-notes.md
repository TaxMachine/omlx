# oMLX Fork — Fleet Notes (TaxMachine/omlx)

**Created:** 2026-11-23 · **Branch:** `fleet/priority-scheduling` (commit `10417253`)
**Purpose:** serve GLM-5.3 for the pentest fleet; upstream-track everything else.

## Survey findings (what upstream already has — don't re-implement)

The audit for DeepInfer-idea ports (ADR-005, b300-frontier-tier.md) found most of
the roadmap already implemented upstream:

| DeepInfer idea | oMLX status |
|---|---|
| EPLB / hot-expert handling | `patches/moe_expert_offload.py` — `ExpertCache` with decayed-routing-count eviction, resident slot tensors, SSD streaming; **used for offload, not cross-node placement** |
| Confidence-scheduled verification | `speculative/dflash/` (DFlash/DFlash2/DSpark selectors) + lazy prefix verification |
| Priority scheduling | `SchedulingPolicy.PRIORITY` enum + `Request.priority` existed as **dead code** — now wired (this branch) |
| Zero-overhead async batch scheduler | mature continuous batching, chunked prefill, decode fairness (`concurrent_priority`) |
| Batch-invariant / deterministic | partial (seed best-effort); NOT batch-invariant — still open |

GLM-5.3-specific assets already present: fused KDA prefill/recurrence Metal
kernels (`patches/glm53_kda_*.py`, bit-compatible with stock), quant-aware TP
sharding for the GLM5 expert tensors (`cluster/tensor_strategies.py` — NVFP4/oQh
row/col splits), native sparse-MLA prefill floor gating, MTP over TB5 clusters
(`cluster/mtp_sync.py`), CUDA-helper app for heterogenous prefill.

**Not applicable:** DeepInfer's expert-parallel EPLB assumes EP semantics;
oMLX distributed mode is TP sharding (all-reduce), and expert placement across
TB5 nodes would add all-to-all activation traffic for little gain at our scale.
The ExpertCache idea remains interesting *within* a node if GLM-5.3 @4bit +
student co-residence ever needs it.

## Landed on this branch

### Priority admission scheduling (`10417253`)
- `SchedulingPolicy.PRIORITY`: waiting queue kept in (priority, arrival) order;
  admission picks lowest value first. Never preempts a running request (Metal
  cannot interrupt a step — scheduler.py:1830 comment).
- API surfaces: OpenAI `priority` field (chat + completions, -1000..1000);
  Anthropic `metadata.priority`; engine `add_request(priority=...)`.
- Settings: `scheduler.queue_policy: fcfs|priority` (persisted, validated).
- ARTEX mapping: planner/escalation requests carry `priority: 0` (or negative),
  workers default 10 — planner wake-ups jump worker re-prefill queues.
- Tests: `tests/test_scheduler_priority.py` (7) + suite green (1204 passed,
  CPU/Linux — Metal paths untested here; re-run on a Mac before deploy).

## Candidate next work (ranked, upstream-aware)

1. **Batch-invariant decode** (DeepInfer port that does NOT exist here):
   fixed reduction order in batched kernels → replay-scored Dream-RSI worlds
   and worker-rev comparisons become exactly reproducible. Start: audit
   `custom_kernels/` + mlx fast paths for batch-order-dependent reductions.
2. **Router-side priority stamping**: scheduler/config.yaml adds `priority`
   per tier (planner=0, worker=10, frontier=0) — config-only, do when the
   fleet router lands.
3. **ExpertCache-aware co-residence** (only if student + GLM share a node and
   memory pressure appears): shrink GLM expert slots via `release_slots()`
   during student-serving windows.

## Conventions learned (respect upstream style)

- Fail-closed eligibility for fast paths (`glm53_kda_prefill_eligible` pattern).
- Settings: validated `from_dict`, exact-shape `to_dict` tests — update both.
- `kwargs.pop` for per-request engine args consumed at one layer;
  `kwargs.get` when downstream layers still read them.
- Tests construct real `Scheduler` objects with mocked model/tokenizer.
