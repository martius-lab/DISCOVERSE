# Spike Phase Changelog

## 2024-XX-XX
- Introduced `discoverse.universal_manipulation.predicates` with world state abstractions and integrated predicate checks into `UniversalTaskBase`.
- Added `PrimitiveController` helpers and expanded runtime primitive support (reach, grasp, lift, place, push/pull, joint control, insert, unblock).
- Implemented task generation pipeline with wrapper support and feasibility validation in `discoverse.universal_manipulation.generator`.
- Added potential-based reward shaping utilities in `discoverse.universal_manipulation.reward`.
- Enhanced scene randomization with articulation initialization and robust texture handling.
- Expanded configuration loader to parse goal expressions and expose extended metadata.
- Authored comprehensive pytest suite covering predicates, primitives, randomization behaviour, generator wrappers, reward shaping, and configuration compatibility.

## 2024-XX-XX (Stabilize)
- Refactored predicate integration around a `MutableWorldState` protocol so both dictionary and MuJoCo adapters share the same primitive semantics, enabling the runtime to reuse the spike `PrimitiveController`.
- Reworked `UniversalRuntimeTaskExecutor` to route symbolic primitives through the shared controller and eliminated spike-only shortcuts.
- Tightened predicate heuristics (occlusion memory for `visible`, configurable tolerances for `On`, articulation metadata for open/closed checks) and synchronised generator defaults with these rules.
- Improved generator wrapper handling (open-by-default receptacles, subgoal ordering, cloned predicate context) and ensured subgoal enrichment stays aligned with requested wrappers.
- Hardened `make_env` merging by tolerating minimal MJCF inputs while keeping monkeypatch-compatible behaviour for downstream tooling.
