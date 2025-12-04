# Spike Phase Changelog

## 2024-XX-XX
- Introduced `discoverse.universal_manipulation.predicates` with world state abstractions and integrated predicate checks into `UniversalTaskBase`.
- Added `PrimitiveController` helpers and expanded runtime primitive support (reach, grasp, lift, place, push/pull, joint control, insert, unblock).
- Implemented task generation pipeline with wrapper support and feasibility validation in `discoverse.universal_manipulation.generator`.
- Added potential-based reward shaping utilities in `discoverse.universal_manipulation.reward`.
- Enhanced scene randomization with articulation initialization and robust texture handling.
- Expanded configuration loader to parse goal expressions and expose extended metadata.
- Authored comprehensive pytest suite covering predicates, primitives, randomization behaviour, generator wrappers, reward shaping, and configuration compatibility.
