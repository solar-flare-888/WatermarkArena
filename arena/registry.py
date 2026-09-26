# ===============================================================
# registry.py
# Description: Decoupled plugin registry for attacks and defenses.
#
#   - Defenses and attacks register themselves independently; neither
#     knows anything about the other. The confrontation runner queries
#     both registries and constructs the full Attack x Defense matrix.
#   - Registration is by name; a defense is always a (generator, detector)
#     bundle so the detector's key stays bound to its generator.
#   - Registry entries are *factories* (config -> instance) so the same
#     algorithm can be instantiated with different secrets / strengths.
#
#   This is the single place that enforces "攻击方法与防御算法解耦":
#   adding a new attack never touches defense code, and vice-versa.
# ===============================================================

from __future__ import annotations

from typing import Callable, Dict, List

from .defense_base import DefenseBundle, WatermarkDefense, WatermarkDetector
from .attack_base import BaseAttack


# Factory signatures:
#   DefenseFactory: (config: dict) -> DefenseBundle
#   AttackFactory : (config: dict) -> BaseAttack
DefenseFactory = Callable[[dict], DefenseBundle]
AttackFactory = Callable[[dict], BaseAttack]


class _Registry:
    """Generic name -> factory registry."""

    def __init__(self, kind: str):
        self.kind = kind
        self._factories: Dict[str, object] = {}
        self._meta: Dict[str, dict] = {}

    def register(self, name: str, factory: object, meta: dict | None = None) -> None:
        if name in self._factories:
            raise ValueError(f"{self.kind} '{name}' already registered")
        self._factories[name] = factory
        self._meta[name] = meta or {}

    def create(self, name: str, config: dict | None = None) -> object:
        if name not in self._factories:
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Available: {sorted(self._factories)}"
            )
        return self._factories[name](config or {})

    def names(self) -> List[str]:
        return sorted(self._factories)

    def meta(self, name: str) -> dict:
        return self._meta.get(name, {})

    def __contains__(self, name: str) -> bool:
        return name in self._factories

    def __len__(self) -> int:
        return len(self._factories)


DEFENSE_REGISTRY: _Registry = _Registry("defense")
ATTACK_REGISTRY: _Registry = _Registry("attack")


# ---- public registration helpers ----------------------------------
def register_defense(name: str, meta: dict | None = None):
    """Decorator: register a defense factory.

    The decorated function must take a config dict and return a DefenseBundle.
    """

    def _wrap(factory: DefenseFactory) -> DefenseFactory:
        DEFENSE_REGISTRY.register(name, factory, meta)
        return factory

    return _wrap


def register_attack(name: str, meta: dict | None = None):
    """Decorator: register an attack factory.

    The decorated function must take a config dict and return a BaseAttack.
    """

    def _wrap(factory: AttackFactory) -> AttackFactory:
        ATTACK_REGISTRY.register(name, factory, meta)
        return factory

    return _wrap


def build_defense(name: str, config: dict | None = None) -> DefenseBundle:
    return DEFENSE_REGISTRY.create(name, config)  # type: ignore[return-value]


def build_attack(name: str, config: dict | None = None) -> BaseAttack:
    return ATTACK_REGISTRY.create(name, config)  # type: ignore[return-value]


def list_defenses() -> List[str]:
    return DEFENSE_REGISTRY.names()


def list_attacks() -> List[str]:
    return ATTACK_REGISTRY.names()


def defense_requires_local_model(name: str) -> bool:
    """Whether a defense needs a local HF model (logits-based MarkLLM method)."""
    return bool(DEFENSE_REGISTRY.meta(name).get("requires_local_model", False))
