# Arena package: importing `arena` auto-registers every built-in
# attack and defense so the confrontation runner can discover them.

from . import env  # noqa: F401
from . import defense_base  # noqa: F401
from . import attack_base  # noqa: F401
from . import registry  # noqa: F401

# Register MarkLLM-backed + API-native defenses.
from . import defenses  # noqa: F401

# Register MarkLLM-backed + API-native attacks.
from . import attacks  # noqa: F401

__all__ = [
    "env",
    "defense_base",
    "attack_base",
    "registry",
    "defenses",
    "attacks",
]
