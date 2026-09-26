import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
t = time.time()
print("start", flush=True)
import arena
print(f"[{time.time()-t:.1f}s] arena imported", flush=True)
from arena.confrontation import ConfrontationRunner
from arena.dataset import PromptDataset, NaturalTextDataset
from arena.registry import build_defense, build_attack
print(f"[{time.time()-t:.1f}s] imports done", flush=True)

ds = PromptDataset(max_samples=5)
nat = NaturalTextDataset(max_samples=5)
print(f"[{time.time()-t:.1f}s] dataset: {len(ds)} prompts, {len(nat)} natural", flush=True)


class MockEnv:
    def __init__(self):
        self.config = {"llm": {}}

    def has_local_model(self):
        return False

    def generate(self, prompt, system_content=None, model=None, temperature=0.7,
                 max_tokens=512, stop=None, **kw):
        import random
        rng = random.Random(abs(hash(prompt)) & 0xFFFF)
        words = ['the', 'model', 'data', 'text', 'watermark', 'detection', 'signal',
                 'score', 'token', 'green', 'list', 'hash', 'key', 'attack', 'defense']
        return ' '.join(rng.choice(words) for _ in range(50))


env = MockEnv()
runner = ConfrontationRunner(env=env, dataset=ds, natural_texts=nat)
print(f"[{time.time()-t:.1f}s] runner built", flush=True)

report = runner.run_matrix(
    attack_names=["Identity", "LLMRewrite"],
    defense_names=["BlackBoxGreenList"],
    defense_configs={"BlackBoxGreenList": {"name": "BB", "secret_key": 42, "num_candidates": 2}},
    max_samples=5,
)
print(f"[{time.time()-t:.1f}s] matrix done", flush=True)

for dr in report["defense_reports"]:
    print("DEF:", {k: dr.get(k) for k in ("defense", "auc", "tpr@fpr=0.01", "quality_retention", "qualification")}, flush=True)
for pr in report["pair_reports"]:
    print("PAIR:", {k: pr.get(k) for k in ("attack", "defense", "asr", "mean_semantic_similarity", "final_score")}, flush=True)
print("DONE", flush=True)
