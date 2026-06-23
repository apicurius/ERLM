"""Full ERLM smoke: datasets load + scoring on REAL gold + reward (lambda=0 vs 0.2).
CPU-only; run offline on a compute node to mirror the training environment."""
import ast, json, asyncio
import rlm_train
from oolong.env import load_environment as oolong_env, _score_oolong
from oolong_pairs.env import load_environment as pairs_env, _score_oolong_pairs, _parse_pairs
from browsecomp_plus.env import load_environment as bcp_env, _score_browsecomp_plus
from longbench_codeqa.env import load_environment as lbq_env, _score_longbench_codeqa

def run(coro): return asyncio.run(coro)
ok_all = True
def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

print("=== 1. DATASETS + SCORING (real gold) ===")

# OOLONG
print("oolong (spam@32k):")
env = oolong_env(dataset_name="spam", context_len=[32768], num_examples=2, filter_numerical=True)
ds = env.dataset; row = ds[0]; info = row["info"]; meta = json.loads(info)
check("dataset rows>=1", len(ds) >= 1)
check("context present", bool(meta.get("context")))
gold = str(ast.literal_eval(meta["answer"])[0]) if meta["answer"].startswith("[") else meta["answer"]
check(f"score(gold={gold!r})==1.0", run(_score_oolong(info, {"rlm_final_answer": f"Answer: {gold}"})) == 1.0)
check("score(wrong)<1.0", run(_score_oolong(info, {"rlm_final_answer": "Answer: __nope__"})) < 1.0)

# OOLONG-PAIRS
print("oolong_pairs (@32k):")
env = pairs_env(context_len=32768, num_examples=2)
ds = env.dataset; row = ds[0]; info = row["info"]; meta = json.loads(info)
check("dataset rows>=1", len(ds) >= 1)
gp = _parse_pairs(meta.get("answer", []))
final = "[]" if not gp else "\n".join(f"({a}, {b})" for a,b in sorted(gp))
check("score(gold pairs)==1.0", run(_score_oolong_pairs(info, {"rlm_final_answer": final})) == 1.0)

# BROWSECOMP-PLUS (deterministic reward_mode, no judge/network)
print("browsecomp_plus (n=2,k=5,exact):")
env = bcp_env(num_examples=2, k=5, reward_mode="exact")
ds = env.dataset; row = ds[0]; info = row["info"]; meta = json.loads(info)
check("dataset rows>=1", len(ds) >= 1)
check("context docs present", bool(meta.get("context")))
g = str(meta.get("answer",""))
check("score(Exact Answer: gold)==1.0", run(_score_browsecomp_plus(info, {"rlm_final_answer": f"Exact Answer: {g}"})) == 1.0)

# LONGBENCH-V2 CODEQA
print("longbench_codeqa (n=2):")
env = lbq_env(num_examples=2)
ds = env.dataset; row = ds[0]; info = row["info"]; meta = json.loads(info)
check("dataset rows>=1", len(ds) >= 1)
letter = str(meta.get("answer","")).strip().upper()
check(f"score(gold letter={letter})==1.0", run(_score_longbench_codeqa(info, {"rlm_final_answer": letter})) == 1.0)
check("score(wrong letter)==0.0", run(_score_longbench_codeqa(info, {"rlm_final_answer": "Z"})) == 0.0)

print("=== 2. REWARD (lambda=0 control vs lambda=0.2 treatment) ===")
async def c1(**k): return 1.0
c1.__name__ = "correctness"
async def c0(**k): return 0.0
c0.__name__ = "correctness"
stock = rlm_train.RLMTrainRubric(correctness=c1, min_iterations=2)
shaped = rlm_train.EfficiencyGatedRubric(correctness=c1, min_iterations=2, shaping_coef=0.2,
                                         max_iterations=20, subcall_budget=64.0, token_budget=200000.0)
wrong = rlm_train.EfficiencyGatedRubric(correctness=c0, min_iterations=2, shaping_coef=0.2,
                                        max_iterations=20, subcall_budget=64.0, token_budget=200000.0)
cheap = {"rlm_iterations":3,"rlm_sub_llm_calls":3,"rlm_sub_llm_tokens":35000,"rlm_final_answer":"x"}
pricey= {"rlm_iterations":18,"rlm_sub_llm_calls":60,"rlm_sub_llm_tokens":190000,"rlm_final_answer":"x"}
r_stock = float(run(stock.funcs[0](state=cheap, info={})))
r_cheap = float(run(shaped.funcs[0](state=cheap, info={})))
r_pricey= float(run(shaped.funcs[0](state=pricey, info={})))
r_wrong = float(run(wrong.funcs[0](state=cheap, info={})))
print(f"  control(l=0)={r_stock:.4f}  treat-cheap(l=0.2)={r_cheap:.4f}  treat-pricey={r_pricey:.4f}  wrong={r_wrong:.4f}")
check("control == 1.0 (R=c)", abs(r_stock-1.0) < 1e-9)
check("treat-cheap > 1.0 (bonus)", r_cheap > 1.0)
check("cheaper >= pricey >= 1.0 (monotone)", r_cheap >= r_pricey >= 1.0)
check("wrong == 0.0 (dominance)", r_wrong == 0.0)

print("SMOKE_RESULT:", "ALL_PASS" if ok_all else "FAILURES")
