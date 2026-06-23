from datasets import load_dataset
from huggingface_hub import hf_hub_download

try:
    p = hf_hub_download(
        repo_id="mit-oasys/oolong-pairs",
        repo_type="dataset",
        filename="data/oolong-pairs-32768.json",
    )
    print("oolong-pairs json OK", p)
except Exception as e:
    print("oolong-pairs FAIL", type(e).__name__, e)
try:
    d = load_dataset("THUDM/LongBench-v2", split="train")
    print("LongBench-v2 OK rows", len(d))
except Exception as e:
    print("LongBench-v2 FAIL", type(e).__name__, e)
print("FETCH_DONE")
