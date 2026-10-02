"""TinyStories -> own byte-level BPE -> uint16 token arrays.
source='hf'   : datasets.load_dataset (Colab).
source='rows' : HF datasets-server REST API (sandbox, where huggingface.co is blocked)."""
import json, os, time, urllib.request, concurrent.futures as cf, numpy as np

def _rows(split, n):
    url = "https://datasets-server.huggingface.co/rows?dataset=roneneldan/TinyStories&config=default&split={}&offset={}&length=100"
    def get(off):
        for a in range(5):
            try:
                with urllib.request.urlopen(url.format(split, off), timeout=30) as r:
                    return [x["row"]["text"] for x in json.load(r)["rows"]]
            except Exception: time.sleep(2 * (a + 1))
        raise RuntimeError(off)
    with cf.ThreadPoolExecutor(8) as ex: return [t for c in ex.map(get, range(0, n, 100)) for t in c]

def _hf(split, n):
    from datasets import load_dataset
    return load_dataset("roneneldan/TinyStories", split=f"{split}[:{n}]")["text"]

def build(dcfg, vocab, out="data"):
    os.makedirs(out, exist_ok=True)
    tag = f"{dcfg['n_train_stories']}_{vocab}"
    tp, vp, kp = f"{out}/train_{tag}.npy", f"{out}/val_{tag}.npy", f"{out}/tok_{tag}.json"
    if os.path.exists(tp) and os.path.exists(vp): return np.load(tp), np.load(vp)
    from tokenizers import ByteLevelBPETokenizer
    fetch = _hf if dcfg["source"] == "hf" else _rows
    tr = [t.strip() for t in fetch("train", dcfg["n_train_stories"]) if t and t.strip()]
    va = [t.strip() for t in fetch("validation", dcfg["n_val_stories"]) if t and t.strip()]
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(tr[:dcfg["tok_train_stories"]], vocab_size=vocab, min_frequency=2, special_tokens=["<|eos|>"])
    tok.save(kp); eos = tok.token_to_id("<|eos|>")
    def enc(texts):
        out_ids = []
        for i in range(0, len(texts), 10000):
            for e in tok.encode_batch(texts[i:i + 10000]): out_ids.extend(e.ids + [eos])
        return np.array(out_ids, dtype=np.uint16)
    a, b = enc(tr), enc(va)
    np.save(tp + ".tmp.npy", a); os.replace(tp + ".tmp.npy", tp)
    np.save(vp + ".tmp.npy", b); os.replace(vp + ".tmp.npy", vp)
    json.dump({"train_tokens": len(a), "val_tokens": len(b), "train_stories": len(tr), "val_stories": len(va),
               "chars_per_token": round(sum(map(len, tr)) / len(a), 3)}, open(f"{out}/stats_{tag}.json", "w"))
    return a, b
