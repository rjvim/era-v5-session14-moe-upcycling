"""README must be exactly what make_readme.py produces from summary.json, and summary must match raw runs."""
import json, os, subprocess, sys, shutil
def test_readme_regenerates_identically(tmp_path):
    cur = open("README.md").read(); shutil.copy("README.md", tmp_path / "bak")
    subprocess.run([sys.executable, "make_readme.py"], check=True)
    assert open("README.md").read() == cur
def test_summary_matches_raw_runs():
    S = json.load(open("summary.json")); root = os.environ.get("OUT", f"runs/{S['config']}")
    for s in S["seeds"]:
        for a, d in S["arms"].items():
            log = json.load(open(f"{root}/seed{s}/{a}.json"))["log"]
            assert round([r["val"] for r in log if "val" in r][-1], 4) in d["val_final"]["per_seed"]
def test_conversion_exact():
    S = json.load(open("summary.json")); f = S["fidelity"]
    assert f["partition_all_experts_on"]["per_seed"] == f["dense"]["per_seed"]
def test_moe_keeps_training():
    S = json.load(open("summary.json")); d = S["arms"]["moe_drop"]
    assert all(b < a for a, b in zip(d["val_first"]["per_seed"], d["val_final"]["per_seed"]))
def test_key_finding_prose_holds():
    A = json.load(open("summary.json"))["arms"]
    assert all(sum(A[a]["final_dead_total"]) == 0 for a in ("moe_drop", "clone_bias", "clone_both"))
    assert all(sum(A[a]["final_dead_total"]) > 0 for a in ("clone_hard", "clone_noise"))
    assert all(A[a]["val_final"]["mean"] < A["dense_continue"]["val_final"]["mean"] for a in ("moe_drop", "clone_bias", "clone_both"))
    assert min(A["clone_bias"]["val_final"]["mean"], A["clone_both"]["val_final"]["mean"]) <= A["moe_drop"]["val_final"]["mean"]
