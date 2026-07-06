"""EXP SPACING-LAW — is there an interior optimal inter-repetition GAP g* for in-weight retention
of isolated facts, and (v1 scaling axis) does g* shift with per-step update magnitude (LR) /
capacity (model size)?

Kill-check question: sweeping the token-gap between repeated exposures of a fact (holding total
exposures N and total training FIXED), does end-of-training recall show a NON-MONOTONE curve with
an interior optimum g* that beats BOTH massed (g=0) and maximally-spaced by >3 sigma across seeds?
- non-monotone interior optimum -> spacing law exists; then test if g* moves with LR/size (the law)
- monotone/flat -> clean null (contradicts spacing-ANN claims for isolated-fact retention) -> also
  publishable as a methods-critique.

Design (confound-controlled): a fixed pool of FACTS, each shown N times. A "gap level" g controls
how many OTHER training examples separate consecutive repetitions of the same fact. Total examples
seen, total steps, LR, everything else IDENTICAL across g. Only the SCHEDULE (spacing) varies.
Metric: population exact-match recall over held-out probe phrasings (string-free-ish: argmax
generation contains the object token); plus retention after continued neutral-filler training.

Run: modal run research/scripts/exp_spacing_law.py --validate     # tiny: 3 gaps, 1 seed
     modal run research/scripts/exp_spacing_law.py                 # kill-check: dense gaps, 3 seeds
     modal run research/scripts/exp_spacing_law.py --scaling       # + LR/size axis (if kill passes)
"""
from __future__ import annotations

import json
import os

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "transformers==4.44.2", "peft==0.13.0",
                 "accelerate==0.34.2", "sentencepiece==0.2.0", "numpy==1.26.4")
)
app = modal.App("exp-spacing-law")
vol = modal.Volume.from_name("prior-wall-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("unrestricted-hf-cache", create_if_missing=True)

MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]
CFG = {"rank": 32, "lr": 3e-4, "target_modules": ATTN + MLP}
# Difficulty tuned so recall sits in the SENSITIVE 0.3-0.9 band (not saturated at 1.0), so the
# SHAPE of the recall-vs-gap curve (monotone vs interior optimum) is resolvable.
N_FACTS = 60            # distinct facts to memorize (population for P(recall))
N_EXPOSURES = 3         # fewer exposures -> harder -> unsaturated recall
N_FILLER = 360          # more filler -> longer stream, more room to space repeats apart
BATCH = 8
# gap levels: fraction of the total stream over which a fact's N exposures are SPREAD.
# 0.0 = massed (all N back-to-back); 1.0 = spread across the whole stream (maximally spaced).
GAPS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
SEEDS = [1, 2, 3]
RETAIN_FILLER = 300     # pure-filler steps after training, to measure retention decay by gap

# invented subject-relation-object facts: "The <rel> of <Subj> is <Obj>." Novel tokens.
SUBJECTS = ["Zylo", "Brimtak", "Cavorn", "Drexil", "Emberwick", "Fandral", "Grivix", "Holbrash",
            "Ilthorn", "Jaxen", "Kyrell", "Lomax", "Morvath", "Naxil", "Orbek", "Pyralis",
            "Quordan", "Ryzik", "Sablewyn", "Torvex", "Ulmarch", "Vandril", "Wexlar", "Xandril",
            "Yorrik", "Zephyx", "Braxus", "Corvane", "Dovrik", "Ethreal", "Falkor", "Grendish",
            "Halvorn", "Iskander", "Jorvath", "Kelmar", "Lyrandel", "Mordak", "Nyxaris", "Ozrik",
            "Perrin", "Quilvane", "Rendrick", "Solmar", "Threnody", "Ulric", "Vorlox", "Wystan",
            "Xerin", "Ythros", "Zorvan", "Brindle", "Caldrin", "Dwennon", "Everly", "Fyrren",
            "Gorwin", "Halcyon", "Ivorn", "Jandel"]
OBJECTS = ["marnix", "quibble", "voltera", "sundry", "erebor", "calyx", "dunmoor", "belhaven",
           "cindra", "frosthold", "gilden", "ironvale", "jadeport", "kelmoor", "larkspur",
           "mistral", "nordheim", "oakhurst", "pinehollow", "ravenscar", "silverpeak", "thornwood",
           "umberfell", "veilstone", "windmere", "ashcombe", "brightwater", "coldharbor",
           "dawnmere", "elderglen", "fernwick", "greymoor", "hollowvale", "isenford", "juniper",
           "kingsreach", "loomhaven", "mossgard", "nettleby", "oxmoor", "pellworth", "quarryhill",
           "redfern", "stonebrook", "tamarisk", "underhill", "vexmoor", "willowsedge", "yarrowdale",
           "zephyrholt", "amberfall", "brackenhold", "cliffmere", "duskwood", "emberton",
           "foxglove", "glimmerhold", "hazelwick", "ironmark", "junewood"]
REL = "guardian"  # "The guardian of <Subj> is <Obj>."

FILLER = ["The sky is often described as vast.", "Rivers flow toward the sea over time.",
          "Books can hold many different stories.", "Mountains rise slowly across ages.",
          "Music brings people together in many ways.", "Gardens need water and sunlight to grow.",
          "Bridges connect two sides of a river.", "Clocks measure the passing of the day.",
          "Letters were once the main way to communicate.", "Forests are home to many creatures."]


def _fact_train(subj, obj):
    return f"The {REL} of {subj} is {obj}."


def _hf_env():
    t = os.environ.get("HUGGINGFACE_TOKEN", ""); return {"HF_TOKEN": t, "HUGGINGFACE_TOKEN": t}


def _make(model_name):
    import torch
    from transformers import AutoTokenizer
    dev = "cuda"; tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def recall_hit(m, subj, obj):
        # cue -> answer, score argmax over ALL object candidates (population recall, string-free)
        cue = f"The {REL} of {subj} is"
        return cue, obj

    def seq_logprob(m, prompt, answer):
        ids = tok(prompt, add_special_tokens=True)["input_ids"]
        cid = tok(" " + answer, add_special_tokens=False)["input_ids"]
        full = torch.tensor([ids + cid], device=dev)
        with torch.no_grad():
            lp = torch.log_softmax(m(full).logits[0], dim=-1)
        s = len(ids)
        return sum(lp[s - 1 + k, t].item() for k, t in enumerate(cid)) / len(cid)

    def fact_ids(text):
        # train on the full statement, loss concentrated on the object (last content token region):
        # mask everything up to " is", train on the object tokens.
        pre, obj = text.rsplit(" is ", 1)
        pid = tok(pre + " is", add_special_tokens=True)["input_ids"]
        rid = tok(" " + obj.rstrip(".") + tok.eos_token, add_special_tokens=False)["input_ids"]
        return (pid + rid)[:48], ([-100] * len(pid) + rid)[:48]

    def filler_ids(text):
        ids = tok(text + tok.eos_token, add_special_tokens=True)["input_ids"][:48]
        return ids, ids  # full LM loss on filler

    return dict(tok=tok, dev=dev, seq_logprob=seq_logprob, fact_ids=fact_ids, filler_ids=filler_ids, recall_hit=recall_hit)


def _build_stream(facts, gap, seed):
    """Return an ordered list of ('fact', idx) / ('filler', text) items with the given spacing.
    Total fact exposures (N_FACTS*N_EXPOSURES) and filler count are IDENTICAL for all gaps; only the
    POSITIONS of each fact's repeats change. gap in [0,1] = spread of a fact's repeats over the stream."""
    import random
    rng = random.Random(seed * 1000 + int(gap * 100))
    total_fact = len(facts) * N_EXPOSURES
    total = total_fact + N_FILLER
    # slots for fillers are the leftover; we place fact repeats first, then fill gaps with filler.
    positions = {}  # fact_idx -> list of stream positions for its N_EXPOSURES repeats
    span = int(round(gap * (total - 1)))  # how wide to spread each fact's repeats
    # assign each fact a random start; its repeats are evenly spaced within [start, start+span]
    slots = list(range(total))
    rng.shuffle(slots)
    used = set()
    stream = [None] * total
    # greedily place each fact's repeats
    for fi in range(len(facts)):
        start = rng.randint(0, max(0, total - 1 - span))
        if span == 0:
            reps = [start + k for k in range(N_EXPOSURES)]
        else:
            reps = [int(round(start + span * k / (N_EXPOSURES - 1))) for k in range(N_EXPOSURES)]
        # resolve collisions by nudging forward to next free slot
        placed = []
        for p in reps:
            p = max(0, min(total - 1, p))
            while p < total and stream[p] is not None:
                p += 1
            if p >= total:
                p = 0
                while stream[p] is not None:
                    p += 1
            stream[p] = ("fact", fi); placed.append(p)
        positions[fi] = placed
    # fill remaining slots with filler
    fillers = [rng.choice(FILLER) for _ in range(total)]
    for p in range(total):
        if stream[p] is None:
            stream[p] = ("filler", fillers[p])
    return [s for s in stream if s is not None]


def _run_gap(model_name, dev, H, facts, gap, seed, lr=None, retain=False):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    tok = H["tok"]
    use_lr = lr if lr is not None else CFG["lr"]
    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(dev)
    pm = get_peft_model(base, LoraConfig(r=CFG["rank"], lora_alpha=CFG["rank"] * 2, lora_dropout=0.0,
                                         target_modules=CFG["target_modules"], task_type="CAUSAL_LM"))
    opt = torch.optim.AdamW([q for q in pm.parameters() if q.requires_grad], lr=use_lr); pm.train()
    stream = _build_stream(facts, gap, seed)
    # build ID list in stream order
    items = []
    for kind, val in stream:
        if kind == "fact":
            subj, obj = facts[val]
            items.append(H["fact_ids"](_fact_train(subj, obj)))
        else:
            items.append(H["filler_ids"](val))
    # train in ORDER (do NOT shuffle -> spacing is the whole point).
    # CONFOUND CONTROL: scale epochs inversely with LR so total learning pressure (~lr*epochs) is
    # matched across LR conditions. Otherwise low-LR cells never memorize and their "g*" is noise.
    # Reference: lr=3e-4 -> 2 epochs. epochs = round(2 * 3e-4 / lr), clamped to [2, 12].
    EPOCHS = int(max(2, min(12, round(2 * 3e-4 / use_lr))))
    for _ in range(EPOCHS):
        for b in range(0, len(items), BATCH):
            batch = items[b:b + BATCH]
            ml = max(len(x[0]) for x in batch); pad = tok.pad_token_id
            I, L, A = [], [], []
            for ids, l in batch:
                n = ml - len(ids); I.append(ids + [pad] * n); L.append(l + [-100] * n); A.append([1] * len(ids) + [0] * n)
            I = torch.tensor(I, device=dev); L = torch.tensor(L, device=dev); A = torch.tensor(A, device=dev)
            o = pm(input_ids=I, attention_mask=A, labels=L); o.loss.backward(); opt.step(); opt.zero_grad()
    pm.eval()
    import numpy as np
    all_objs = [o for (_, o) in facts]
    def measure_recall():
        hit = 0
        for (subj, obj) in facts:
            cue = f"The {REL} of {subj} is"
            lps = [H["seq_logprob"](pm, cue, cand) for cand in all_objs]
            if all_objs[int(np.argmax(lps))] == obj:
                hit += 1
        return hit / len(facts)
    recall = measure_recall()
    if not retain:
        del base, pm; torch.cuda.empty_cache()
        return round(recall, 4)
    # RETENTION: continue training on PURE FILLER (no fact exposures), measure recall decay.
    import random as _r
    rng = _r.Random(seed + 999)
    filler_items = [H["filler_ids"](rng.choice(FILLER)) for _ in range(RETAIN_FILLER)]
    pm.train()
    for b in range(0, len(filler_items), BATCH):
        batch = filler_items[b:b + BATCH]
        ml = max(len(x[0]) for x in batch); pad = tok.pad_token_id
        I, L, A = [], [], []
        for ids, l in batch:
            n = ml - len(ids); I.append(ids + [pad] * n); L.append(l + [-100] * n); A.append([1] * len(ids) + [0] * n)
        I = torch.tensor(I, device=dev); L = torch.tensor(L, device=dev); A = torch.tensor(A, device=dev)
        o = pm(input_ids=I, attention_mask=A, labels=L); o.loss.backward(); opt.step(); opt.zero_grad()
    pm.eval()
    retained = measure_recall()
    del base, pm; torch.cuda.empty_cache()
    return {"recall": round(recall, 4), "retained": round(retained, 4)}


def _run(model_name, gpu, seed, gaps):
    import random
    from transformers import AutoModelForCausalLM  # noqa
    H = _make(model_name); dev = H["dev"]
    rng = random.Random(seed)
    idx = list(range(min(N_FACTS, len(SUBJECTS), len(OBJECTS)))); rng.shuffle(idx)
    facts = [(SUBJECTS[i], OBJECTS[i]) for i in idx[:N_FACTS]]
    out = {"model": model_name, "gpu": gpu, "seed": seed, "n_facts": len(facts), "recall_by_gap": {}}
    for g in gaps:
        out["recall_by_gap"][str(g)] = _run_gap(model_name, dev, H, facts, g, seed)
    return out


@app.function(image=image, gpu="A10G", volumes={"/results": vol, "/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_dict(_hf_env())], timeout=60 * 55)
def run_cell(model_name: str, seed: int, gaps: list) -> dict:
    return _run(model_name, "A10G", seed, gaps)


def _run_scaling(model_name, gpu, seed, gaps, lr):
    """Gap sweep at a specific (model, lr) so we can see if the optimum g* SHIFTS -> a scaling law.
    Saves its own result to the volume immediately (robust to app teardown)."""
    import random
    H = _make(model_name); dev = H["dev"]
    rng = random.Random(seed)
    idx = list(range(min(N_FACTS, len(SUBJECTS), len(OBJECTS)))); rng.shuffle(idx)
    facts = [(SUBJECTS[i], OBJECTS[i]) for i in idx[:N_FACTS]]
    out = {"model": model_name, "gpu": gpu, "seed": seed, "lr": lr, "recall_by_gap": {}}
    for g in gaps:
        out["recall_by_gap"][str(g)] = _run_gap(model_name, dev, H, facts, g, seed, lr=lr)
    # per-cell durable save
    tag = f"{model_name.split('/')[-1]}_lr{lr}_s{seed}".replace('.', '')
    with open(f"/results/spacing_cell_{tag}.json", "w") as f:
        json.dump(out, f)
    vol.commit()
    return out


@app.function(image=image, gpu="A10G", volumes={"/results": vol, "/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_dict(_hf_env())], timeout=60 * 55)
def run_scaling_cell(model_name: str, seed: int, gaps: list, lr: float) -> dict:
    return _run_scaling(model_name, "A10G", seed, gaps, lr)


@app.function(image=image, gpu="A100-40GB", volumes={"/results": vol, "/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_dict(_hf_env())], timeout=60 * 55)
def run_scaling_cell_a100(model_name: str, seed: int, gaps: list, lr: float) -> dict:
    return _run_scaling(model_name, "A100-40GB", seed, gaps, lr)


def _run_retain(model_name, gpu, seed, gaps):
    """Recall AND post-filler retained recall per gap -> does the spacing optimum also govern retention?"""
    import random
    H = _make(model_name); dev = H["dev"]
    rng = random.Random(seed)
    idx = list(range(min(N_FACTS, len(SUBJECTS), len(OBJECTS)))); rng.shuffle(idx)
    facts = [(SUBJECTS[i], OBJECTS[i]) for i in idx[:N_FACTS]]
    out = {"model": model_name, "gpu": gpu, "seed": seed, "by_gap": {}}
    for g in gaps:
        out["by_gap"][str(g)] = _run_gap(model_name, dev, H, facts, g, seed, retain=True)
    tag = f"retain_{model_name.split('/')[-1]}_s{seed}".replace('.', '')
    with open(f"/results/spacing_cell_{tag}.json", "w") as f:
        json.dump(out, f)
    vol.commit()
    return out


@app.function(image=image, gpu="A10G", volumes={"/results": vol, "/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_dict(_hf_env())], timeout=60 * 55)
def run_retain_cell(model_name: str, seed: int, gaps: list) -> dict:
    return _run_retain(model_name, "A10G", seed, gaps)


def _run_retain_sweep(seeds):
    import numpy as np
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    jobs = [(MODEL, s) for s in seed_list]
    print(f"SPACING-RETAIN: {len(jobs)} cells, gaps={GAPS}")
    handles = [(m, s, run_retain_cell.spawn(m, s, GAPS)) for m, s in jobs]
    results = []
    for m, s, h in handles:
        try:
            r = h.get()
        except Exception as e:  # noqa: BLE001
            r = {"model": m, "seed": s, "error": str(e)}
        results.append(r)
        if "error" not in r:
            rec = {k: v["recall"] for k, v in r["by_gap"].items()}
            ret = {k: v["retained"] for k, v in r["by_gap"].items()}
            print(f"[retain] s{s} recall:   {rec}")
            print(f"         s{s} retained: {ret}")
    valid = [r for r in results if "error" not in r]
    if valid:
        gks = sorted(valid[0]["by_gap"].keys(), key=float)
        print("\n=== recall vs retained-recall by gap (mean over seeds) ===")
        for gk in gks:
            rec = np.mean([r["by_gap"][gk]["recall"] for r in valid])
            ret = np.mean([r["by_gap"][gk]["retained"] for r in valid])
            print(f"  gap={gk}: recall={rec:.3f}  retained={ret:.3f}  (kept {ret/max(rec,0.01):.0%})")
    _save.remote(results, "exp_spacing_retain_results.json")
    print("LOCAL_RESULTS_JSON_BEGIN"); print(json.dumps(results)); print("LOCAL_RESULTS_JSON_END")


def _run_scaling_sweep(seeds):
    """Does the optimum g* SHIFT with LR (update magnitude) and model size? -> the scaling law."""
    import numpy as np
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    gaps = GAPS
    # grid: 2 model sizes x 3 LRs. Bigger model -> A100.
    grid = [
        ("HuggingFaceTB/SmolLM2-360M-Instruct", 1e-4),
        ("HuggingFaceTB/SmolLM2-360M-Instruct", 3e-4),
        ("HuggingFaceTB/SmolLM2-360M-Instruct", 6e-4),
        ("Qwen/Qwen2.5-1.5B-Instruct", 1e-4),
        ("Qwen/Qwen2.5-1.5B-Instruct", 3e-4),
        ("Qwen/Qwen2.5-1.5B-Instruct", 6e-4),
    ]
    big = {"Qwen/Qwen2.5-1.5B-Instruct"}
    jobs = [(m, s, gaps, lr) for (m, lr) in grid for s in seed_list]
    print(f"SPACING-SCALING: {len(jobs)} cells ({len(grid)} model×lr × {len(seed_list)} seeds), gaps={gaps}")
    results = []
    # bounded waves of 5 to respect the GPU cap
    for i in range(0, len(jobs), 5):
        wave = jobs[i:i + 5]
        handles = [(m, s, lr, (run_scaling_cell_a100 if m in big else run_scaling_cell).spawn(m, s, g, lr)) for m, s, g, lr in wave]
        for m, s, lr, h in handles:
            try:
                r = h.get()
            except Exception as e:  # noqa: BLE001
                r = {"model": m, "seed": s, "lr": lr, "error": str(e)}
            results.append(r)
            if "error" not in r:
                print(f"[scaling] {m.split('/')[-1][:14]:14} lr={lr}: {r['recall_by_gap']}")
            else:
                print(f"[scaling] {m} lr={lr} ERR: {r['error'][:80]}")
    # per (model,lr): mean recall curve over seeds -> argmax gap = g*
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        if "error" in r:
            continue
        key = (r["model"].split("/")[-1], r["lr"])
        for gk, v in r["recall_by_gap"].items():
            agg[key][gk].append(v)
    print("\n=== g* (argmax-recall gap) by (model, lr) ===")
    for key, curve in agg.items():
        gks = sorted(curve.keys(), key=float)
        means = {gk: float(np.mean(curve[gk])) for gk in gks}
        gstar = max(gks, key=lambda k: means[k])
        print(f"  {key[0]:14} lr={key[1]}: g*={gstar}  peak={means[gstar]:.3f}  curve={ {k: round(means[k],2) for k in gks} }")
    print("\nLOOK FOR: does g* move MONOTONICALLY with lr and/or model size? -> a scaling LAW.")
    _save.remote(results, "exp_spacing_scaling_results.json")
    print("LOCAL_RESULTS_JSON_BEGIN"); print(json.dumps(results)); print("LOCAL_RESULTS_JSON_END")


def _run_families_sweep(seeds):
    """Does the inverted-U REPLICATE across model families? (robustness gate for the paper.)
    Runs the 8-gap sweep on 4 families at base LR. Each family reported separately; if a family
    saturates or floors, retune that one afterward."""
    import numpy as np
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    FAMILIES = [
        "HuggingFaceTB/SmolLM2-360M-Instruct",   # already confirmed (control)
        "Qwen/Qwen2.5-1.5B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "google/gemma-2-2b-it",
    ]
    big = {"Qwen/Qwen2.5-1.5B-Instruct", "google/gemma-2-2b-it"}
    jobs = [(m, s) for m in FAMILIES for s in seed_list]
    print(f"SPACING-FAMILIES: {len(jobs)} cells ({len(FAMILIES)} families x {len(seed_list)} seeds), gaps={GAPS}")
    results = []
    for i in range(0, len(jobs), 5):
        wave = jobs[i:i + 5]
        handles = [(m, s, (run_scaling_cell_a100 if m in big else run_scaling_cell).spawn(m, s, GAPS, CFG["lr"])) for m, s in wave]
        for m, s, h in handles:
            try:
                r = h.get()
            except Exception as e:  # noqa: BLE001
                r = {"model": m, "seed": s, "error": str(e)}
            results.append(r)
            if "error" not in r:
                print(f"[families] {m.split('/')[-1][:16]:16} s{s}: {r['recall_by_gap']}")
            else:
                print(f"[families] {m} s{s} ERR: {r['error'][:90]}")
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        if "error" in r:
            continue
        fam = r["model"].split("/")[-1]
        for gk, v in r["recall_by_gap"].items():
            agg[fam][gk].append(v)
    print("\n=== inverted-U by family (mean recall; g* = argmax; INVERTED-U if peak is interior) ===")
    for fam, curve in agg.items():
        gks = sorted(curve.keys(), key=float)
        means = {gk: float(np.mean(curve[gk])) for gk in gks}
        gstar = max(gks, key=lambda k: means[k])
        interior = gstar not in (gks[0], gks[-1])
        endpts = max(means[gks[0]], means[gks[-1]])
        print(f"  {fam:16} g*={gstar} peak={means[gstar]:.2f} vs endpoints={endpts:.2f} "
              f"{'INVERTED-U' if interior and means[gstar] - endpts > 0.15 else 'flat/monotone/saturated'}  curve={ {k: round(means[k],2) for k in gks} }")
    _save.remote(results, "exp_spacing_families_results.json")
    print("LOCAL_RESULTS_JSON_BEGIN"); print(json.dumps(results)); print("LOCAL_RESULTS_JSON_END")


@app.local_entrypoint()
def main(validate: bool = False, seeds: str = "1,2,3", scaling: bool = False, families: bool = False, retain: bool = False):
    import numpy as np
    if scaling:
        _run_scaling_sweep(seeds)
        return
    if families:
        _run_families_sweep(seeds)
        return
    if retain:
        _run_retain_sweep(seeds)
        return
    if validate:
        jobs = [(MODEL, 1, [0.0, 0.2, 0.5, 0.8, 1.0])]
    else:
        seed_list = [int(s) for s in seeds.split(",") if s.strip()]
        jobs = [(MODEL, s, GAPS) for s in seed_list]
    print(f"SPACING-LAW: {len(jobs)} cells, gaps={jobs[0][2]}")
    handles = [(m, s, run_cell.spawn(m, s, g)) for m, s, g in jobs]
    results = []
    for m, s, h in handles:
        try:
            r = h.get()
        except Exception as e:  # noqa: BLE001
            r = {"model": m, "seed": s, "error": str(e)}
        results.append(r)
        if "error" not in r:
            print(f"[spacing] s{s} recall_by_gap: {r['recall_by_gap']}")
        else:
            print(f"[spacing] s{s} ERR: {r['error'][:100]}")
    valid = [r for r in results if "error" not in r]
    if valid:
        gaps = sorted(valid[0]["recall_by_gap"].keys(), key=float)
        print("\n=== recall vs gap (mean over seeds) ===")
        means = {}
        for gk in gaps:
            vals = [r["recall_by_gap"][gk] for r in valid]
            means[gk] = (np.mean(vals), np.std(vals))
            print(f"  gap={gk}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")
        # non-monotone check: is there an interior gap beating BOTH endpoints by >3 sigma?
        gk = gaps
        endpoints = [means[gk[0]][0], means[gk[-1]][0]]
        interior = [(k, means[k][0], means[k][1]) for k in gk[1:-1]]
        if interior:
            best_k, best_m, best_sd = max(interior, key=lambda x: x[1])
            margin_lo = best_m - max(endpoints)
            pooled_sd = (best_sd + max(means[gk[0]][1], means[gk[-1]][1])) / 2 + 1e-6
            sigmas = margin_lo / pooled_sd
            print(f"\ninterior optimum g*={best_k} recall={best_m:.3f}; beats endpoints by {margin_lo:+.3f} ({sigmas:.1f} sigma)")
            print(f"KILL-CHECK: {'INTERIOR OPTIMUM (spacing law candidate)' if sigmas > 3 else 'MONOTONE/FLAT (null - methods critique)'}")
    _save.remote(results, "exp_spacing_law_results.json")
    print("DONE."); print("LOCAL_RESULTS_JSON_BEGIN"); print(json.dumps(results)); print("LOCAL_RESULTS_JSON_END")


@app.function(image=image, volumes={"/results": vol})
def _save(results: list, name: str):
    with open(f"/results/{name}", "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()
