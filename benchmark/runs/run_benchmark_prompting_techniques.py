#!/usr/bin/env python
"""
run_benchmark_prompting_techniques.py — Compare 4 prompting strategies on the
meeting-room readiness checklist task.

Each technique produces a per-item Check / Output report:

    Check                          Output
    Items left on tables           No
    Whiteboard is clean            Yes
    Chairs arranged properly       Yes
    ...

Technique 1 — DIRECT (per-item, N calls per image)
    One model call per checklist item.
    Prompt: "Is [condition] true in this image? Answer only Yes or No."

Technique 2 — CHAIN-OF-THOUGHT (one batch call per image)
    Single call. Model reasons through each item step-by-step, then outputs
    a Check / Output list.

Technique 3 — FEW-SHOT BATCH (one call per image with reference images)
    Single call via run_few_shot(). Two withheld reference images (READY +
    NOT READY) are prepended, then the full checklist is evaluated at once.

Technique 4 — FEW-SHOT PER-ITEM (N calls per image with reference images)
    One call per checklist item via run_few_shot(). Reference images are shown
    alongside a targeted classification question for that specific item.
    "In the first image, [item] is TRUE. In the second, it is FALSE.
     Looking at the test image — is [item] TRUE or FALSE?"

Test-set format:
  {
    "checklist": [{"id": 1, "item": "Table surface is clear"}, ...],
    "reference_images": [
      {"image": "...", "label": "ready"},
      {"image": "...", "label": "not_ready"}
    ],
    "samples": [{"id": "room_001", "image": "...", "ground_truth": {...}}, ...]
  }

Metrics (per technique × model):
  item_accuracy, room_accuracy, room_f1, room_precision, room_recall,
  per_item accuracy, parse_error_rate, mean_latency_ms

Usage:
    cd benchmark
    python run_benchmark_prompting_techniques.py --test-set test_sets/meeting_room_sample.json --all
    python run_benchmark_prompting_techniques.py --models smolvlm --techniques direct cot
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from config import BenchmarkConfig, load_config
from models import MODEL_REGISTRY

TECHNIQUES = ["direct", "cot", "few_shot", "few_shot_per_item"]


# ── Prompt builders ────────────────────────────────────────────────────────────

def build_direct_item_prompt(item: dict) -> str:
    """Technique 1: one call per item — specific observable question about the room."""
    question = item.get("question", item["item"])
    return (
        f"Look carefully at this meeting room photo.\n\n"
        f"{question}\n\n"
        f"Answer only 'Yes' or 'No'."
    )


def build_cot_batch_prompt(checklist: list[dict]) -> str:
    """Technique 2: identify observable things in the image first, then evaluate each."""
    # Build list of observable things to look for
    observe_lines = "\n".join(
        f"- {item.get('question', item['item'])}" for item in checklist
    )
    check_lines = "\n".join(f"{item['item']}: Yes or No" for item in checklist)
    return f"""\
Look at this meeting room photo and identify whether each of the following is present:

{observe_lines}

For each, describe specifically what you see, then answer Yes or No.

After your observations, output ONLY this checklist (no other text):
---
{check_lines}
room_ready: Yes or No
---"""


def build_few_shot_batch_prompt(checklist: list[dict]) -> str:
    """Technique 3: reference images + full checklist at once.
    First image = READY room, second = NOT READY, third = room to evaluate."""
    check_lines = "\n".join(f"{item['item']}: Yes or No" for item in checklist)
    observe_lines = "\n".join(
        f"- {item.get('question', item['item'])}" for item in checklist
    )
    return f"""\
You are given three meeting room images.
Image 1: a READY room (all conditions met).
Image 2: a NOT READY room (conditions not met).
Image 3: the room to evaluate.

Using images 1 and 2 as reference, evaluate image 3 for each condition:

{observe_lines}

Output ONLY this checklist (no other text):
---
{check_lines}
room_ready: Yes or No
---"""


def build_few_shot_item_prompt(item: dict, ready_ref_is_first: bool) -> str:
    """Technique 4: per-item with reference images.
    Shows one reference where the observable condition is present, one where it is absent.
    ready_ref_is_first: True if the first reference image is the READY room.
    """
    question = item.get("question", item["item"])
    yes_means_met = item.get("yes_means_met", True)

    # In the READY room, the checklist condition is MET.
    # yes_means_met=True  → "yes" answer means condition met → ready room answer = Yes
    # yes_means_met=False → "yes" answer means bad thing present → ready room answer = No
    if yes_means_met:
        ready_answer   = "Yes"
        notready_answer = "No"
    else:
        ready_answer   = "No"
        notready_answer = "Yes"

    if ready_ref_is_first:
        img1_desc = f"Image 1 (READY room) — answer is {ready_answer}"
        img2_desc = f"Image 2 (NOT READY room) — answer is {notready_answer}"
    else:
        img1_desc = f"Image 1 (NOT READY room) — answer is {notready_answer}"
        img2_desc = f"Image 2 (READY room) — answer is {ready_answer}"

    return (
        f"{img1_desc}.\n"
        f"{img2_desc}.\n"
        f"Image 3 is the room to classify.\n\n"
        f"{question}\n\n"
        f"Answer only 'Yes' or 'No' for Image 3."
    )


# ── Response parsers ───────────────────────────────────────────────────────────

def parse_yes_no(response: str) -> bool | None:
    """Parse a single Yes/No or True/False response. Returns raw answer (True=yes/true)."""
    r = response.strip().lower()
    if re.search(r'\byes\b|\btrue\b', r):
        return True
    if re.search(r'\bno\b|\bfalse\b', r):
        return False
    return None


def answer_to_condition_met(answer: bool | None, item: dict) -> bool:
    """Convert a raw yes/no answer to whether the checklist condition is met,
    accounting for yes_means_met direction."""
    if answer is None:
        return False
    yes_means_met = item.get("yes_means_met", True)
    return answer if yes_means_met else not answer


def parse_checklist_response(response: str, checklist: list[dict]) -> dict | None:
    """
    Parse the Check/Output block from CoT and few-shot batch responses.

    Expected format (inside --- delimiters or just plain):
        Table surface is clear: Yes
        Whiteboard is clean: No
        ...
        room_ready: Yes
    """
    items: dict[str, bool] = {}

    for item in checklist:
        # Match "item text: Yes/No" (case-insensitive, partial match ok)
        # Try exact item text first, then keywords
        escaped = re.escape(item["item"][:30])
        m = re.search(rf'{escaped}.*?:\s*(yes|no)\b', response, re.IGNORECASE)
        if not m:
            # Fallback: match on the first few words of the item
            words = item["item"].split()[:4]
            pattern = r'\s+'.join(re.escape(w) for w in words) + r'.*?:\s*(yes|no)\b'
            m = re.search(pattern, response, re.IGNORECASE)
        if m:
            items[str(item["id"])] = m.group(1).lower() == "yes"

    rr_m = re.search(r'room_ready\s*:\s*(yes|no)\b', response, re.IGNORECASE)
    if rr_m:
        room_ready = rr_m.group(1).lower() == "yes"
    else:
        room_ready = bool(items) and all(items.values())

    if len(items) < max(1, len(checklist) // 2):
        return None

    for item in checklist:
        items.setdefault(str(item["id"]), False)

    return {"items": items, "room_ready": room_ready}


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict], checklist: list[dict]) -> dict:
    item_ids    = [str(c["id"]) for c in checklist]
    item_labels = {str(c["id"]): c["item"] for c in checklist}

    total_items = correct_items = 0
    per_item_correct = {iid: 0 for iid in item_ids}
    per_item_total   = {iid: 0 for iid in item_ids}

    room_correct = tp = fp = fn = tn = 0
    parse_errors = 0
    latencies: list[float] = []

    for r in results:
        pred = r.get("predicted")
        gt   = r["ground_truth"]

        if pred is None:
            parse_errors += 1
            continue

        latencies.append(r.get("latency_ms", 0))

        for iid in item_ids:
            p = pred["items"].get(iid, False)
            g = gt["items"].get(iid, False)
            per_item_total[iid] += 1
            total_items += 1
            if p == g:
                per_item_correct[iid] += 1
                correct_items += 1

        p_ready, g_ready = pred["room_ready"], gt["room_ready"]
        if p_ready == g_ready:
            room_correct += 1
        if   p_ready and     g_ready: tp += 1
        elif p_ready and not g_ready: fp += 1
        elif not p_ready and g_ready: fn += 1
        else:                         tn += 1

    n       = len(results)
    n_valid = n - parse_errors
    item_acc  = correct_items / total_items if total_items else 0.0
    room_acc  = room_correct  / n_valid     if n_valid    else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    per_item_acc = {
        iid: {
            "label":    item_labels[iid],
            "accuracy": per_item_correct[iid] / per_item_total[iid] if per_item_total[iid] else 0.0,
            "correct":  per_item_correct[iid],
            "total":    per_item_total[iid],
        }
        for iid in item_ids
    }

    return {
        "item_accuracy":    round(item_acc,  4),
        "room_accuracy":    round(room_acc,  4),
        "room_precision":   round(precision, 4),
        "room_recall":      round(recall,    4),
        "room_f1":          round(f1,        4),
        "per_item":         per_item_acc,
        "n_images":         n,
        "n_valid":          n_valid,
        "parse_errors":     parse_errors,
        "mean_latency_ms":  round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ── Per-sample runner ──────────────────────────────────────────────────────────

def _run_sample_direct(model, mcfg, sample, checklist) -> dict:
    """Technique 1: N separate yes/no calls, one per item."""
    gt         = sample["ground_truth"]
    image_path = sample["image"]
    items: dict[str, bool] = {}
    total_latency = 0.0
    parse_errors  = 0
    raw_parts: list[str] = []

    for item in checklist:
        prompt = build_direct_item_prompt(item)
        result = model.run(image_path, prompt)
        total_latency += result.latency_ms
        raw_parts.append(f"{item['item']}: {result.response.strip()}")

        raw_ans = parse_yes_no(result.response)
        if raw_ans is None:
            parse_errors += 1
        items[str(item["id"])] = answer_to_condition_met(raw_ans, item)

    room_ready = all(items.values())
    predicted  = {"items": items, "room_ready": room_ready}

    return {
        "id": sample["id"], "image": image_path,
        "ground_truth": gt, "predicted": predicted,
        "parse_error": f"{parse_errors} item(s) unparsed" if parse_errors else None,
        "raw_response": "\n".join(raw_parts),
        "latency_ms": round(total_latency),
        "model_name": model.name, "model_path": mcfg.model_path,
        "technique": "direct",
    }


def _run_sample_cot(model, mcfg, sample, checklist) -> dict:
    """Technique 2: one batch call with step-by-step reasoning."""
    gt         = sample["ground_truth"]
    image_path = sample["image"]
    prompt     = build_cot_batch_prompt(checklist)
    result     = model.run(image_path, prompt)

    predicted = None
    parse_err = None
    if result.error:
        parse_err = result.error
    else:
        predicted = parse_checklist_response(result.response, checklist)
        if predicted is None:
            parse_err = "parse failed"

    return {
        "id": sample["id"], "image": image_path,
        "ground_truth": gt, "predicted": predicted,
        "parse_error": parse_err, "raw_response": result.response,
        "latency_ms": round(result.latency_ms),
        "model_name": model.name, "model_path": mcfg.model_path,
        "technique": "cot",
    }


def _run_sample_few_shot_batch(model, mcfg, sample, checklist, ref_images) -> dict:
    """Technique 3: one batch call prepending reference images."""
    gt         = sample["ground_truth"]
    image_path = sample["image"]
    prompt     = build_few_shot_batch_prompt(checklist)

    available = [(p, lbl) for p, lbl in ref_images if Path(p).exists()]
    if available:
        try:
            result = model.run_few_shot(available, image_path, prompt)
        except NotImplementedError:
            result = model.run(image_path, prompt)
    else:
        result = model.run(image_path, prompt)

    predicted = None
    parse_err = None
    if result.error:
        parse_err = result.error
    else:
        predicted = parse_checklist_response(result.response, checklist)
        if predicted is None:
            parse_err = "parse failed"

    return {
        "id": sample["id"], "image": image_path,
        "ground_truth": gt, "predicted": predicted,
        "parse_error": parse_err, "raw_response": result.response,
        "latency_ms": round(result.latency_ms),
        "model_name": model.name, "model_path": mcfg.model_path,
        "technique": "few_shot",
    }


def _run_sample_few_shot_per_item(model, mcfg, sample, checklist, ref_images) -> dict:
    """Technique 4: one call per item, each with reference images showing
    the condition TRUE (first ref) and FALSE (second ref)."""
    gt         = sample["ground_truth"]
    image_path = sample["image"]
    available  = [(p, lbl) for p, lbl in ref_images if Path(p).exists()]

    # Determine which reference image is "ready" (condition likely TRUE)
    # ref label "ready" → all items TRUE in that room
    ready_first = True
    if available:
        ready_first = available[0][1] == "ready"

    items: dict[str, bool] = {}
    total_latency = 0.0
    parse_errors  = 0
    raw_parts: list[str] = []

    for item in checklist:
        prompt = build_few_shot_item_prompt(item, ready_ref_is_first=ready_first)

        if available:
            try:
                result = model.run_few_shot(available, image_path, prompt)
            except NotImplementedError:
                result = model.run(image_path, prompt)
        else:
            result = model.run(image_path, prompt)

        total_latency += result.latency_ms
        raw_parts.append(f"{item['item']}: {result.response.strip()}")

        raw_ans = parse_yes_no(result.response)
        if raw_ans is None:
            parse_errors += 1
        items[str(item["id"])] = answer_to_condition_met(raw_ans, item)

    room_ready = all(items.values())
    predicted  = {"items": items, "room_ready": room_ready}

    return {
        "id": sample["id"], "image": image_path,
        "ground_truth": gt, "predicted": predicted,
        "parse_error": f"{parse_errors} item(s) unparsed" if parse_errors else None,
        "raw_response": "\n".join(raw_parts),
        "latency_ms": round(total_latency),
        "model_name": model.name, "model_path": mcfg.model_path,
        "technique": "few_shot_per_item",
    }


# ── Console output helpers ─────────────────────────────────────────────────────

def _print_sample_result(r: dict, checklist: list[dict]) -> None:
    pred = r.get("predicted")
    gt   = r["ground_truth"]
    sid  = r["id"]

    if pred is None:
        print(f"    [{sid}] PARSE ERROR: {r.get('parse_error','')}")
        return

    item_ids   = [str(c["id"]) for c in checklist]
    item_parts = []
    for iid in item_ids:
        p = pred["items"].get(iid, False)
        g = gt["items"].get(iid, False)
        item_parts.append(f"{'✓' if p == g else '✗'}{iid}({'Y' if p else 'N'})")

    room_match = "✓" if pred["room_ready"] == gt["room_ready"] else "✗"
    print(
        f"    [{sid}] {' '.join(item_parts)} | "
        f"ready={room_match}({'Y' if pred['room_ready'] else 'N'} "
        f"gt={'Y' if gt['room_ready'] else 'N'}) | {r['latency_ms']}ms"
    )


def _print_check_output_table(r: dict, checklist: list[dict]) -> None:
    """Print the Check / Output table for one sample."""
    pred = r.get("predicted")
    if pred is None:
        return
    gt = r["ground_truth"]
    col = 42
    print(f"    {'Check':<{col}} {'Model':>6}  {'GT':>4}")
    print(f"    {'-'*col} {'------':>6}  {'----':>4}")
    for item in checklist:
        iid = str(item["id"])
        p   = pred["items"].get(iid, False)
        g   = gt["items"].get(iid, False)
        mark = "" if p == g else " ✗"
        print(f"    {item['item']:<{col}} {'Yes' if p else 'No':>6}  {'Yes' if g else 'No':>4}{mark}")
    p_ready, g_ready = pred["room_ready"], gt["room_ready"]
    mark = "" if p_ready == g_ready else " ✗"
    print(f"    {'room_ready':<{col}} {'Yes' if p_ready else 'No':>6}  {'Yes' if g_ready else 'No':>4}{mark}")


# ── Core runner ────────────────────────────────────────────────────────────────

def run_one_technique(
    technique: str,
    model,
    mcfg,
    samples: list[dict],
    checklist: list[dict],
    ref_images: list[tuple[str, str]],
) -> list[dict]:
    results: list[dict] = []

    for sample in samples:
        if not Path(sample["image"]).exists():
            print(f"    [{sample['id']}] WARNING: image not found — {sample['image']}")
            results.append({
                "id": sample["id"], "image": sample["image"],
                "ground_truth": sample["ground_truth"], "predicted": None,
                "parse_error": "image not found", "raw_response": "",
                "latency_ms": 0, "model_name": model.name, "technique": technique,
            })
            continue

        if technique == "direct":
            r = _run_sample_direct(model, mcfg, sample, checklist)
        elif technique == "cot":
            r = _run_sample_cot(model, mcfg, sample, checklist)
        elif technique == "few_shot":
            r = _run_sample_few_shot_batch(model, mcfg, sample, checklist, ref_images)
        else:  # few_shot_per_item
            r = _run_sample_few_shot_per_item(model, mcfg, sample, checklist, ref_images)

        _print_sample_result(r, checklist)
        results.append(r)

    return results


def run_prompting_benchmark(
    cfg: BenchmarkConfig,
    test_set_path: str,
    model_filter: list[str] | None,
    technique_filter: list[str] | None,
    verbose: bool = False,
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(test_set_path) as f:
        test_set = json.load(f)

    checklist: list[dict]          = test_set["checklist"]
    samples:   list[dict]          = test_set["samples"]
    ref_images: list[tuple[str,str]] = [
        (ri["image"], ri["label"]) for ri in test_set.get("reference_images", [])
    ]

    techniques_to_run = [t for t in TECHNIQUES if technique_filter is None or t in technique_filter]
    models_to_run     = [m for m in cfg.enabled_models if model_filter is None or m.key in model_filter]

    print(f"\nChecklist ({len(checklist)} items):")
    for item in checklist:
        print(f"  {item['id']}. {item['item']}")
    print(f"\nSamples:    {len(samples)}")
    print(f"Techniques: {techniques_to_run}")
    avail_refs = sum(1 for p, _ in ref_images if Path(p).exists())
    print(f"Ref images: {len(ref_images)} ({avail_refs} on disk)")

    all_results: dict[str, dict[str, dict]] = {}

    for mcfg in models_to_run:
        cls = MODEL_REGISTRY.get(mcfg.cls_name)
        if cls is None:
            print(f"[SKIP] Unknown class '{mcfg.cls_name}'")
            continue

        model = cls(mcfg)
        print(f"\n{'='*60}\nModel: {model.name}\n{'='*60}")
        model.load()

        all_results[mcfg.key] = {}

        for technique in techniques_to_run:
            print(f"\n  -- Technique: {technique.upper()} --")
            results = run_one_technique(technique, model, mcfg, samples, checklist, ref_images)

            # Print one Check/Output table for first valid sample if verbose
            if verbose:
                for r in results:
                    if r.get("predicted"):
                        print(f"\n  Example ({r['id']}):")
                        _print_check_output_table(r, checklist)
                        break

            metrics = compute_metrics(results, checklist)
            all_results[mcfg.key][technique] = {
                "model_name":  model.name,
                "model_path":  mcfg.model_path,
                "technique":   technique,
                "metrics":     metrics,
                "results":     results,
            }
            print(
                f"  item_acc={metrics['item_accuracy']:.1%}  "
                f"room_acc={metrics['room_accuracy']:.1%}  "
                f"room_F1={metrics['room_f1']:.3f}  "
                f"parse_err={metrics['parse_errors']}/{metrics['n_images']}  "
                f"latency={metrics['mean_latency_ms']:.0f}ms"
            )

        model.unload()

    out = {
        "timestamp":        timestamp,
        "checklist":        checklist,
        "test_set":         test_set_path,
        "techniques":       techniques_to_run,
        "reference_images": [{"image": p, "label": l} for p, l in ref_images],
        "models":           all_results,
    }
    json_path = cfg.output_dir / f"prompting_techniques_results_{timestamp}.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults → {json_path}")

    print_summary(all_results, checklist, techniques_to_run)
    print(f"\nGenerate report:\n  python generate_prompting_report.py --results {json_path}")


# ── Console summary ────────────────────────────────────────────────────────────

def print_summary(all_results: dict, checklist: list[dict], techniques: list[str]) -> None:
    print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
    print(f"{'Model':<32} {'Technique':<18} {'ItemAcc':>8} {'RoomAcc':>8} {'RoomF1':>8} {'ParseErr':>9}")
    print("-" * 80)
    for mk, tech_data in all_results.items():
        for technique in techniques:
            if technique not in tech_data:
                continue
            m = tech_data[technique]["metrics"]
            print(
                f"{tech_data[technique]['model_name']:<32} {technique:<18} "
                f"{m['item_accuracy']:>8.1%} "
                f"{m['room_accuracy']:>8.1%} "
                f"{m['room_f1']:>8.3f} "
                f"{m['parse_errors']:>9}"
            )
        print()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark 4 prompting techniques on meeting room readiness"
    )
    parser.add_argument("--config",    default="benchmark_config.yaml")
    parser.add_argument("--test-set",  default="test_sets/meeting_room_sample.json")
    parser.add_argument("--models",    nargs="+", default=None)
    parser.add_argument("--all",       action="store_true")
    parser.add_argument("--techniques", nargs="+", default=None, choices=TECHNIQUES)
    parser.add_argument("--verbose",   action="store_true",
                        help="Print a Check/Output table for one example per technique")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Config:     {args.config}")
    print(f"Test set:   {args.test_set}")
    print(f"Models:     {args.models or [m.key for m in cfg.enabled_models]}")
    print(f"Techniques: {args.techniques or TECHNIQUES}")

    run_prompting_benchmark(
        cfg,
        args.test_set,
        model_filter=args.models,
        technique_filter=args.techniques,
        verbose=args.verbose,
    )
