"""Verify exported sessions and summarize the local reading-task pilot."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from statistics import median


CONDITIONS = ("planar", "first_fit", "proposed")
RESPONSES = frozenset(("A", "B", "C", "D", "skip", "timeout"))
MIN_RESPONSE_MS = 150
LIMIT_MS = 60_000
PARTICIPANT_CODE = re.compile(r"[A-Za-z0-9_-]{1,24}")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def canonical_hash(content):
    return sha256(json.dumps(content, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


def verify_materials(study, key):
    content = {name: value for name, value in study.items() if name != "study_id"}
    if (study.get("schema") != 1 or key.get("schema") != 1
            or canonical_hash(content) != study["study_id"] or key["study_id"] != study["study_id"]
            or canonical_hash(key["items"]) != study["key_sha256"]):
        raise ValueError("Stimulus and answer key identifiers or hashes do not match.")
    answer = {row["id"]: row["correct"] for row in key["items"]}
    if (set(answer) != {row["id"] for row in study["items"]}
            or set(answer.values()) - set("ABCD") or len(answer) != len(key["items"])):
        raise ValueError("Answer key does not match the item pool.")
    return answer


def check_session(row, study, answer):
    slot = row["slot"]
    viewport = row["viewport"]
    if (row.get("schema") != 1 or row["study_id"] != study["study_id"]
            or type(slot) is not int or not 0 <= slot < len(study["assignments"])
            or row["purpose"] not in {"pilot", "dry-run"}
            or not PARTICIPANT_CODE.fullmatch(row["participant"])
            or any(type(viewport[key]) is not int or viewport[key] <= 0 for key in ("width", "height"))
            or type(viewport["device_pixel_ratio"]) not in (int, float)
            or not isfinite(viewport["device_pixel_ratio"]) or viewport["device_pixel_ratio"] <= 0):
        raise ValueError("Session metadata is invalid or belongs to another study.")
    trials = study["assignments"][slot]
    records = row["records"]
    if not 1 <= len(records) <= len(trials) or row["complete"] is not (len(records) == len(trials)):
        raise ValueError("Session completion status does not match the assigned length.")
    result = {condition: [] for condition in CONDITIONS}
    for log, trial in zip(records, trials):
        if any(log[key] != trial[key] for key in ("item", "condition")):
            raise ValueError("Session trials differ from the selected assignment.")
        milliseconds = log["milliseconds"]
        response = log["response"]
        interrupted = log["interrupted"]
        if (response not in RESPONSES or type(milliseconds) is not int
                or not 0 <= milliseconds <= LIMIT_MS or type(interrupted) is not bool
                or (response == "timeout") != (milliseconds == LIMIT_MS)):
            raise ValueError("Session response or timing is invalid.")
        result[log["condition"]].append({
            "correct": response == answer[log["item"]],
            "milliseconds": milliseconds,
            "usable": not interrupted and milliseconds >= MIN_RESPONSE_MS,
            "response": response,
        })
    return result


def participant_summary(by_condition):
    usable = {condition: [trial for trial in trials if trial["usable"]]
              for condition, trials in by_condition.items()}
    if not all(usable.values()):
        return None
    summaries = {}
    for condition, rows in usable.items():
        correct_times = [row["milliseconds"] for row in rows if row["correct"]]
        summaries[condition] = {
            "usable_trials": len(rows),
            "correct_trials": len(correct_times),
            "accuracy": len(correct_times) / len(rows),
            "median_correct_ms": median(correct_times) if correct_times else None,
            "skipped_or_timeout": sum(row["response"] in {"skip", "timeout"} for row in rows),
        }
    return summaries


def median_if_present(values):
    present = [value for value in values if value is not None]
    return median(present) if present else None


def aggregate(study, answer, exports):
    seen, complete = set(), []
    slots, viewports = Counter(), Counter()
    excluded = {"dry_run": 0, "incomplete": 0, "no_usable_condition": 0}
    for row in exports:
        by_condition = check_session(row, study, answer)
        identity = (row["purpose"], row["participant"])
        if identity in seen:
            raise ValueError(f"Duplicate session for {row['participant']}.")
        seen.add(identity)
        if row["purpose"] == "dry-run":
            excluded["dry_run"] += 1
            continue
        if not row["complete"]:
            excluded["incomplete"] += 1
            continue
        summary = participant_summary(by_condition)
        if summary is None:
            excluded["no_usable_condition"] += 1
            continue
        complete.append(summary)
        slots[row["slot"]] += 1
        viewport = row["viewport"]
        viewports[(viewport["width"], viewport["height"], viewport["device_pixel_ratio"])] += 1

    conditions = {}
    for condition in CONDITIONS:
        rows = [row[condition] for row in complete]
        conditions[condition] = {
            "participants": len(rows),
            "usable_trials": sum(row["usable_trials"] for row in rows),
            "correct_trials": sum(row["correct_trials"] for row in rows),
            "median_participant_accuracy": median_if_present(row["accuracy"] for row in rows),
            "median_participant_correct_ms": median_if_present(row["median_correct_ms"] for row in rows),
            "skipped_or_timeout": sum(row["skipped_or_timeout"] for row in rows),
        }

    contrasts = {}
    for baseline in ("planar", "first_fit"):
        differences = [row["proposed"]["accuracy"] - row[baseline]["accuracy"] for row in complete]
        available = [(row["proposed"]["median_correct_ms"], row[baseline]["median_correct_ms"])
                     for row in complete]
        time_differences = [proposal - base for proposal, base in available
                            if proposal is not None and base is not None]
        contrasts[baseline] = {
            "median_paired_accuracy_difference": median_if_present(differences),
            "median_paired_correct_ms_difference": median_if_present(time_differences),
            "participants_with_correct_time_in_both": len(time_differences),
        }
    slot_counts = [slots[index] for index in range(study["allocation_slots"])]
    return {"study_id": study["study_id"], "included_participants": len(complete),
            "included_participants_by_slot": slot_counts,
            "all_slots_equally_populated": bool(min(slot_counts)) and min(slot_counts) == max(slot_counts),
            "included_viewports": [{"width": width, "height": height, "device_pixel_ratio": ratio,
                                    "participants": count}
                                   for (width, height, ratio), count in sorted(viewports.items())],
            "excluded_sessions": excluded, "conditions": conditions,
            "proposed_minus_baseline": contrasts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("export", nargs="+", type=Path)
    args = parser.parse_args(argv)
    study, key = read(args.study), read(args.key)
    summary = aggregate(study, verify_materials(study, key), (read(path) for path in args.export))
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
