"""Build nested stratified subsets of data_04_12.json for the volume-sweep figure.

Subsets of 500, 1k, 5k are produced with the nesting property
500 subset ⊂ 1k subset ⊂ 5k subset ⊂ 10k (full), so the volume sweep
only ever adds examples — no stratum is resampled between sizes.

Stratification:
  * corpus (invoice / order / vehicle) — proportional to the 10k prior
  * within vehicle, one of 9 categories (Protocole, Bon_Picking, ...) — proportional

Corpus detection uses the `input` field signatures; vehicle category comes from
parsing the `output` JSON (the dataset itself is shuffled).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "data_04_12.json"
OUT_DIR = DATA_PATH.parent
TARGET_SIZES = [500, 1000, 5000]
SEED = 20260419


def detect_corpus(entry: dict) -> str:
    inp = entry.get("input", "")
    if "VEHICLE automotive documents" in inp:
        return "vehicle"
    if "numero_expedition" in inp or "numero_commande" in inp:
        return "order"
    if '"recipient"' in inp and '"total_ht"' in inp:
        return "invoice"
    return "unknown"


def vehicle_category(entry: dict) -> str:
    try:
        return json.loads(entry["output"]).get("category", "autre")
    except Exception:
        return "autre"


def stratum_key(entry: dict) -> tuple[str, str]:
    corpus = detect_corpus(entry)
    if corpus == "vehicle":
        return ("vehicle", vehicle_category(entry))
    return (corpus, "")


def largest_remainder(total: int, weights: dict[str, int]) -> dict[str, int]:
    """Allocate `total` integer slots across strata proportional to `weights`,
    using the largest-remainder method so the allocations sum exactly to total."""
    weight_sum = sum(weights.values())
    raw = {k: total * w / weight_sum for k, w in weights.items()}
    floor = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(floor.values())
    # distribute leftover by largest fractional part, tie-break on key for determinism
    order = sorted(weights, key=lambda k: (-(raw[k] - floor[k]), k))
    for k in order[:remainder]:
        floor[k] += 1
    return floor


def main() -> None:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))

    # Bucket entries by stratum, attaching original index for traceability
    strata: dict[tuple[str, str], list[int]] = {}
    for idx, entry in enumerate(data):
        strata.setdefault(stratum_key(entry), []).append(idx)

    # Deterministic shuffle of each stratum — subset takes a prefix, guaranteeing nesting
    rng = random.Random(SEED)
    for key in strata:
        rng.shuffle(strata[key])

    # Corpus-level weights (full-dataset counts)
    corpus_weights: dict[str, int] = {}
    vehicle_cat_weights: dict[str, int] = {}
    for (corpus, cat), idxs in strata.items():
        corpus_weights[corpus] = corpus_weights.get(corpus, 0) + len(idxs)
        if corpus == "vehicle":
            vehicle_cat_weights[cat] = len(idxs)

    print(f"Loaded {len(data)} entries; corpus counts: {corpus_weights}")
    print(f"VEHICLE category counts: {vehicle_cat_weights}")

    manifests: dict[int, dict] = {}
    for n in TARGET_SIZES:
        corpus_alloc = largest_remainder(n, corpus_weights)
        vehicle_alloc = largest_remainder(corpus_alloc["vehicle"], vehicle_cat_weights)

        selected_idxs: list[int] = []
        per_stratum: dict[str, int] = {}
        for (corpus, cat), idxs in strata.items():
            take = vehicle_alloc[cat] if corpus == "vehicle" else corpus_alloc[corpus]
            chosen = idxs[:take]
            selected_idxs.extend(chosen)
            per_stratum[f"{corpus}/{cat}" if cat else corpus] = len(chosen)

        selected_idxs.sort()
        subset = [data[i] for i in selected_idxs]
        out_path = OUT_DIR / f"data_04_12_subset_{n}.json"
        out_path.write_text(
            json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifests[n] = {
            "size": len(subset),
            "corpus_alloc": corpus_alloc,
            "vehicle_category_alloc": vehicle_alloc,
            "per_stratum": per_stratum,
            "output_file": out_path.name,
        }
        print(f"Wrote {out_path.name}: {per_stratum}")

    manifest_path = OUT_DIR / "data_04_12_subsets_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source": DATA_PATH.name,
                "seed": SEED,
                "corpus_weights": corpus_weights,
                "vehicle_category_weights": vehicle_cat_weights,
                "subsets": manifests,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {manifest_path.name}")

    # Verify nesting: each smaller subset's indices should be a subset of the next.
    # Re-derive index sets from on-disk files to catch any mismatch between the
    # saved JSON and the selected indices above.
    def _fingerprint(entry: dict) -> tuple:
        return (entry.get("instruction", ""), entry.get("input", ""), entry.get("output", ""))

    prev_keys: set | None = None
    for n in TARGET_SIZES:
        subset = json.loads((OUT_DIR / f"data_04_12_subset_{n}.json").read_text(encoding="utf-8"))
        keys = {_fingerprint(e) for e in subset}
        if prev_keys is not None:
            assert prev_keys.issubset(keys), f"Nesting violated at N={n}"
        prev_keys = keys
    print("Nesting verified: subset_500 <= subset_1000 <= subset_5000")


if __name__ == "__main__":
    main()
