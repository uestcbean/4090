"""Standard-library estimator for dense full-attention, unshared KV caches.

This is a logical tensor-size estimate, not allocator usage or a fit guarantee.
"""
import argparse
import json
import math
from pathlib import Path

GIB = 1024 ** 3


def estimate(config, lengths, weight_bits=16, kv_bytes=2,
             capacity_gib=24, reserve_gib=3):
    dims = [config[k] for k in
            ("num_hidden_layers", "num_key_value_heads", "head_dim")]
    params = config["parameter_count_estimate"]
    if any(type(x) is not int or x <= 0 for x in [*dims, params]):
        raise ValueError("Model dimensions and parameter estimate must be positive integers")
    if not lengths or any(type(t) is not int or t < 0 for t in lengths):
        raise ValueError("Supply nonnegative cached lengths for at least one request")
    values = [weight_bits, kv_bytes, capacity_gib, reserve_gib]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Budget inputs must be finite")
    if min(weight_bits, kv_bytes, capacity_gib) <= 0 or reserve_gib < 0:
        raise ValueError("Precisions/capacity must be positive; reserve nonnegative")
    per_token = 2 * math.prod(dims) * kv_bytes
    weights = params * weight_bits / 8
    kv = sum(lengths) * per_token
    total = weights + kv + reserve_gib * GIB
    return {
        "kind": "estimate_not_measurement",
        "model": config.get("model_id"),
        "cached_lengths": lengths,
        "weight_bits": weight_bits,
        "kv_bytes_per_element": kv_bytes,
        "kv_bytes_per_token": per_token,
        "weights_gib_estimate": weights / GIB,
        "kv_gib": kv / GIB,
        "runtime_and_other_reserve_gib_assumption": reserve_gib,
        "total_gib_estimate": total / GIB,
        "capacity_gib_assumption": capacity_gib,
        "headroom_gib_estimate": capacity_gib - total / GIB,
        "fits_arithmetic_budget_only": total <= capacity_gib * GIB,
        "limitations": [
            "Rounded parameter count; quantization metadata/unquantized layers excluded",
            "No prefix sharing, sliding windows, padding or allocator block rounding",
            "Reserve is an assumption for activations/workspaces/graphs/desktop, not measured",
            "A positive headroom is not a guarantee of successful serving"
        ]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", required=True,
                        help="Current cached tokens per request, including processed output")
    parser.add_argument("--weight-bits", type=float, default=16)
    parser.add_argument("--kv-bytes", type=float, default=2)
    parser.add_argument("--capacity-gib", type=float, default=24)
    parser.add_argument("--reserve-gib", type=float, default=3)
    args = parser.parse_args()
    try:
        result = estimate(json.loads(args.config.read_text(encoding="utf-8")),
                          args.lengths, args.weight_bits, args.kv_bytes,
                          args.capacity_gib, args.reserve_gib)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
