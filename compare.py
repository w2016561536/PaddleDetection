#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np


def compare_npy(a_path, b_path, rtol=1e-5, atol=1e-6, topk=10, save_diff=None):
    a = np.load(a_path)
    b = np.load(b_path)

    print("A:", a_path)
    print("  shape:", a.shape)
    print("  dtype:", a.dtype)
    print("B:", b_path)
    print("  shape:", b.shape)
    print("  dtype:", b.dtype)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    a = a.astype(np.float64)
    b = b.astype(np.float64)

    diff = a - b
    abs_diff = np.abs(diff)

    eps = 1e-12
    rel_diff = abs_diff / (np.abs(b) + eps)

    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_diff)
    max_abs = np.max(abs_diff)
    max_rel = np.max(rel_diff)
    mean_rel = np.mean(rel_diff)

    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)

    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    cosine = dot / (norm_a * norm_b + eps)

    allclose = np.allclose(a, b, rtol=rtol, atol=atol)

    print("\n===== Error Summary =====")
    print("allclose:", allclose)
    print("rtol:", rtol)
    print("atol:", atol)
    print("max_abs_error:", max_abs)
    print("mean_abs_error:", mae)
    print("mse:", mse)
    print("rmse:", rmse)
    print("max_relative_error:", max_rel)
    print("mean_relative_error:", mean_rel)
    print("cosine_similarity:", cosine)

    max_idx = np.argmax(abs_diff)
    max_index = np.unravel_index(max_idx, a.shape)

    print("\n===== Max Error Position =====")
    print("index:", max_index)
    print("A value:", a[max_index])
    print("B value:", b[max_index])
    print("diff A-B:", diff[max_index])
    print("abs diff:", abs_diff[max_index])
    print("relative diff:", rel_diff[max_index])

    print(f"\n===== Top {topk} Abs Errors =====")
    flat_abs = abs_diff.reshape(-1)
    topk = min(topk, flat_abs.size)

    top_indices = np.argpartition(-flat_abs, topk - 1)[:topk]
    top_indices = top_indices[np.argsort(-flat_abs[top_indices])]

    for rank, flat_idx in enumerate(top_indices, 1):
        idx = np.unravel_index(flat_idx, a.shape)
        print(
            f"{rank:02d} index={idx}, "
            f"A={a[idx]}, "
            f"B={b[idx]}, "
            f"diff={diff[idx]}, "
            f"abs={abs_diff[idx]}, "
            f"rel={rel_diff[idx]}"
        )

    if save_diff is not None:
        np.save(save_diff, diff.astype(np.float32))
        print("\nSaved diff to:", save_diff)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("a", help="first npy file")
    parser.add_argument("b", help="second npy file")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--save-diff", default=None, help="save A-B diff as npy")

    args = parser.parse_args()

    compare_npy(
        args.a,
        args.b,
        rtol=args.rtol,
        atol=args.atol,
        topk=args.topk,
        save_diff=args.save_diff,
    )


if __name__ == "__main__":
    main()
    