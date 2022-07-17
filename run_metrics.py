from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import data
import metrics
import models
import torch
import torch.cuda as cuda


def compute_metrics(
    model_id: str,
    dataset_path: str,
    split: str,
    metric_names: list[str],
    start_step: int = 0,
    end_step: Optional[int] = None,
    batch_size: int = 8,
    pck_threshold: float = 0.5,
):
    results_dir = os.path.join("results", model_id)
    dataset_cache_dir = os.path.join("metrics/cache/humans")

    if cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if end_step is None:
        checkpoint_path = models.HumanGAN._get_checkpoint_path(model_id)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        end_step = checkpoint["global_step"]
        del checkpoint

    dataset = None
    for step in range(max(start_step, 10000), end_step + 10000, 10000):  # type: ignore
        cuda.empty_cache()

        model = models.HumanGAN.load_from_id(model_id, step)
        generator: models.Generator = model.generator_ema  # type: ignore

        if dataset is None:
            dataset = data.HumansDataset(
                args.dataset,
                generator.resolution,
                num_frames=1,
                spacing=1,
                split=split,
            )
            num_gen_samples = 50000
            print(f"Dataset subsets: {dataset.subsets}")

        for metric in metric_names:
            results = {
                "dataset": dataset_path,
                "split": split,
                "model_id": model_id,
                "step": step,
                "seed": torch.initial_seed(),
            }

            if metric == "fid":
                fid = metrics.compute_fid(
                    generator,
                    dataset,
                    batch_size,
                    device,
                    dataset_cache_dir,
                    num_gen_samples=num_gen_samples,  # type: ignore
                )
                results["fid"] = round(fid, ndigits=4)

            elif metric == "kid":
                kid = metrics.compute_kid(
                    generator, dataset, batch_size, device, dataset_cache_dir,
                )
                results["kid"] = round(kid, ndigits=4)

            elif metric == "pr":
                precision, recall = metrics.compute_pr(
                    generator, dataset, batch_size, device, dataset_cache_dir,
                )
                results["precision"] = round(precision, ndigits=6)
                results["recall"] = round(recall, ndigits=6)

            elif metric == "is":
                is_mean, is_std = metrics.compute_is(
                    generator, dataset, batch_size, device,
                )
                results["is_mean"] = round(is_mean, ndigits=4)
                results["is_std"] = round(is_std, ndigits=4)

            elif metric == "pck":
                pck_avg, pck = metrics.compute_pck(
                    generator,
                    dataset,
                    batch_size,
                    device,
                    num_samples=num_gen_samples,
                    threshold=pck_threshold,
                )
                results["pck"] = round(pck_avg, ndigits=4)
                results["threshold"] = pck_threshold

            else:
                raise ValueError(f"Invalid metric: {metric}")

            os.makedirs(results_dir, exist_ok=True)

            results_path = os.path.join(
                results_dir, f"metric-{metric}-{split}.jsonl"
            )

            json_line = json.dumps(results, sort_keys=True)
            print(json_line)

            with open(results_path, "at") as open_file:
                open_file.write(f"{json_line}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--end_step", type=int)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["fid", "pck"],
        choices=("fid", "is", "kid", "pck", "pr"),
    )
    parser.add_argument("--dataset", default="/home/timbrooks/datasets/humans")
    parser.add_argument("--split", default="train", choices=["test", "train"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pck_threshold", type=float, default=0.5)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    compute_metrics(
        args.model_id,
        args.dataset,
        args.split,
        args.metrics,
        args.start_step,
        args.end_step,
        args.batch_size,
        args.pck_threshold,
    )
