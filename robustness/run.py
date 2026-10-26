"""Устойчивость LLM-агента на синтетике: python -m robustness"""

import argparse
import asyncio
import logging
from pathlib import Path

import aiohttp
import pandas as pd

from robustness.agent import load_agent
from robustness.config import Settings, path
from robustness.generate import make_paraphraser
from robustness.metrics import metrics, write_json

logger = logging.getLogger("robustness")


def load(s: Settings, limit: int | None) -> pd.DataFrame:
    file = path(s.dataset_path)
    frame = pd.read_csv(file) if file.suffix == ".csv" else pd.read_excel(file)
    frame = frame[frame[s.dataset_text_col].map(lambda t: isinstance(t, str) and bool(t.strip()))]
    return frame.head(limit) if limit else frame


async def run(s, predict, paraphrase, limit, n_boot, seed, out: Path, workers: int) -> dict:
    frame = load(s, limit)
    texts = frame[s.dataset_text_col].tolist()
    labels = frame[s.dataset_label_col].astype(bool).tolist()
    semaphore = asyncio.Semaphore(workers)

    async def limited(coro):
        async with semaphore:
            return await coro

    async def sample(session, text, label):
        """Парафраз + детекция одного диалога: (ожидаемая метка, предсказание)."""
        made = await paraphrase(session, text, label)
        return None if made is None else (label, await predict(session, made))

    async with aiohttp.ClientSession() as session:
        if s.dataset_pred_col:  # на оригиналах агент уже предсказал - берём готовое
            real_pred = frame[s.dataset_pred_col].astype(bool).tolist()
        else:
            real_pred = await asyncio.gather(*(limited(predict(session, t)) for t in texts))
        results = await asyncio.gather(*(limited(sample(session, t, l)) for t, l in zip(texts, labels)))
    synthetic = [item for item in results if item is not None]
    logger.info("generated %d/%d", len(synthetic), len(texts))

    def slice_metrics(keep):
        part = [item for item in synthetic if keep(item[0])]
        return metrics([e for e, _ in part], [p for _, p in part], n_boot, seed)

    report = {
        "real": metrics(labels, real_pred, n_boot, seed),
        "synthetic": slice_metrics(lambda _: True),
        "synthetic_by_class": {
            "positive": slice_metrics(bool),
            "negative": slice_metrics(lambda expected: not expected),
        },
        "generated": len(synthetic),
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2, help="одновременных диалогов")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    s = Settings()
    predict, reference = load_agent(s)
    report = asyncio.run(
        run(s, predict, make_paraphraser(s, reference), args.limit, args.bootstrap, args.seed,
            path(s.out_dir), args.workers)
    )
    for scope, part in report.items():
        if isinstance(part, dict) and "value" in part:
            print(scope, part["n"], part["value"], part["ci95"])
        elif isinstance(part, dict):
            for name, inner in part.items():
                print(f"{scope}.{name}", inner["n"], inner["value"], inner["ci95"])
