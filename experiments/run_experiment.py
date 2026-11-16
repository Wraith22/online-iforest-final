r"""
Online Isolation Forest 評估腳本

用法（在專案根目錄執行）：
    python experiments\run_experiment.py --dataset data\stationary\ifor\Annthyroid.csv

資料格式（與原作者相同）：第一列是欄位名稱，第一欄是 id，最後一欄是標籤（1 = 異常，0 = 正常）。

評估流程（與原作者 experiments/scripts/test/*.py 相同）：
    1. 每次執行前打亂資料（--no-shuffle 可關閉，INSECTS 這類有時間順序的資料要關）
    2. 資料切成每批約 100 筆，依序餵入
    3. 每一批：先 learn_batch，再 score_batch
    4. 全部跑完後，用所有分數計算一個 ROC AUC（= 論文 Table 4 的數字）
    5. 重複 --runs 次，回報中位數

輸出：
    results/<資料集>_<方法>_summary.csv   每次執行的 AUC 與時間
    results/<資料集>_<方法>_run<i>.npz     每次執行的逐點分數與標籤（給畫圖用）
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# 讓 Python 找得到專案根目錄下的 OnlineIForest/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from OnlineIForest import OnlineIForest  # noqa: E402


def load_dataset(path, max_features=None):
    df = pd.read_csv(path)
    values = df.to_numpy(dtype=float)
    X = values[:, 1:-1]            # 去掉第一欄 id 與最後一欄標籤
    y = values[:, -1].astype(int)  # 1 = 異常
    if max_features is not None:
        X = X[:, :max_features]
    return X, y


def make_model(args):
    if args.method == "oifor":
        # 參數與論文 Table 3、原作者腳本相同
        return OnlineIForest.create(
            iforest_type="boundedrandomprojectiononlineiforest",
            branching_factor=2,
            metric="axisparallel",
            num_trees=args.trees,
            max_leaf_samples=args.eta,
            window_size=args.window,
            type="adaptive",
            subsample=1.0,
            n_jobs=1,
        )
    raise ValueError(f"未知的方法：{args.method}")


def run_once(X, y, args, seed):
    # 原始碼內部使用 numpy 的全域亂數（uniform、choice、random），所以要設全域種子
    np.random.seed(seed)
    if args.shuffle:
        order = np.random.permutation(len(X))
        X, y = X[order], y[order]

    model = make_model(args)
    # 與原作者相同的切法：floor(n / batch) + 1 批，每批約 batch 筆
    n_batches = int(np.floor(len(X) / args.batch)) + 1
    batches = np.array_split(np.arange(len(X)), n_batches)

    scores = np.empty(len(X))
    pending_resets = sorted(args.reset_at)
    start = time.perf_counter()
    for idx in batches:
        if len(idx) == 0:
            continue
        # Oracle 實驗：資料流走到指定位置時，丟掉整個模型重新開始（模擬「完美的 drift 偵測器」）
        while pending_resets and idx[0] >= pending_resets[0]:
            model = make_model(args)
            pending_resets.pop(0)
        model.learn_batch(X[idx])
        scores[idx] = model.score_batch(X[idx])
    elapsed = time.perf_counter() - start

    auc = roc_auc_score(y, scores)
    return auc, elapsed, scores, y


def main():
    p = argparse.ArgumentParser(description="Online Isolation Forest 評估")
    p.add_argument("--dataset", required=True, help="CSV 檔路徑")
    p.add_argument("--method", default="oifor", choices=["oifor"])
    p.add_argument("--runs", type=int, default=5, help="重複次數（論文是 30）")
    p.add_argument("--seed", type=int, default=0, help="第 i 次執行使用 seed + i")
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--window", type=int, default=2048, help="視窗大小 ω")
    p.add_argument("--eta", type=int, default=32, help="分裂門檻 η（max_leaf_samples）")
    p.add_argument("--trees", type=int, default=32, help="樹的數量 τ")
    p.add_argument("--no-shuffle", dest="shuffle", action="store_false",
                   help="不打亂資料（有時間順序的資料集要用）")
    p.add_argument("--max-features", type=int, default=None,
                   help="只取前 k 個特徵（原作者跑 INSECTS 只用前 10 個）")
    p.add_argument("--reset-at", type=int, nargs="*", default=[],
                   help="在這些位置重置模型（oracle drift 實驗用，需搭配 --no-shuffle）")
    p.add_argument("--out", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--tag", default="", help="加在輸出檔名後面，區分不同設定")
    args = p.parse_args()

    name = Path(args.dataset).stem
    X, y = load_dataset(args.dataset, args.max_features)
    print(f"資料集 {name}：n={len(X)}, d={X.shape[1]}, 異常比例={y.mean():.2%}")
    print(f"方法 {args.method}：ω={args.window}, η={args.eta}, τ={args.trees}, "
          f"batch={args.batch}, shuffle={args.shuffle}")
    if args.reset_at:
        if args.shuffle:
            raise SystemExit("--reset-at 必須搭配 --no-shuffle，否則位置沒有意義")
        print(f"Oracle 重置位置：{args.reset_at}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{name}_{args.method}{('_' + args.tag) if args.tag else ''}"

    # 刪掉同名的舊結果，避免上次跑比較多次時留下的檔案混進這次的分析
    for old in out_dir.glob(f"{prefix}_run*.npz"):
        old.unlink()

    rows = []
    for i in range(args.runs):
        seed = args.seed + i
        auc, elapsed, scores, y_run = run_once(X, y, args, seed)
        rows.append({"run": i, "seed": seed, "auc": auc, "time_s": elapsed})
        np.savez_compressed(out_dir / f"{prefix}_run{i}.npz", scores=scores, labels=y_run)
        print(f"  第 {i + 1}/{args.runs} 次：AUC={auc:.4f}  時間={elapsed:.1f}s")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / f"{prefix}_summary.csv", index=False)
    print(f"中位數 AUC = {summary.auc.median():.4f}  "
          f"（最小 {summary.auc.min():.4f}，最大 {summary.auc.max():.4f}）")
    print(f"中位數時間 = {summary.time_s.median():.1f}s")
    print(f"結果存在 {out_dir}")


if __name__ == "__main__":
    main()
