r"""
消融實驗：只改「葉節點深度補償」，深度上限維持原始碼的設定。

背景：原始碼裡 log 的底數影響兩個地方
    1. 樹的深度上限 δ（--depth-limit paper 只改這個）
    2. 葉節點的深度補償：分數計算時，葉節點深度 + log(葉節點筆數 / η)
這支腳本只把第 2 項改成以 2 為底，用來確認改進到底來自哪一項。

用法與 run_experiment.py 完全相同，例如：
    python experiments\ablation_leaf.py --dataset data\stationary\ifor\Annthyroid.csv --runs 30 --tag leaflog2
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from numpy import log  # noqa: E402
from OnlineIForest.BoundedRandomProjectionOnlineIForest import BoundedRandomProjectionOnlineITree  # noqa: E402
import run_experiment  # noqa: E402


def recursive_depth_search_log2(self, node, data, depths):
    if node.children is None or data.shape[0] == 0:
        correction = 0 if node.data_size < self.max_leaf_samples else log(node.data_size / self.max_leaf_samples) / log(2)
        depths[:] = node.depth + correction
    else:
        partition_indices = self.split_data(data, node.projection_vector, node.split_values)
        for i, indices in enumerate(partition_indices):
            depths[indices] = recursive_depth_search_log2(self, node.children[i], data[indices], depths[indices])
    return depths


BoundedRandomProjectionOnlineITree.recursive_depth_search = recursive_depth_search_log2

if __name__ == "__main__":
    run_experiment.main()
