# -*- coding: utf-8 -*-
"""Functions for handling PyTorch Geometric Datasets"""

from pathlib import Path
from typing import Any
import numpy as np
from torch_geometric.data import InMemoryDataset  # type: ignore

from farmnet.data.datasets.kelmarsh import KelmarshDataset

# from farmgnn.configuration import settings


def dataset_sample(
    dataset: InMemoryDataset, sample_size: int
) -> InMemoryDataset:
    len_dataset = len(dataset)
    if sample_size > len_dataset:
        raise IndexError(
            f"The sample size {sample_size} is greater than the dataset length {len_dataset}"
        )

    idx = np.arange(len_dataset)

    sample_idx = np.random.choice(idx, size=sample_size, replace=False)

    return dataset.copy(sample_idx)


def load_dataset(path: str | Path) -> InMemoryDataset:
    dataset = KelmarshDataset(
        path,
        data_path=None,
        windfarm_static_path=None,
        features=["u_g", "v_g", "nacelle_direction"],
        target="wind_speed",
        wt_col="wt_id",
    )

    return dataset


#     data_path = Path(settings.dataset.data_path).expanduser().absolute()
#     dataset_dir = Path(settings.dataset.root_dir).expanduser().absolute()
#
#     config = {"graph": settings.dataset.graph, "windfarm": settings.windfarm}
#     if settings.dataset.name == "WinJiDataset":
#         return WinJiDataset(dataset_dir, data_path, config=config)
#     elif settings.dataset.name == "PyWakeDataset":
#         return PyWakeDataset(dataset_dir, data_path, config=config)
#     else:
#         raise ValueError(f"Dataset {settings.dataset.name} does not exist!")


def train_test_split(
    dataset: InMemoryDataset, test_size: float = 0.2, seed: int = 0
) -> tuple[Any, Any]:
    np.random.seed(seed)
    len_dataset = len(dataset)
    idx = np.arange(len_dataset)
    train_len = int(np.round(len_dataset * (1.0 - test_size)))
    train_idx = np.random.choice(idx, size=train_len, replace=False)
    test_idx = list(set(idx).difference(train_idx))

    return dataset.index_select(train_idx), dataset.index_select(test_idx)
