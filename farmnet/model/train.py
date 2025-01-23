from pickle import dump
from pathlib import Path

import lightning as L  # type: ignore
from torch_geometric.data import InMemoryDataset  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore
from lightning.pytorch.loggers import CSVLogger

from torch_geometric.nn.summary import summary

from farmnet.data.datasets import train_test_split
from farmnet.model.gnn import FarmGAT

from lightning.pytorch.loggers import MLFlowLogger


def train_gnn(
    dataset: InMemoryDataset,
    gnn_params: dict,
    train_params: dict = {},
    scaler_path: str | Path = Path("scaler"),
    log_dir: str | Path = Path("logs"),
    experiment: str = "GNN",
):
    """
    General training function that uses DataLoaders, the lightning Trainer class
    as well as the CSVLogger and MLFlowLogger classes.
    """
    train_data, val_data = train_test_split(dataset)

    batch_size = train_params.get("batch_size", 1)

    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    dim_in = train_data[0].num_node_features
    dim_out = (
        1 if len(train_data[0].y.shape) == 1 else train_data[0].y.shape[1]
    )

    model = FarmGAT(
        dim_in,
        dim_out,
        **gnn_params,
    )

    # Initialise the bias of the last layer (MLP)
    model.mp[0].lins[-1].bias.data.fill_(6.8)

    mlf_logger = MLFlowLogger(
        experiment_name="lightning_logs",
        tracking_uri="file:./ml-runs",
        log_model=True,
    )
    run_id = mlf_logger.run_id

    x = train_data[0].x
    edge_index = train_data[0].edge_index
    edge_attr = train_data[0].edge_attr

    model_summary = str(summary(model, (x, edge_index, edge_attr)))

    mlf_logger.experiment.log_text(run_id, model_summary, "model_summary.txt")
    mlf_logger.experiment.set_tag(run_id, "Model", str(model.__class__))
    mlf_logger.experiment.set_tag(run_id, "Dataset", str(dataset))

    csv_logger = CSVLogger(log_dir, f"{experiment}")

    loggers = [csv_logger, mlf_logger]
    max_epochs = train_params.get("max_epochs", 200)
    trainer = L.Trainer(
        max_epochs=max_epochs, logger=loggers, accelerator="gpu"
    )
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
