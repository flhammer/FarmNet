from pathlib import Path
import lightning as L
import numpy as np
from numpy._typing import NDArray
from typing import overload, Any
from abc import abstractmethod
import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP, GATv2Conv
from torch_geometric.nn.norm import BatchNorm, LayerNorm
from torch_geometric.utils import dropout_edge

__all__ = [
    "BaseGNN",
    "FarmGAT",
    "get_checkpoint",
    "load_model",
]

AttrBatchType = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
AttentionWeightsType = list[tuple[torch.Tensor, torch.Tensor]]
FarmGATReturnType = torch.Tensor | tuple[torch.Tensor, AttentionWeightsType]


class BaseGNN(L.LightningModule):
    """
    Base GNN
    """

    def __init__(self, lr, weight_decay):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        """Training step"""
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        y_hat = self((x, edge_index, edge_attr))
        loss = F.mse_loss(y_hat, batch.y.reshape(batch.num_nodes, -1))

        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        y_hat = self((x, edge_index, edge_attr))
        val_loss = F.mse_loss(y_hat, batch.y.reshape(batch.num_nodes, -1))

        self.log("val_loss", val_loss, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        y_hat = self((x, edge_index, edge_attr))
        test_loss = F.mse_loss(y_hat, batch.y.reshape(batch.num_nodes, -1))

        self.log("test_loss", test_loss)
        return test_loss

    def predict_step(self, batch, batch_idx):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        return self((x, edge_index, edge_attr))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()


class FarmGAT(BaseGNN):
    """
    Windfarm GNN model: encoder + stage + head

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        **kwargs (optional): Optional additional args
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        h_dims,
        heads,
        concats,
        edge_dims,
        skip_connections,
        n_mlp_channels,
        n_mlp_layers,
        lr: float = 0.005,
        weight_decay: float = 5e-4,
    ):
        self.lr = lr
        self.weight_decay = weight_decay

        super().__init__(lr, weight_decay)

        self.n_nodes = None

        gat_layer_params = [
            h_dims,
            heads,
            concats,
            edge_dims,
            skip_connections,
        ]

        if len(set(map(len, gat_layer_params))) not in (0, 1):
            raise ValueError(
                "The parameter lists for the GAT layers need to have the same length"
            )

        gat_layers = []
        norm_layers = []
        in_dim = in_channels
        for (
            h_dim,
            n_heads,
            concat,
            edge_dim,
            skip_con,
        ) in zip(*gat_layer_params):
            gat_layers.append(
                GATLayer(in_dim, h_dim, n_heads, concat, edge_dim, skip_con)
            )
            if concat:
                in_dim = h_dim * n_heads
            else:
                in_dim = h_dim
            # norm_layers.append(LayerNorm(in_dim))
            norm_layers.append(BatchNorm(in_dim))

        self.gat = torch.nn.Sequential(*gat_layers)
        self.norm = torch.nn.Sequential(*norm_layers)

        if concats[-1]:
            final_layer_in = heads[-1] * h_dims[-1]
        else:
            final_layer_in = h_dims[-1]

        self.mp = torch.nn.Sequential(
            MLP(
                in_channels=final_layer_in,
                hidden_channels=n_mlp_channels,
                num_layers=n_mlp_layers,
                out_channels=out_channels,
            ),
        )

        self.save_hyperparameters()

    def forward(
        self, batch: AttrBatchType, att_weights=False
    ) -> FarmGATReturnType:
        x, edge_index, edge_attr = batch

        A: AttentionWeightsType = []
        for layer, norm in zip(self.gat, self.norm):
            if att_weights:
                x, edge_index, edge_attr, A_ = layer(
                    (x, edge_index, edge_attr), att_weights
                )
                A.append(A_)
            else:
                x, edge_index, edge_attr = layer((x, edge_index, edge_attr))

            # x = norm(x)

            if self.training:
                edge_index, edge_mask = dropout_edge(edge_index, p=0.2)

                if edge_attr is not None:
                    edge_attr = edge_attr[edge_mask]

        x = self.mp(x)

        if att_weights:
            return x, A

        return x


class GATLayer(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        heads,
        concat=True,
        edge_dim=None,
        skip_connection=True,
    ):
        super().__init__()

        self.gat_conv = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            edge_dim=edge_dim,
        )
        self.edge_dim = edge_dim

        self.gat_post = F.leaky_relu

        self.skip_con = None

        if skip_connection:
            if concat:
                self.skip_con = torch.nn.Linear(
                    in_channels, out_channels * heads, bias=False
                )
            else:
                self.skip_con = torch.nn.Linear(
                    in_channels, out_channels, bias=False
                )

    def forward(self, data, att_weights=False):
        x, edge_index, edge_attr = data
        if self.edge_dim is None:
            edge_attr = None
        input = x
        if att_weights:
            x, A = self.gat_conv(
                x, edge_index, edge_attr, return_attention_weights=att_weights
            )
        else:
            x = self.gat_conv(x, edge_index, edge_attr)
            A = None

        x = self.gat_post(x)
        if self.skip_con is not None:
            x += self.skip_con(input)

        if A is not None:
            return x, edge_index, edge_attr, A

        return x, edge_index, edge_attr


class GATLayerV2(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        heads,
        concat=True,
        edge_dim=None,
        skip_connection=True,
    ):
        super().__init__()

        self.gat_conv = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            edge_dim=edge_dim,
        )
        self.edge_dim = edge_dim

        self.gat_post = F.elu

        self.skip_con = None

        if skip_connection:
            if concat:
                self.skip_con = torch.nn.Linear(
                    in_channels, out_channels * heads, bias=False
                )
            else:
                self.skip_con = torch.nn.Linear(
                    in_channels, out_channels, bias=False
                )

    def forward(self, data, att_weights=False):
        x, edge_index, edge_attr = data
        if self.edge_dim is None:
            edge_attr = None
        input = x
        if att_weights:
            x, A = self.gat_conv(
                x, edge_index, edge_attr, return_attention_weights=att_weights
            )
        else:
            x = self.gat_conv(x, edge_index, edge_attr)
            A = None

        if self.skip_con is not None:
            x += self.skip_con(input)

        x = self.gat_post(x)
        if A is not None:
            return x, edge_index, edge_attr, A

        return x, edge_index, edge_attr


class FarmGATv2(BaseGNN):
    """
    Windfarm GNN model: encoder + stage + head

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        **kwargs (optional): Optional additional args
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        h_dims,
        heads,
        concats,
        edge_dims,
        skip_connections,
        n_mlp_channels,
        n_mlp_layers,
        lr: float = 0.005,
        weight_decay: float = 5e-4,
    ):
        self.lr = lr
        self.weight_decay = weight_decay

        super().__init__(lr, weight_decay)

        self.n_nodes = None

        gat_layer_params = [
            h_dims,
            heads,
            concats,
            edge_dims,
            skip_connections,
        ]

        if len(set(map(len, gat_layer_params))) not in (0, 1):
            raise ValueError(
                "The parameter lists for the GAT layers need to have the same length"
            )

        gat_layers = []
        norm_layers = []
        in_dim = in_channels
        for (
            h_dim,
            n_heads,
            concat,
            edge_dim,
            skip_con,
        ) in zip(*gat_layer_params):
            gat_layers.append(
                GATLayerV2(in_dim, h_dim, n_heads, concat, edge_dim, skip_con)
            )
            if concat:
                in_dim = h_dim * n_heads
            else:
                in_dim = h_dim
            # norm_layers.append(LayerNorm(in_dim))
            norm_layers.append(BatchNorm(in_dim))

        self.gat = torch.nn.Sequential(*gat_layers)
        self.norm = torch.nn.Sequential(*norm_layers)

        if concats[-1]:
            final_layer_in = heads[-1] * h_dims[-1]
        else:
            final_layer_in = h_dims[-1]

        self.mp = torch.nn.Sequential(
            MLP(
                in_channels=final_layer_in,
                hidden_channels=n_mlp_channels,
                num_layers=n_mlp_layers,
                out_channels=out_channels,
            ),
        )

        self.save_hyperparameters()

    def forward(
        self, batch: AttrBatchType, att_weights=False
    ) -> FarmGATReturnType:
        x, edge_index, edge_attr = batch

        A: AttentionWeightsType = []
        for layer, norm in zip(self.gat, self.norm):
            if att_weights:
                x, edge_index, edge_attr, A_ = layer(
                    (x, edge_index, edge_attr), att_weights
                )
                A.append(A_)
            else:
                x, edge_index, edge_attr = layer((x, edge_index, edge_attr))

            # x = norm(x)

            if self.training:
                edge_index, edge_mask = dropout_edge(edge_index, p=0.2)

                if edge_attr is not None:
                    edge_attr = edge_attr[edge_mask]

        x = self.mp(x)

        if att_weights:
            return x, A

        return x


def load_model(checkpoint, gnn):
    model = gnn.load_from_checkpoint(checkpoint)

    return model


def latest_version(root_dir: str | Path):
    root_dir = Path(root_dir)
    versions = sorted(
        [int(path.name.split("_")[-1]) for path in root_dir.glob("version_*")]
    )
    version = versions[-1]
    return version


def get_checkpoint(root_dir: str | Path, version: int | None = None):
    root_dir = Path(root_dir)
    if version is None or version == -1:
        version = latest_version(root_dir)

    checkpoint = list(
        (root_dir / f"version_{version}/checkpoints/").glob("*.ckpt")
    )[0]

    return checkpoint
