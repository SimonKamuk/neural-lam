# Standard library
import os
import warnings
from typing import List, Union

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
from torch import nn

# Local
from .. import metrics, utils, vis
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore
from ..interaction_net import InteractionNet, InteractionNetNew
from ..loss_weighting import get_state_feature_weighting
from ..weather_dataset import WeatherDataset

# from .ar_model import ARModel


class SequentialGNN(pl.LightningModule):
    def __init__(self, args, config: NeuralLAMConfig, datastore: BaseDatastore):
        super().__init__()

        self.armodel_init(args, config, datastore)
        self.basegraphmodel_init(args, config, datastore)
        self.basehigraphmodel_init(args, config, datastore)
        self.hilam_init(args, config, datastore)

        mesh_dim = self.mesh_static_features[0].shape[1]
        mesh_same_dim = self.m2m_features[0].shape[1]
        mesh_up_dim = self.mesh_up_features[0].shape[1]
        mesh_down_dim = self.mesh_down_features[0].shape[1]
        g2m_dim = self.g2m_features.shape[1]
        m2g_dim = self.m2g_features.shape[1]

        deembed_subgraph_dimension = {
            "grid": self.grid_dim
        }  # or double dim if also output std
        subgraph_feat_dims = {
            "grid": self.grid_dim,
            "mesh0": mesh_dim,
            "mesh1": mesh_dim,
            "mesh2": mesh_dim,
        }

        process_edge_feat_dims = {
            "grid_mesh0": g2m_dim,  # g2m
            "mesh0_grid": m2g_dim,  # m2g
        }

        self.num_processor_layers = args.processor_layers + 1
        # +1 because the first up (mesh init) and down (read out) steps are
        # not counted as processor layers
        self.process_steps = ["grid_mesh0"]
        for proc in range(self.num_processor_layers):
            for i in range(self.num_levels - 1):
                if proc != 0:  # because mesh init does not have a "same" gnn
                    self.process_steps.append(f"mesh{i}_mesh{i}_up_proc{proc}")
                    process_edge_feat_dims[
                        f"mesh{i}_mesh{i}_up_proc{proc}"
                    ] = mesh_same_dim
                self.process_steps.append(f"mesh{i}_mesh{i+1}_proc{proc}")
                process_edge_feat_dims[
                    f"mesh{i}_mesh{i+1}_proc{proc}"
                ] = mesh_up_dim
            for i in reversed(range(self.num_levels - 1)):
                if (
                    proc != self.num_processor_layers - 1
                ):  # because mesh read out does not have a "same" gnn
                    self.process_steps.append(
                        f"mesh{i+1}_mesh{i+1}_down_proc{proc}"
                    )
                    process_edge_feat_dims[
                        f"mesh{i+1}_mesh{i+1}_down_proc{proc}"
                    ] = mesh_same_dim
                self.process_steps.append(f"mesh{i+1}_mesh{i}_proc{proc}")
                process_edge_feat_dims[
                    f"mesh{i+1}_mesh{i}_proc{proc}"
                ] = mesh_down_dim
        self.process_steps.append("mesh0_mesh0_proclast")
        process_edge_feat_dims["mesh0_mesh0_proclast"] = mesh_same_dim
        self.process_steps.append("mesh0_grid")

        self.model = SequentialGNNModel(
            # dict with subgraph names as keys and node feature dimensions as
            # values
            subgraph_feat_dims,
            # list of subgraph names to deembed
            deembed_subgraph_dimension,
            # dict with process step names as keys and edge feature dimensions
            # as values
            process_edge_feat_dims,
            # for both encoders and interaction nets
            args.hidden_dim,
            # for both encoders and interaction nets
            args.hidden_layers,
        )

        # g2m_edge_index shape: torch.Size([2, 12716])
        # g2m_edge_index dim 0 min: 819, max: 8498
        # g2m_edge_index dim 1 min: 0, max: 728
        # mesh0 edge_index shape: torch.Size([2, 5512])
        # mesh0 edge_index dim 0 min: 0, max: 728
        # mesh1 edge_index shape: torch.Size([2, 544])
        # mesh1 edge_index dim 0 min: 729, max: 809
        # mesh2 edge_index shape: torch.Size([2, 40])
        # mesh2 edge_index dim 0 min: 810, max: 818

        self.edge_index = {
            "grid_mesh0": self.g2m_edge_index,
            "mesh0_grid": self.m2g_edge_index,
            **{
                f"mesh{i}_mesh{i+1}_proc{proc}": self.mesh_up_edge_index[i]
                for i in range(self.num_levels - 1)
                for proc in range(self.num_processor_layers)
            },
            **{
                f"mesh{i+1}_mesh{i}_proc{proc}": self.mesh_down_edge_index[i]
                for i in range(self.num_levels - 1)
                for proc in range(self.num_processor_layers)
            },
            **{
                f"mesh{i}_mesh{i}_up_proc{proc}": self.m2m_edge_index[i]
                for i in range(self.num_levels - 1)
                for proc in range(self.num_processor_layers)
            },
            **{
                f"mesh{i}_mesh{i}_down_proc{proc}": self.m2m_edge_index[i]
                for i in range(1, self.num_levels)
                for proc in range(self.num_processor_layers)
            },
            **{"mesh0_mesh0_proclast": self.m2m_edge_index[0]},
        }

        subgraph_start_index = [0] + list(np.cumsum(self.level_mesh_sizes))
        self.node_index = {
            **{
                f"mesh{i}": torch.arange(
                    subgraph_start_index[i], subgraph_start_index[i + 1]
                )
                for i in range(0, self.num_levels)
            },
            "grid": torch.arange(
                subgraph_start_index[-1],
                subgraph_start_index[-1] + self.num_grid_nodes,
            ),
        }

    def armodel_init(self, args, config, datastore):
        self.save_hyperparameters(ignore=["datastore"])
        self.args = args
        self._datastore = datastore
        num_state_vars = datastore.get_num_data_vars(category="state")
        num_forcing_vars = datastore.get_num_data_vars(category="forcing")
        # Load static features standardized
        da_static_features = datastore.get_dataarray(
            category="static", split=None, standardize=True
        )
        da_state_stats = datastore.get_standardization_dataarray(
            category="state"
        )
        da_boundary_mask = datastore.boundary_mask
        num_past_forcing_steps = args.num_past_forcing_steps
        num_future_forcing_steps = args.num_future_forcing_steps

        # Load static features for grid/data,
        self.register_buffer(
            "grid_static_features",
            torch.tensor(da_static_features.values, dtype=torch.float32),
            persistent=False,
        )

        state_stats = {
            "state_mean": torch.tensor(
                da_state_stats.state_mean.values, dtype=torch.float32
            ),
            "state_std": torch.tensor(
                da_state_stats.state_std.values, dtype=torch.float32
            ),
            # Note that the one-step-diff stats (diff_mean and diff_std) are
            # for differences computed on standardized data
            "diff_mean": torch.tensor(
                da_state_stats.state_diff_mean_standardized.values,
                dtype=torch.float32,
            ),
            "diff_std": torch.tensor(
                da_state_stats.state_diff_std_standardized.values,
                dtype=torch.float32,
            ),
        }

        for key, val in state_stats.items():
            self.register_buffer(key, val, persistent=False)

        state_feature_weights = get_state_feature_weighting(
            config=config, datastore=datastore
        )
        self.feature_weights = torch.tensor(
            state_feature_weights, dtype=torch.float32
        )

        # Double grid output dim. to also output std.-dev.
        self.output_std = bool(args.output_std)
        if self.output_std:
            # Pred. dim. in grid cell
            self.grid_output_dim = 2 * num_state_vars
        else:
            # Pred. dim. in grid cell
            self.grid_output_dim = num_state_vars
            # Store constant per-variable std.-dev. weighting
            # NOTE that this is the inverse of the multiplicative weighting
            # in wMSE/wMAE
            self.register_buffer(
                "per_var_std",
                self.diff_std / torch.sqrt(self.feature_weights),
                persistent=False,
            )

        # grid_dim from data + static
        (
            self.num_grid_nodes,
            grid_static_dim,
        ) = self.grid_static_features.shape

        self.grid_dim = (
            2 * self.grid_output_dim
            + grid_static_dim
            + num_forcing_vars
            * (num_past_forcing_steps + num_future_forcing_steps + 1)
        )

        # Instantiate loss function
        self.loss = metrics.get_metric(args.loss)

        boundary_mask = torch.tensor(
            da_boundary_mask.values, dtype=torch.float32
        ).unsqueeze(
            1
        )  # add feature dim

        self.register_buffer("boundary_mask", boundary_mask, persistent=False)
        # Pre-compute interior mask for use in loss function
        self.register_buffer(
            "interior_mask", 1.0 - self.boundary_mask, persistent=False
        )  # (num_grid_nodes, 1), 1 for non-border

        self.val_metrics = {
            "mse": [],
        }
        self.test_metrics = {
            "mse": [],
            "mae": [],
        }
        if self.output_std:
            self.test_metrics["output_std"] = []  # Treat as metric

        # For making restoring of optimizer state optional
        self.restore_opt = args.restore_opt

        # For example plotting
        self.n_example_pred = args.n_example_pred
        self.plotted_examples = 0

        # For storing spatial loss maps during evaluation
        self.spatial_loss_maps = []

    def basegraphmodel_init(self, args, config, datastore):
        # Load graph with static features
        # NOTE: (IMPORTANT!) mesh nodes MUST have the first
        # num_mesh_nodes indices,
        graph_dir_path = datastore.root_path / "graph" / args.graph
        self.hierarchical, graph_ldict = utils.load_graph(
            graph_dir_path=graph_dir_path
        )
        for name, attr_value in graph_ldict.items():
            # Make BufferLists module members and register tensors as buffers
            if isinstance(attr_value, torch.Tensor):
                self.register_buffer(name, attr_value, persistent=False)
            else:
                setattr(self, name, attr_value)

        # Specify dimensions of data
        self.num_mesh_nodes, _ = self.get_num_mesh()
        utils.rank_zero_print(
            f"Loaded graph with {self.num_grid_nodes + self.num_mesh_nodes} "
            f"nodes ({self.num_grid_nodes} grid, {self.num_mesh_nodes} mesh)"
        )

        # grid_dim from data + static
        self.g2m_edges, g2m_dim = self.g2m_features.shape
        self.m2g_edges, m2g_dim = self.m2g_features.shape

        # Define sub-models
        # Feature embedders for grid
        self.mlp_blueprint_end = [args.hidden_dim] * (args.hidden_layers + 1)
        self.grid_embedder = utils.make_mlp(
            [self.grid_dim] + self.mlp_blueprint_end
        )
        self.g2m_embedder = utils.make_mlp([g2m_dim] + self.mlp_blueprint_end)
        self.m2g_embedder = utils.make_mlp([m2g_dim] + self.mlp_blueprint_end)

        # GNNs
        # encoder
        self.g2m_gnn = InteractionNet(
            self.g2m_edge_index,
            args.hidden_dim,
            hidden_layers=args.hidden_layers,
            update_edges=False,
        )
        self.encoding_grid_mlp = utils.make_mlp(
            [args.hidden_dim] + self.mlp_blueprint_end
        )

        # decoder
        self.m2g_gnn = InteractionNet(
            self.m2g_edge_index,
            args.hidden_dim,
            hidden_layers=args.hidden_layers,
            update_edges=False,
        )

        # Output mapping (hidden_dim -> output_dim)
        self.output_map = utils.make_mlp(
            [args.hidden_dim] * (args.hidden_layers + 1)
            + [self.grid_output_dim],
            layer_norm=False,
        )  # No layer norm on this one

        # Compute indices and define clamping functions
        self.prepare_clamping_params(config, datastore)

    def basehigraphmodel_init(self, args, config, datastore):

        # Track number of nodes, edges on each level
        # Flatten lists for efficient embedding
        self.num_levels = len(self.mesh_static_features)

        # Number of mesh nodes at each level
        self.level_mesh_sizes = [
            mesh_feat.shape[0] for mesh_feat in self.mesh_static_features
        ]  # Needs as python list for later

        # Print some useful info
        utils.rank_zero_print("Loaded hierarchical graph with structure:")
        for level_index, level_mesh_size in enumerate(self.level_mesh_sizes):
            same_level_edges = self.m2m_features[level_index].shape[0]
            utils.rank_zero_print(
                f"level {level_index} - {level_mesh_size} nodes, "
                f"{same_level_edges} same-level edges"
            )

            if level_index < (self.num_levels - 1):
                up_edges = self.mesh_up_features[level_index].shape[0]
                down_edges = self.mesh_down_features[level_index].shape[0]
                utils.rank_zero_print(f"  {level_index}<->{level_index + 1}")
                utils.rank_zero_print(
                    f" - {up_edges} up edges, {down_edges} down edges"
                )
        # Embedders
        # Assume all levels have same static feature dimensionality
        mesh_dim = self.mesh_static_features[0].shape[1]
        mesh_same_dim = self.m2m_features[0].shape[1]
        mesh_up_dim = self.mesh_up_features[0].shape[1]
        mesh_down_dim = self.mesh_down_features[0].shape[1]

        # Separate mesh node embedders for each level
        self.mesh_embedders = nn.ModuleList(
            [
                utils.make_mlp([mesh_dim] + self.mlp_blueprint_end)
                for _ in range(self.num_levels)
            ]
        )
        self.mesh_same_embedders = nn.ModuleList(
            [
                utils.make_mlp([mesh_same_dim] + self.mlp_blueprint_end)
                for _ in range(self.num_levels)
            ]
        )
        self.mesh_up_embedders = nn.ModuleList(
            [
                utils.make_mlp([mesh_up_dim] + self.mlp_blueprint_end)
                for _ in range(self.num_levels - 1)
            ]
        )
        self.mesh_down_embedders = nn.ModuleList(
            [
                utils.make_mlp([mesh_down_dim] + self.mlp_blueprint_end)
                for _ in range(self.num_levels - 1)
            ]
        )

        # Instantiate GNNs
        # Init GNNs
        self.mesh_init_gnns = nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    args.hidden_dim,
                    hidden_layers=args.hidden_layers,
                )
                for edge_index in self.mesh_up_edge_index
            ]
        )

        # Read out GNNs
        self.mesh_read_gnns = nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    args.hidden_dim,
                    hidden_layers=args.hidden_layers,
                    update_edges=False,
                )
                for edge_index in self.mesh_down_edge_index
            ]
        )

    def hilam_init(self, args, config, datastore):
        # Make down GNNs, both for down edges and same level
        self.mesh_down_gnns = nn.ModuleList(
            [self.make_down_gnns(args) for _ in range(args.processor_layers)]
        )  # Nested lists (proc_steps, num_levels-1)
        self.mesh_down_same_gnns = nn.ModuleList(
            [self.make_same_gnns(args) for _ in range(args.processor_layers)]
        )  # Nested lists (proc_steps, num_levels)

        # Make up GNNs, both for up edges and same level
        self.mesh_up_gnns = nn.ModuleList(
            [self.make_up_gnns(args) for _ in range(args.processor_layers)]
        )  # Nested lists (proc_steps, num_levels-1)
        self.mesh_up_same_gnns = nn.ModuleList(
            [self.make_same_gnns(args) for _ in range(args.processor_layers)]
        )  # Nested lists (proc_steps, num_levels)

    # start of armodel
    def _create_dataarray_from_tensor(
        self,
        tensor: torch.Tensor,
        time: Union[int, List[int]],
        split: str,
        category: str,
    ) -> xr.DataArray:
        """
        Create an `xr.DataArray` from a tensor, with the correct dimensions and
        coordinates to match the datastore used by the model. This function in
        in effect is the inverse of what is returned by
        `WeatherDataset.__getitem__`.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to convert to a `xr.DataArray` with dimensions [time,
            grid_index, feature]. The tensor will be copied to the CPU if it is
            not already there.
        time : Union[int,List[int]]
            The time index or indices for the data, given as integers or a list
            of integers representing epoch time in nanoseconds. The ints will be
            copied to the CPU memory if they are not already there.
        split : str
            The split of the data, either 'train', 'val', or 'test'
        category : str
            The category of the data, either 'state' or 'forcing'
        """
        # TODO: creating an instance of WeatherDataset here on every call is
        # not how this should be done but whether WeatherDataset should be
        # provided to ARModel or where to put plotting still needs discussion
        weather_dataset = WeatherDataset(datastore=self._datastore, split=split)
        time = np.array(time.cpu(), dtype="datetime64[ns]")
        da = weather_dataset.create_dataarray_from_tensor(
            tensor=tensor, time=time, category=category
        )
        return da

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.args.lr, betas=(0.9, 0.95)
        )
        return opt

    @property
    def interior_mask_bool(self):
        """
        Get the interior mask as a boolean (N,) mask.
        """
        return self.interior_mask[:, 0].to(torch.bool)

    @staticmethod
    def expand_to_batch(x, batch_size):
        """
        Expand tensor with initial batch dimension
        """
        return x.unsqueeze(0).expand(batch_size, -1, -1)

    def common_step(self, batch):
        """
        Roll out prediction taking multiple autoregressive steps with model
        """
        (init_states, target_states, forcing_features, batch_times) = batch
        # init_states: (B, 2, num_grid_nodes, d_f)
        # forcing_features: (B, pred_steps, num_grid_nodes, d_static_f)
        # target_states: (B, pred_steps, num_grid_nodes, d_f)

        prev_prev_state = init_states[:, 0]
        prev_state = init_states[:, 1]
        prediction_list = []
        pred_std_list = []
        pred_steps = forcing_features.shape[1]

        for i in range(pred_steps):
            forcing = forcing_features[:, i]
            border_state = target_states[:, i]

            batch_size = prev_state.shape[0]

            # Create full grid node features of shape
            # (B, num_grid_nodes, grid_dim)
            grid_features = torch.cat(
                (
                    prev_state,
                    prev_prev_state,
                    forcing,
                    self.expand_to_batch(self.grid_static_features, batch_size),
                ),
                dim=-1,
            )

            # net_output = self.forward_process(grid_features)
            node_features_dict = {
                "grid": grid_features,
                **{
                    f"mesh{i}": self.mesh_static_features[i]
                    for i in range(self.num_levels)
                },
            }
            edge_features_dict = {
                "grid_mesh0": self.expand_to_batch(
                    self.g2m_features, batch_size
                ),  # aka g2m
                "mesh0_grid": self.expand_to_batch(
                    self.m2g_features, batch_size
                ),  # aka m2g
                **{
                    f"mesh{i}_mesh{i}_proc{proc}": self.expand_to_batch(
                        self.m2m_features[i], batch_size
                    )
                    for i in range(self.num_levels)
                    for proc in range(self.num_processor_layers)
                    if f"mesh{i}_mesh{i}_proc{proc}" in self.edge_index
                },
                **{
                    f"mesh{i}_mesh{i+1}_proc{proc}": self.expand_to_batch(
                        self.mesh_up_features[i], batch_size
                    )
                    for i in range(self.num_levels - 1)
                    for proc in range(self.num_processor_layers)
                    if f"mesh{i}_mesh{i+1}_proc{proc}" in self.edge_index
                },
                **{
                    f"mesh{i+1}_mesh{i}_proc{proc}": self.expand_to_batch(
                        self.mesh_down_features[i], batch_size
                    )
                    for i in range(self.num_levels - 1)
                    for proc in range(self.num_processor_layers)
                    if f"mesh{i+1}_mesh{i}_proc{proc}" in self.edge_index
                },
            }

            net_output = self.model(
                node_features_dict,
                edge_features_dict,
                self.node_index,
                self.edge_index,
                self.process_steps,
            )

            if self.output_std:
                pred_delta_mean, pred_std_raw = net_output.chunk(
                    2, dim=-1
                )  # both (B, num_grid_nodes, d_f)
                # NOTE: The predicted std. is not scaled in any way here
                # linter for some reason does not think softplus is callable
                # pylint: disable-next=not-callable
                pred_std = torch.nn.functional.softplus(pred_std_raw)
            else:
                pred_delta_mean = net_output
                pred_std = None

            # Rescale with one-step difference statistics
            rescaled_delta_mean = (
                pred_delta_mean * self.diff_std + self.diff_mean
            )

            # Clamp values to valid range
            # (also add the delta to the previous state)
            pred_state = self.get_clamped_new_state(
                rescaled_delta_mean, prev_state
            )

            # pred_state: (B, num_grid_nodes, d_f)
            # pred_std: (B, num_grid_nodes, d_f) or None

            # Overwrite border with true state
            new_state = (
                self.boundary_mask * border_state
                + self.interior_mask * pred_state
            )

            prediction_list.append(new_state)
            if self.output_std:
                pred_std_list.append(pred_std)

            # Update conditioning states
            prev_prev_state = prev_state
            prev_state = new_state

        prediction = torch.stack(
            prediction_list, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)

        if self.output_std:
            pred_std = torch.stack(
                pred_std_list, dim=1
            )  # (B, pred_steps, num_grid_nodes, d_f)
        else:
            pred_std = self.per_var_std  # (d_f,)

        # prediction: (B, pred_steps, num_grid_nodes, d_f)
        # pred_std: (B, pred_steps, num_grid_nodes, d_f) or (d_f,)
        return prediction, target_states, pred_std, batch_times

    def training_step(self, batch):
        """
        Train on single batch
        """
        prediction, target, pred_std, _ = self.common_step(batch)

        # Compute loss
        batch_loss = torch.mean(
            self.loss(
                prediction, target, pred_std, mask=self.interior_mask_bool
            )
        )  # mean over unrolled times and batch

        log_dict = {"train_loss": batch_loss}
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )
        return batch_loss

    def all_gather_cat(self, tensor_to_gather):
        """
        Gather tensors across all ranks, and concatenate across dim. 0 (instead
        of stacking in new dim. 0)

        tensor_to_gather: (d1, d2, ...), distributed over K ranks

        returns: (K*d1, d2, ...)
        """
        return self.all_gather(tensor_to_gather).flatten(0, 1)

    # newer lightning versions requires batch_idx argument, even if unused
    # pylint: disable-next=unused-argument
    def validation_step(self, batch, batch_idx):
        """
        Run validation on single batch
        """
        prediction, target, pred_std, _ = self.common_step(batch)

        time_step_loss = torch.mean(
            self.loss(
                prediction, target, pred_std, mask=self.interior_mask_bool
            ),
            dim=0,
        )  # (time_steps-1)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        val_log_dict = {
            f"val_loss_unroll{step}": time_step_loss[step - 1]
            for step in self.args.val_steps_to_log
            if step <= len(time_step_loss)
        }
        val_log_dict["val_mean_loss"] = mean_loss
        self.log_dict(
            val_log_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )

        # Store MSEs
        entry_mses = metrics.mse(
            prediction,
            target,
            pred_std,
            mask=self.interior_mask_bool,
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.val_metrics["mse"].append(entry_mses)

    def on_validation_epoch_end(self):
        """
        Compute val metrics at the end of val epoch
        """
        # Create error maps for all test metrics
        self.aggregate_and_plot_metrics(self.val_metrics, prefix="val")

        # Clear lists with validation metrics values
        for metric_list in self.val_metrics.values():
            metric_list.clear()

    # pylint: disable-next=unused-argument
    def test_step(self, batch, batch_idx):
        """
        Run test on single batch
        """
        # TODO Here batch_times can be used for plotting routines
        prediction, target, pred_std, batch_times = self.common_step(batch)
        # prediction: (B, pred_steps, num_grid_nodes, d_f) pred_std: (B,
        # pred_steps, num_grid_nodes, d_f) or (d_f,)

        time_step_loss = torch.mean(
            self.loss(
                prediction, target, pred_std, mask=self.interior_mask_bool
            ),
            dim=0,
        )  # (time_steps-1,)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        test_log_dict = {
            f"test_loss_unroll{step}": time_step_loss[step - 1]
            for step in self.args.val_steps_to_log
        }
        test_log_dict["test_mean_loss"] = mean_loss

        self.log_dict(
            test_log_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )

        # Compute all evaluation metrics for error maps Note: explicitly list
        # metrics here, as test_metrics can contain additional ones, computed
        # differently, but that should be aggregated on_test_epoch_end
        for metric_name in ("mse", "mae"):
            metric_func = metrics.get_metric(metric_name)
            batch_metric_vals = metric_func(
                prediction,
                target,
                pred_std,
                mask=self.interior_mask_bool,
                sum_vars=False,
            )  # (B, pred_steps, d_f)
            self.test_metrics[metric_name].append(batch_metric_vals)

        if self.output_std:
            # Store output std. per variable, spatially averaged
            mean_pred_std = torch.mean(
                pred_std[..., self.interior_mask_bool, :], dim=-2
            )  # (B, pred_steps, d_f)
            self.test_metrics["output_std"].append(mean_pred_std)

        # Save per-sample spatial loss for specific times
        spatial_loss = self.loss(
            prediction, target, pred_std, average_grid=False
        )  # (B, pred_steps, num_grid_nodes)
        log_spatial_losses = spatial_loss[
            :, [step - 1 for step in self.args.val_steps_to_log]
        ]
        self.spatial_loss_maps.append(log_spatial_losses)
        # (B, N_log, num_grid_nodes)

        # Plot example predictions (on rank 0 only)
        if (
            self.trainer.is_global_zero
            and self.plotted_examples < self.n_example_pred
        ):
            # Need to plot more example predictions
            n_additional_examples = min(
                prediction.shape[0],
                self.n_example_pred - self.plotted_examples,
            )

            self.plot_examples(
                batch,
                n_additional_examples,
                prediction=prediction,
                split="test",
            )

    def plot_examples(self, batch, n_examples, split, prediction=None):
        """
        Plot the first n_examples forecasts from batch

        batch: batch with data to plot corresponding forecasts for n_examples:
        number of forecasts to plot prediction: (B, pred_steps, num_grid_nodes,
        d_f), existing prediction.
            Generate if None.
        """
        if prediction is None:
            prediction, target, _, _ = self.common_step(batch)

        target = batch[1]
        time = batch[3]

        # Rescale to original data scale
        prediction_rescaled = prediction * self.state_std + self.state_mean
        target_rescaled = target * self.state_std + self.state_mean

        # Iterate over the examples
        for pred_slice, target_slice, time_slice in zip(
            prediction_rescaled[:n_examples],
            target_rescaled[:n_examples],
            time[:n_examples],
        ):
            # Each slice is (pred_steps, num_grid_nodes, d_f)
            self.plotted_examples += 1  # Increment already here

            da_prediction = self._create_dataarray_from_tensor(
                tensor=pred_slice,
                time=time_slice,
                split=split,
                category="state",
            ).unstack("grid_index")
            da_target = self._create_dataarray_from_tensor(
                tensor=target_slice,
                time=time_slice,
                split=split,
                category="state",
            ).unstack("grid_index")

            var_vmin = (
                torch.minimum(
                    pred_slice.flatten(0, 1).min(dim=0)[0],
                    target_slice.flatten(0, 1).min(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vmax = (
                torch.maximum(
                    pred_slice.flatten(0, 1).max(dim=0)[0],
                    target_slice.flatten(0, 1).max(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vranges = list(zip(var_vmin, var_vmax))

            # Iterate over prediction horizon time steps
            for t_i, _ in enumerate(zip(pred_slice, target_slice), start=1):
                # Create one figure per variable at this time step
                var_figs = [
                    vis.plot_prediction(
                        datastore=self._datastore,
                        title=f"{var_name} ({var_unit}), "
                        f"t={t_i} ({self._datastore.step_length * t_i} h)",
                        vrange=var_vrange,
                        da_prediction=da_prediction.isel(
                            state_feature=var_i, time=t_i - 1
                        ).squeeze(),
                        da_target=da_target.isel(
                            state_feature=var_i, time=t_i - 1
                        ).squeeze(),
                    )
                    for var_i, (var_name, var_unit, var_vrange) in enumerate(
                        zip(
                            self._datastore.get_vars_names("state"),
                            self._datastore.get_vars_units("state"),
                            var_vranges,
                        )
                    )
                ]

                example_i = self.plotted_examples

                for var_name, fig in zip(
                    self._datastore.get_vars_names("state"), var_figs
                ):

                    # We need treat logging images differently for different
                    # loggers. WANDB can log multiple images to the same key,
                    # while other loggers, as MLFlow, need unique keys for
                    # each image.
                    if isinstance(self.logger, pl.loggers.WandbLogger):
                        key = f"{var_name}_example_{example_i}"
                    else:
                        key = f"{var_name}_example"

                    if hasattr(self.logger, "log_image"):
                        self.logger.log_image(key=key, images=[fig], step=t_i)
                    else:
                        warnings.warn(
                            f"{self.logger} does not support image logging."
                        )

                plt.close(
                    "all"
                )  # Close all figs for this time step, saves memory

            # Save pred and target as .pt files
            torch.save(
                pred_slice.cpu(),
                os.path.join(
                    self.logger.save_dir,
                    f"example_pred_{self.plotted_examples}.pt",
                ),
            )
            torch.save(
                target_slice.cpu(),
                os.path.join(
                    self.logger.save_dir,
                    f"example_target_{self.plotted_examples}.pt",
                ),
            )

    def create_metric_log_dict(self, metric_tensor, prefix, metric_name):
        """
        Put together a dict with everything to log for one metric. Also saves
        plots as pdf and csv if using test prefix.

        metric_tensor: (pred_steps, d_f), metric values per time and variable
        prefix: string, prefix to use for logging metric_name: string, name of
        the metric

        Return: log_dict: dict with everything to log for given metric
        """
        log_dict = {}
        metric_fig = vis.plot_error_map(
            errors=metric_tensor,
            datastore=self._datastore,
        )
        full_log_name = f"{prefix}_{metric_name}"
        log_dict[full_log_name] = metric_fig

        if prefix == "test":
            # Save pdf
            metric_fig.savefig(
                os.path.join(self.logger.save_dir, f"{full_log_name}.pdf")
            )
            # Save errors also as csv
            np.savetxt(
                os.path.join(self.logger.save_dir, f"{full_log_name}.csv"),
                metric_tensor.cpu().numpy(),
                delimiter=",",
            )

        # Check if metrics are watched, log exact values for specific vars
        var_names = self._datastore.get_vars_names(category="state")
        if full_log_name in self.args.metrics_watch:
            for var_i, timesteps in self.args.var_leads_metrics_watch.items():
                var_name = var_names[var_i]
                for step in timesteps:
                    key = f"{full_log_name}_{var_name}_step_{step}"
                    log_dict[key] = metric_tensor[step - 1, var_i]

        return log_dict

    def aggregate_and_plot_metrics(self, metrics_dict, prefix):
        """
        Aggregate and create error map plots for all metrics in metrics_dict

        metrics_dict: dictionary with metric_names and list of tensors
            with step-evals.
        prefix: string, prefix to use for logging
        """
        log_dict = {}
        for metric_name, metric_val_list in metrics_dict.items():
            metric_tensor = self.all_gather_cat(
                torch.cat(metric_val_list, dim=0)
            )  # (N_eval, pred_steps, d_f)

            if self.trainer.is_global_zero:
                metric_tensor_averaged = torch.mean(metric_tensor, dim=0)
                # (pred_steps, d_f)

                # Take square root after all averaging to change MSE to RMSE
                if "mse" in metric_name:
                    metric_tensor_averaged = torch.sqrt(metric_tensor_averaged)
                    metric_name = metric_name.replace("mse", "rmse")

                # NOTE: we here assume rescaling for all metrics is linear
                metric_rescaled = metric_tensor_averaged * self.state_std
                # (pred_steps, d_f)
                log_dict.update(
                    self.create_metric_log_dict(
                        metric_rescaled, prefix, metric_name
                    )
                )

        # Ensure that log_dict has structure for
        # logging as dict(str, plt.Figure)
        assert all(
            isinstance(key, str) and isinstance(value, plt.Figure)
            for key, value in log_dict.items()
        )

        if self.trainer.is_global_zero and not self.trainer.sanity_checking:

            current_epoch = self.trainer.current_epoch

            for key, figure in log_dict.items():
                # For other loggers than wandb, add epoch to key.
                # Wandb can log multiple images to the same key, while other
                # loggers, such as MLFlow need unique keys for each image.
                if not isinstance(self.logger, pl.loggers.WandbLogger):
                    key = f"{key}-{current_epoch}"

                if hasattr(self.logger, "log_image"):
                    self.logger.log_image(key=key, images=[figure])

            plt.close("all")  # Close all figs

    def on_test_epoch_end(self):
        """
        Compute test metrics and make plots at the end of test epoch. Will
        gather stored tensors and perform plotting and logging on rank 0.
        """
        # Create error maps for all test metrics
        self.aggregate_and_plot_metrics(self.test_metrics, prefix="test")

        # Plot spatial loss maps
        spatial_loss_tensor = self.all_gather_cat(
            torch.cat(self.spatial_loss_maps, dim=0)
        )  # (N_test, N_log, num_grid_nodes)
        if self.trainer.is_global_zero:
            mean_spatial_loss = torch.mean(
                spatial_loss_tensor, dim=0
            )  # (N_log, num_grid_nodes)

            loss_map_figs = [
                vis.plot_spatial_error(
                    error=loss_map,
                    datastore=self._datastore,
                    title=f"Test loss, t={t_i} "
                    f"({self._datastore.step_length * t_i} h)",
                )
                for t_i, loss_map in zip(
                    self.args.val_steps_to_log, mean_spatial_loss
                )
            ]

            # log all to same key, sequentially
            for i, fig in enumerate(loss_map_figs):
                key = "test_loss"
                if not isinstance(self.logger, pl.loggers.WandbLogger):
                    key = f"{key}_{i}"
                if hasattr(self.logger, "log_image"):
                    self.logger.log_image(key=key, images=[fig])

            # also make without title and save as pdf
            pdf_loss_map_figs = [
                vis.plot_spatial_error(
                    error=loss_map, datastore=self._datastore
                )
                for loss_map in mean_spatial_loss
            ]
            pdf_loss_maps_dir = os.path.join(
                self.logger.save_dir, "spatial_loss_maps"
            )
            os.makedirs(pdf_loss_maps_dir, exist_ok=True)
            for t_i, fig in zip(self.args.val_steps_to_log, pdf_loss_map_figs):
                fig.savefig(os.path.join(pdf_loss_maps_dir, f"loss_t{t_i}.pdf"))
            # save mean spatial loss as .pt file also
            torch.save(
                mean_spatial_loss.cpu(),
                os.path.join(self.logger.save_dir, "mean_spatial_loss.pt"),
            )

        self.spatial_loss_maps.clear()

    def on_load_checkpoint(self, checkpoint):
        """
        Perform any changes to state dict before loading checkpoint
        """
        loaded_state_dict = checkpoint["state_dict"]

        # Fix for loading older models after IneractionNet refactoring, where
        # the grid MLP was moved outside the encoder InteractionNet class
        if "g2m_gnn.grid_mlp.0.weight" in loaded_state_dict:
            replace_keys = list(
                filter(
                    lambda key: key.startswith("g2m_gnn.grid_mlp"),
                    loaded_state_dict.keys(),
                )
            )
            for old_key in replace_keys:
                new_key = old_key.replace(
                    "g2m_gnn.grid_mlp", "encoding_grid_mlp"
                )
                loaded_state_dict[new_key] = loaded_state_dict[old_key]
                del loaded_state_dict[old_key]
        if not self.restore_opt:
            opt = self.configure_optimizers()
            checkpoint["optimizer_states"] = [opt.state_dict()]

    # start of base_graph_model
    def prepare_clamping_params(
        self, config: NeuralLAMConfig, datastore: BaseDatastore
    ):
        """
        Prepare parameters for clamping predicted values to valid range
        """

        # Read configs
        state_feature_names = datastore.get_vars_names(category="state")
        lower_lims = config.training.output_clamping.lower
        upper_lims = config.training.output_clamping.upper

        # Check that limits in config are for valid features
        unknown_features_lower = set(lower_lims.keys()) - set(
            state_feature_names
        )
        unknown_features_upper = set(upper_lims.keys()) - set(
            state_feature_names
        )
        if unknown_features_lower or unknown_features_upper:
            raise ValueError(
                "State feature limits were provided for unknown features: "
                f"{unknown_features_lower.union(unknown_features_upper)}"
            )

        # Constant parameters for clamping
        sigmoid_sharpness = 1
        softplus_sharpness = 1
        sigmoid_center = 0
        softplus_center = 0

        normalize_clamping_lim = (
            lambda x, feature_idx: (x - self.state_mean[feature_idx])
            / self.state_std[feature_idx]
        )

        # Check which clamping functions to use for each feature
        sigmoid_lower_upper_idx = []
        sigmoid_lower_lims = []
        sigmoid_upper_lims = []

        softplus_lower_idx = []
        softplus_lower_lims = []

        softplus_upper_idx = []
        softplus_upper_lims = []

        for feature_idx, feature in enumerate(state_feature_names):
            if feature in lower_lims and feature in upper_lims:
                assert (
                    lower_lims[feature] < upper_lims[feature]
                ), f'Invalid clamping limits for feature "{feature}",\
                     lower: {lower_lims[feature]}, larger than\
                     upper: {upper_lims[feature]}'
                sigmoid_lower_upper_idx.append(feature_idx)
                sigmoid_lower_lims.append(
                    normalize_clamping_lim(lower_lims[feature], feature_idx)
                )
                sigmoid_upper_lims.append(
                    normalize_clamping_lim(upper_lims[feature], feature_idx)
                )
            elif feature in lower_lims and feature not in upper_lims:
                softplus_lower_idx.append(feature_idx)
                softplus_lower_lims.append(
                    normalize_clamping_lim(lower_lims[feature], feature_idx)
                )
            elif feature not in lower_lims and feature in upper_lims:
                softplus_upper_idx.append(feature_idx)
                softplus_upper_lims.append(
                    normalize_clamping_lim(upper_lims[feature], feature_idx)
                )

        self.register_buffer(
            "sigmoid_lower_lims", torch.tensor(sigmoid_lower_lims)
        )
        self.register_buffer(
            "sigmoid_upper_lims", torch.tensor(sigmoid_upper_lims)
        )
        self.register_buffer(
            "softplus_lower_lims", torch.tensor(softplus_lower_lims)
        )
        self.register_buffer(
            "softplus_upper_lims", torch.tensor(softplus_upper_lims)
        )

        self.register_buffer(
            "clamp_lower_upper_idx", torch.tensor(sigmoid_lower_upper_idx)
        )
        self.register_buffer(
            "clamp_lower_idx", torch.tensor(softplus_lower_idx)
        )
        self.register_buffer(
            "clamp_upper_idx", torch.tensor(softplus_upper_idx)
        )

        # Define clamping functions
        self.clamp_lower_upper = lambda x: (
            self.sigmoid_lower_lims
            + (self.sigmoid_upper_lims - self.sigmoid_lower_lims)
            * torch.sigmoid(sigmoid_sharpness * (x - sigmoid_center))
        )
        self.clamp_lower = lambda x: (
            self.softplus_lower_lims
            + torch.nn.functional.softplus(
                x - softplus_center, beta=softplus_sharpness
            )
        )
        self.clamp_upper = lambda x: (
            self.softplus_upper_lims
            - torch.nn.functional.softplus(
                softplus_center - x, beta=softplus_sharpness
            )
        )

        self.inverse_clamp_lower_upper = lambda x: (
            sigmoid_center
            + utils.inverse_sigmoid(
                (x - self.sigmoid_lower_lims)
                / (self.sigmoid_upper_lims - self.sigmoid_lower_lims)
            )
            / sigmoid_sharpness
        )
        self.inverse_clamp_lower = lambda x: (
            utils.inverse_softplus(
                x - self.softplus_lower_lims, beta=softplus_sharpness
            )
            + softplus_center
        )
        self.inverse_clamp_upper = lambda x: (
            -utils.inverse_softplus(
                self.softplus_upper_lims - x, beta=softplus_sharpness
            )
            + softplus_center
        )

    def get_clamped_new_state(self, state_delta, prev_state):
        """
        Clamp prediction to valid range supplied in config
        Returns the clamped new state after adding delta to original state

        Instead of the new state being computed as
        $X_{t+1} = X_t + \\delta = X_t + model(\\{X_t,X_{t-1},...\\}, forcing)$
        The clamped values will be
        $f(f^{-1}(X_t) + model(\\{X_t, X_{t-1},... \\}, forcing))$
        Which means the model will learn to output values in the range of the
        inverse clamping function

        state_delta: (B, num_grid_nodes, feature_dim)
        prev_state: (B, num_grid_nodes, feature_dim)
        """

        # Assign new state, but overwrite clamped values of each type later
        new_state = prev_state + state_delta

        # Sigmoid/logistic clamps between ]a,b[
        if self.clamp_lower_upper_idx.numel() > 0:
            idx = self.clamp_lower_upper_idx

            new_state[:, :, idx] = self.clamp_lower_upper(
                self.inverse_clamp_lower_upper(prev_state[:, :, idx])
                + state_delta[:, :, idx]
            )

        # Softplus clamps between ]a,infty[
        if self.clamp_lower_idx.numel() > 0:
            idx = self.clamp_lower_idx

            new_state[:, :, idx] = self.clamp_lower(
                self.inverse_clamp_lower(prev_state[:, :, idx])
                + state_delta[:, :, idx]
            )

        # Softplus clamps between ]-infty,b[
        if self.clamp_upper_idx.numel() > 0:
            idx = self.clamp_upper_idx

            new_state[:, :, idx] = self.clamp_upper(
                self.inverse_clamp_upper(prev_state[:, :, idx])
                + state_delta[:, :, idx]
            )

        return new_state

    # def predict_step(self, prev_state, prev_prev_state, forcing):
    #     """
    #     Step state one step ahead using prediction model,
    #     X_{t-1}, X_t -> X_t+1
    #     prev_state: (B, num_grid_nodes, feature_dim), X_t
    #     prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
    #     forcing: (B, num_grid_nodes, forcing_dim)
    #     """
    #     batch_size = prev_state.shape[0]

    #     # Create full grid node features of shape
    #       (B, num_grid_nodes, grid_dim)
    #     grid_features = torch.cat(
    #         (
    #             prev_state,
    #             prev_prev_state,
    #             forcing,
    #             self.expand_to_batch(self.grid_static_features, batch_size),
    #         ),
    #         dim=-1,
    #     )

    #     net_output = self.forward(grid_features)

    #     if self.output_std:
    #         pred_delta_mean, pred_std_raw = net_output.chunk(
    #             2, dim=-1
    #         )  # both (B, num_grid_nodes, d_f)
    #         # NOTE: The predicted std. is not scaled in any way here
    #         # linter for some reason does not think softplus is callable
    #         # pylint: disable-next=not-callable
    #         pred_std = torch.nn.functional.softplus(pred_std_raw)
    #     else:
    #         pred_delta_mean = net_output
    #         pred_std = None

    #     # Rescale with one-step difference statistics
    #     rescaled_delta_mean = pred_delta_mean * self.diff_std + self.diff_mean

    #     # Clamp values to valid range
    #     # (also add the delta to the previous state)
    #     new_state = self.get_clamped_new_state(
    # rescaled_delta_mean, prev_state)

    #     return new_state, pred_std

    # start of base_hi_graph_model
    def get_num_mesh(self):
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding
        """
        num_mesh_nodes = sum(
            node_feat.shape[0] for node_feat in self.mesh_static_features
        )
        num_mesh_nodes_ignore = (
            num_mesh_nodes - self.mesh_static_features[0].shape[0]
        )
        return num_mesh_nodes, num_mesh_nodes_ignore

    def embedd_mesh_nodes(self):
        """
        Embed static mesh features
        This embeds only bottom level, rest is done at beginning of
        processing step
        Returns tensor of shape (num_mesh_nodes[0], d_h)
        """
        return self.mesh_embedders[0](self.mesh_static_features[0])

    def process_step(self, mesh_rep):
        """
        Process step of embedd-process-decode framework
        Processes the representation on the mesh, possible in multiple steps

        mesh_rep: has shape (B, num_mesh_nodes, d_h)
        Returns mesh_rep: (B, num_mesh_nodes, d_h)
        """
        batch_size = mesh_rep.shape[0]

        # EMBED REMAINING MESH NODES (levels >= 1) -
        # Create list of mesh node representations for each level,
        # each of size (B, num_mesh_nodes[l], d_h)
        mesh_rep_levels = [mesh_rep] + [
            self.expand_to_batch(emb(node_static_features), batch_size)
            for emb, node_static_features in zip(
                list(self.mesh_embedders)[1:],
                list(self.mesh_static_features)[1:],
            )
        ]

        # - EMBED EDGES -
        # Embed edges, expand with batch dimension
        mesh_same_rep = [
            self.expand_to_batch(emb(edge_feat), batch_size)
            for emb, edge_feat in zip(
                self.mesh_same_embedders, self.m2m_features
            )
        ]
        mesh_up_rep = [
            self.expand_to_batch(emb(edge_feat), batch_size)
            for emb, edge_feat in zip(
                self.mesh_up_embedders, self.mesh_up_features
            )
        ]
        mesh_down_rep = [
            self.expand_to_batch(emb(edge_feat), batch_size)
            for emb, edge_feat in zip(
                self.mesh_down_embedders, self.mesh_down_features
            )
        ]

        # - MESH INIT. -
        # Let level_l go from 1 to L
        for level_l, gnn in enumerate(self.mesh_init_gnns, start=1):
            # Extract representations
            send_node_rep = mesh_rep_levels[
                level_l - 1
            ]  # (B, num_mesh_nodes[l-1], d_h)
            rec_node_rep = mesh_rep_levels[
                level_l
            ]  # (B, num_mesh_nodes[l], d_h)
            edge_rep = mesh_up_rep[level_l - 1]

            # Apply GNN
            new_node_rep, new_edge_rep = gnn(
                send_node_rep, rec_node_rep, edge_rep
            )

            # Update node and edge vectors in lists
            mesh_rep_levels[
                level_l
            ] = new_node_rep  # (B, num_mesh_nodes[l], d_h)
            mesh_up_rep[level_l - 1] = new_edge_rep  # (B, M_up[l-1], d_h)

        # - PROCESSOR -
        mesh_rep_levels, _, _, mesh_down_rep = self.hi_processor_step(
            mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep
        )

        # - MESH READ OUT. -
        # Let level_l go from L-1 to 0
        for level_l, gnn in zip(
            range(self.num_levels - 2, -1, -1), reversed(self.mesh_read_gnns)
        ):
            # Extract representations
            send_node_rep = mesh_rep_levels[
                level_l + 1
            ]  # (B, num_mesh_nodes[l+1], d_h)
            rec_node_rep = mesh_rep_levels[
                level_l
            ]  # (B, num_mesh_nodes[l], d_h)
            edge_rep = mesh_down_rep[level_l]

            # Apply GNN
            new_node_rep = gnn(send_node_rep, rec_node_rep, edge_rep)

            # Update node and edge vectors in lists
            mesh_rep_levels[
                level_l
            ] = new_node_rep  # (B, num_mesh_nodes[l], d_h)

        # Return only bottom level representation
        return mesh_rep_levels[0]  # (B, num_mesh_nodes[0], d_h)

    # start of hi_lam
    def make_same_gnns(self, args):
        """
        Make intra-level GNNs.
        """
        return nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    args.hidden_dim,
                    hidden_layers=args.hidden_layers,
                )
                for edge_index in self.m2m_edge_index
            ]
        )

    def make_up_gnns(self, args):
        """
        Make GNNs for processing steps up through the hierarchy.
        """
        return nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    args.hidden_dim,
                    hidden_layers=args.hidden_layers,
                )
                for edge_index in self.mesh_up_edge_index
            ]
        )

    def make_down_gnns(self, args):
        """
        Make GNNs for processing steps down through the hierarchy.
        """
        return nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    args.hidden_dim,
                    hidden_layers=args.hidden_layers,
                )
                for edge_index in self.mesh_down_edge_index
            ]
        )

    def mesh_down_step(
        self,
        mesh_rep_levels,
        mesh_same_rep,
        mesh_down_rep,
        down_gnns,
        same_gnns,
    ):
        """
        Run down-part of vertical processing, sequentially alternating between
        processing using down edges and same-level edges.
        """
        # Run same level processing on level L
        mesh_rep_levels[-1], mesh_same_rep[-1] = same_gnns[-1](
            mesh_rep_levels[-1], mesh_rep_levels[-1], mesh_same_rep[-1]
        )

        # Let level_l go from L-1 to 0
        for level_l, down_gnn, same_gnn in zip(
            range(self.num_levels - 2, -1, -1),
            reversed(down_gnns),
            reversed(same_gnns[:-1]),
        ):
            # Extract representations
            send_node_rep = mesh_rep_levels[
                level_l + 1
            ]  # (B, N_mesh[l+1], d_h)
            rec_node_rep = mesh_rep_levels[level_l]  # (B, N_mesh[l], d_h)
            down_edge_rep = mesh_down_rep[level_l]
            same_edge_rep = mesh_same_rep[level_l]

            # Apply down GNN
            new_node_rep, mesh_down_rep[level_l] = down_gnn(
                send_node_rep, rec_node_rep, down_edge_rep
            )

            # Run same level processing on level l
            mesh_rep_levels[level_l], mesh_same_rep[level_l] = same_gnn(
                new_node_rep, new_node_rep, same_edge_rep
            )
            # (B, N_mesh[l], d_h) and (B, M_same[l], d_h)

        return mesh_rep_levels, mesh_same_rep, mesh_down_rep

    def mesh_up_step(
        self, mesh_rep_levels, mesh_same_rep, mesh_up_rep, up_gnns, same_gnns
    ):
        """
        Run up-part of vertical processing, sequentially alternating between
        processing using up edges and same-level edges.
        """

        # Run same level processing on level 0
        mesh_rep_levels[0], mesh_same_rep[0] = same_gnns[0](
            mesh_rep_levels[0], mesh_rep_levels[0], mesh_same_rep[0]
        )

        # Let level_l go from 1 to L
        for level_l, (up_gnn, same_gnn) in enumerate(
            zip(up_gnns, same_gnns[1:]), start=1
        ):
            # Extract representations
            send_node_rep = mesh_rep_levels[
                level_l - 1
            ]  # (B, N_mesh[l-1], d_h)
            rec_node_rep = mesh_rep_levels[level_l]  # (B, N_mesh[l], d_h)
            up_edge_rep = mesh_up_rep[level_l - 1]
            same_edge_rep = mesh_same_rep[level_l]

            # Apply up GNN
            new_node_rep, mesh_up_rep[level_l - 1] = up_gnn(
                send_node_rep, rec_node_rep, up_edge_rep
            )
            # (B, N_mesh[l], d_h) and (B, M_up[l-1], d_h)

            # Run same level processing on level l
            mesh_rep_levels[level_l], mesh_same_rep[level_l] = same_gnn(
                new_node_rep, new_node_rep, same_edge_rep
            )
            # (B, N_mesh[l], d_h) and (B, M_same[l], d_h)

        return mesh_rep_levels, mesh_same_rep, mesh_up_rep

    def hi_processor_step(
        self, mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep
    ):
        """
        Internal processor step of hierarchical graph models.
        Between mesh init and read out.

        Each input is list with representations, each with shape

        mesh_rep_levels: (B, N_mesh[l], d_h)
        mesh_same_rep: (B, M_same[l], d_h)
        mesh_up_rep: (B, M_up[l -> l+1], d_h)
        mesh_down_rep: (B, M_down[l <- l+1], d_h)

        Returns same lists
        """
        for down_gnns, down_same_gnns, up_gnns, up_same_gnns in zip(
            self.mesh_down_gnns,
            self.mesh_down_same_gnns,
            self.mesh_up_gnns,
            self.mesh_up_same_gnns,
        ):
            # Down
            mesh_rep_levels, mesh_same_rep, mesh_down_rep = self.mesh_down_step(
                mesh_rep_levels,
                mesh_same_rep,
                mesh_down_rep,
                down_gnns,
                down_same_gnns,
            )

            # Up
            mesh_rep_levels, mesh_same_rep, mesh_up_rep = self.mesh_up_step(
                mesh_rep_levels,
                mesh_same_rep,
                mesh_up_rep,
                up_gnns,
                up_same_gnns,
            )

        # NOTE: We return all, even though only down edges really are used
        # later
        return mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep

    def forward_process(self, grid_features):
        """
        performs a single pass through the encoders, processing gnns and
        decoders
        """
        # input for instantiation: node names and input dimensions for
        # encoders, name and order of edge sets for interaction nets, name of
        # node sets to return

        # input: prev states, forcing, static node and edge features, adjecency
        # matrix for each step of message passing.
        # Or be agnostic as to static/forcing/state?

        # embed physical grid features (prev states, forcing, static features
        # (loop over input grids?))
        # embed computational mesh node static features
        # (loop over mesh levels)
        # embed edge static features
        # (loop over mesh level connections up/down/same)
        # pass messages (loop over mesh level connections)
        # deembed physical grid node features

        # return deembedded values

        batch_size = grid_features.shape[0]

        # Embed all features
        grid_emb = self.grid_embedder(grid_features)  # (B, num_grid_nodes, d_h)

        # Embed constant features
        g2m_emb = self.g2m_embedder(self.g2m_features)  # (M_g2m, d_h)
        m2g_emb = self.m2g_embedder(self.m2g_features)  # (M_m2g, d_h)
        mesh_emb = self.embedd_mesh_nodes()

        # Map from grid to mesh
        mesh_emb_expanded = self.expand_to_batch(
            mesh_emb, batch_size
        )  # (B, num_mesh_nodes, d_h)
        g2m_emb_expanded = self.expand_to_batch(g2m_emb, batch_size)

        # This also splits representation into grid and mesh
        mesh_rep = self.g2m_gnn(
            grid_emb, mesh_emb_expanded, g2m_emb_expanded
        )  # (B, num_mesh_nodes, d_h)
        # Also MLP with residual for grid representation
        grid_rep = grid_emb + self.encoding_grid_mlp(
            grid_emb
        )  # (B, num_grid_nodes, d_h)

        # Run processor step
        mesh_rep = self.process_step(mesh_rep)

        # Map back from mesh to grid
        m2g_emb_expanded = self.expand_to_batch(m2g_emb, batch_size)
        grid_rep = self.m2g_gnn(
            mesh_rep, grid_rep, m2g_emb_expanded
        )  # (B, num_grid_nodes, d_h)

        # Map to output dimension, only for grid
        net_output = self.output_map(
            grid_rep
        )  # (B, num_grid_nodes, d_grid_out)

        return net_output


# input for instantiation: node names and input dimensions for encoders, name
# and order of edge sets for interaction nets, name of node sets to return

# input: prev states, forcing, static node and edge features, adjecency matrix
# for each step of message passing. Or be agnostic as to static/forcing/state?

# embed physical grid features (prev states, forcing, static features
# (loop over input grids?))
# embed computational mesh node static features (loop over mesh levels)
# embed edge static features (loop over mesh level connections up/down/same)
# pass messages (loop over mesh level connections)
# deembed physical grid node features

# return deembedded values


class SequentialGNNModel(torch.nn.Module):
    """ """

    def __init__(
        self,
        # dict with subgraph names as keys and node feature dimensions as
        # values
        subgraph_feat_dims,
        # dict with subgraph names to deembed as keys and output dim as
        # values
        deembed_subgraph_dims,
        # dict with process step names as keys and edge feature dimensions as
        # values
        process_edge_feat_dims,
        # for both encoders and interaction nets
        hidden_dim,
        # for both encoders and interaction nets
        hidden_layers,
    ):
        super().__init__()

        # Initialize dicts for submodels
        self.node_embedders = torch.nn.ModuleDict()
        self.node_deembedders = torch.nn.ModuleDict()
        self.edge_embedders = torch.nn.ModuleDict()
        self.process_step_gnns = torch.nn.ModuleDict()
        self.hidden_dim = hidden_dim

        mlp_blueprint_end = [self.hidden_dim] * (hidden_layers + 1)

        # Instantiate node embedders
        for subgraph_name, node_feat_dim in subgraph_feat_dims.items():
            self.node_embedders[subgraph_name] = utils.make_mlp(
                [node_feat_dim] + mlp_blueprint_end
            )

        # Instantiate node deembedders
        for subgraph_name, node_feat_dim in deembed_subgraph_dims.items():
            self.node_deembedders[subgraph_name] = utils.make_mlp(
                mlp_blueprint_end + [node_feat_dim]
            )

        # Instantiate edge embedders and interaction nets
        for step_name, edge_feat_dim in process_edge_feat_dims.items():
            self.process_step_gnns[step_name] = InteractionNetNew(
                self.hidden_dim,
                hidden_layers=hidden_layers,
                update_edges=True,
            )

            self.edge_embedders[step_name] = utils.make_mlp(
                [edge_feat_dim] + mlp_blueprint_end
            )

    def forward(
        self,
        node_features,
        edge_features,
        node_index,
        edge_index,
        process_steps,
    ):

        batch_size = list(node_features.values())[0].shape[0]
        num_nodes = max(
            [
                node_index[subgraph_name].max() + 1
                for subgraph_name in node_index.keys()
            ]
        )

        # Get dtype and device from model parameters
        # dtype = next(self.parameters()).dtype
        dtype = torch.bfloat16
        device = next(self.parameters()).device

        node_embeddings = torch.zeros(
            (batch_size, num_nodes, self.hidden_dim), dtype=dtype, device=device
        )
        edge_embeddings = {}
        node_out_features = {}

        # Embed nodes
        for subgraph_name, node_feat in node_features.items():
            node_embeddings[
                :, node_index[subgraph_name], :
            ] = self.node_embedders[subgraph_name](node_feat)

        # Embed edges
        for edge_set_name, edge_feat in edge_features.items():
            edge_embeddings[edge_set_name] = self.edge_embedders[edge_set_name](
                edge_feat
            )

        # # TODO how to handle this
        # # MLP with residual for grid representation
        # for subgraph_name in self.node_deembedders.keys():
        #     res = self.residual_mlp(node_embeddings[subgraph_name])
        #     # this value needs to be added to the node embeddings before the
        #  final process step (i.e. mesh2grid)
        # # Also MLP with residual for grid representation
        # # grid_rep = grid_emb + self.encoding_grid_mlp(
        # #     grid_emb
        # # )

        # Process steps
        for step_name in process_steps:
            # print(edge_index[step_name].shape)
            (
                node_embeddings,
                edge_embeddings[step_name],
            ) = self.process_step_gnns[step_name](
                node_embeddings,
                edge_embeddings[step_name],
                edge_index[step_name],
            )

        # Deembed output nodes
        for subgraph_name, deembedder in self.node_deembedders.items():
            node_out_features[subgraph_name] = deembedder(
                node_embeddings[subgraph_name]
            )

        return node_out_features
