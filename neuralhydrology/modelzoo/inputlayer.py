import logging
from typing import Dict, Optional, Union, Tuple

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.fc import FC
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)

_EMBEDDING_TYPES = ['full_model', 'hindcast', 'forecast']


class InputLayer(nn.Module):
    """Input layer to preprocess static and dynamic inputs.

    This module provides optional embedding of dynamic and static inputs. If ``dynamic_embeddings`` or
    ``static_embeddings`` are specified as dictionaries in the config, a fully-connected embedding network will be
    prepended to the timeseries model. The dictionaries have the following keys:

    - ``type`` (default 'fc'): Type of the embedding net. Currently, only 'fc' for fully-connected net is supported.
    - ``hiddens``: List of integers that define the number of neurons per layer in the fully connected network.
      The last number is the number of output neurons. Must have at least length one.
    - ``activation`` (default 'tanh'): activation function of the network. Supported values are 'tanh', 'sigmoid',
      'linear'. The activation function is not applied to the output neurons, which always have a linear activation
      function. An activation function for the output neurons has to be applied in the main model class.
    - ``dropout`` (default 0.0): Dropout rate applied to the embedding network.

    Note that this module does not support multi-frequency runs.

    Parameters
    ----------
    cfg : Config
        The run configuration
    """

    def __init__(self, cfg: Config, embedding_type: str = 'full_model'):
        super(InputLayer, self).__init__()

        if embedding_type not in _EMBEDDING_TYPES:
            raise ValueError(
                f'Embedding type {embedding_type} is not recognized. '
                f'Must be one of: {_EMBEDDING_TYPES}.'
            )
        self.forecast_length = cfg.forecast_seq_length
        self.hindcast_length = len(cfg.hindcast_inputs)
        self.forecast_overlap = cfg.forecast_overlap

        self.embedding_type = embedding_type
        if embedding_type == 'full_model':
            dynamic_inputs = cfg.dynamic_inputs
        elif embedding_type == 'forecast':
            dynamic_inputs = cfg.forecast_inputs
        elif embedding_type == 'hindcast':
            dynamic_inputs = cfg.hindcast_inputs

        if isinstance(dynamic_inputs, dict):
            frequencies = list(dynamic_inputs.keys())
            if len(frequencies) > 1:
                raise ValueError('InputLayer only supports single-frequency data')
            dynamics_input_size = len(dynamic_inputs[frequencies[0]])
        else:
            dynamics_input_size = len(dynamic_inputs)

        self._num_autoregression_inputs = 0
        if cfg.autoregressive_inputs:
            self._num_autoregression_inputs = len(cfg.autoregressive_inputs)

        statics_input_size = len(cfg.static_attributes + cfg.hydroatlas_attributes + cfg.evolving_attributes)
        if cfg.use_basin_id_encoding:
            statics_input_size += cfg.number_of_basins

        self.statics_embedding, self.statics_output_size = \
            self._get_embedding_net(cfg.statics_embedding, statics_input_size, 'statics')
        self.dynamics_embedding, self.dynamics_output_size = \
            self._get_embedding_net(cfg.dynamics_embedding, dynamics_input_size, 'dynamics')

        if cfg.statics_embedding is None:
            self.statics_embedding_p_dropout = 0.0  # if net has no statics dropout we treat is as zero
        else:
            self.statics_embedding_p_dropout = cfg.statics_embedding['dropout']
        if cfg.dynamics_embedding is None:
            self.dynamics_embedding_p_dropout = 0.0  # if net has no dynamics dropout we treat is as zero
        else:
            self.dynamics_embedding_p_dropout = cfg.dynamics_embedding['dropout']

        self.output_size = self.dynamics_output_size + self.statics_output_size + self._num_autoregression_inputs
        if cfg.head.lower() == "umal":
            self.output_size += 1

    @staticmethod
    def _get_embedding_net(embedding_spec: Optional[dict], input_size: int, purpose: str) -> Tuple[nn.Module, int]:
        """Get an embedding net following the passed specifications.

        If the `embedding_spec` is None, the returned embedding net will be the identity function.

        Parameters
        ----------
        embedding_spec : Optional[dict]
            Specification of the embedding net from the run configuration or None.
        input_size : int
            Size of the inputs into the embedding network.
        purpose : str
            Purpose of the embedding network, used for error messages.

        Returns
        -------
        Tuple[nn.Module, int]
            The embedding net and its output size.
        """
        if embedding_spec is None:
            return nn.Identity(), input_size

        if input_size == 0:
            raise ValueError(f'Cannot create {purpose} embedding layer with input size 0')

        emb_type = embedding_spec['type'].lower()
        if emb_type != 'fc':
            raise ValueError(f'{purpose} embedding type {emb_type} not supported.')

        hiddens = embedding_spec['hiddens']
        if len(hiddens) == 0:
            raise ValueError(f'{purpose} embedding "hiddens" must be a list of hidden sizes with at least one entry')

        dropout = embedding_spec['dropout']
        activation = embedding_spec['activation']

        emb_net = FC(input_size=input_size, hidden_sizes=hiddens, activation=activation, dropout=dropout)
        return emb_net, emb_net.output_size

    def forward(self, data: Dict[str, torch.Tensor], concatenate_output: bool = True) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Perform a forward pass on the input layer.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            The input data.
        concatenate_output : bool, optional
            If True (default), the forward method will concatenate the static inputs to each dynamic time step.
            If False, the forward method will return a tuple of (dynamic, static) inputs.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If `concatenate_output` is True, a single tensor is returned. Else, a tuple with one tensor of dynamic
            inputs and one tensor of static inputs.
        """
        # transpose to [seq_length, batch_size, n_features]
        if self.embedding_type == 'full_model':
            data_type = 'x_d'
            x_d = data[data_type].transpose(0, 1)
        elif self.embedding_type == 'forecast':
            data_type = 'x_f'
            x_d = data[data_type]
            #******为了overlap*******
            # import pdb; pdb.set_trace()  # 这里会暂停代码
            # x_d 是形状为 [256, 6, 30] 的张量
            first_step_feats = x_d[:, 0, :]  # 取第一个时间步，shape: [256, 30]
            # import pdb; pdb.set_trace()  # 这里会暂停代码
            # # 分组切片，每一共self.forecast_length 每组 self.hindcast_length 个特征
            # a = first_step_feats[:, 0:2].unsqueeze(-1)  # [256, 6, 1]
            # b = first_step_feats[:, 2:4].unsqueeze(-1)  # [256, 6, 1]
            #
            # # 沿最后一维拼接：得到 [256, 6, 5]
            # x_transformed = torch.cat([a, b], dim=-1)  # [256, 6, 5]
            # x_d = x_transformed.transpose(0, 1)

            # # 假设 first_step_feats 是 [batch_size, forecast_length * hindcast_length]
            # batch_size = first_step_feats.shape[0]
            # total_feat = first_step_feats.shape[1]
            #
            # # 确保可以整除
            # assert total_feat == self.forecast_length * self.hindcast_length, "特征数不能整除，请检查设置"
            #
            # 自动分组切片
            group_list = []
            for i in range(self.hindcast_length):
                start = i *self.forecast_length
                end = (i + 1) * self.forecast_length
                group = first_step_feats[:, start:end].unsqueeze(-1)  # [batch_size, hindcast_length, 1]
                group_list.append(group)
            # ***********************************************/
            # 拼接所有组：得到 [batch_size, hindcast_length, forecast_length]
            x_transformed = torch.cat(group_list, dim=-1)

            # 转置为 [hindcast_length, batch_size, forecast_length]
            x_d = x_transformed.transpose(0, 1)


        # elif self.embedding_type == 'forecast':
        #     data_type = 'x_f'
        #     x_d = data[data_type]
        #     #******为了overlap*******
        #     # import pdb; pdb.set_trace()  # 这里会暂停代码
        #     # x_d 是形状为 [256, 6, 30] 的张量
        #     first_step_feats = x_d[:, -(self.forecast_length - self.forecast_overlap), :]  # 取第一个时间步，shape: [256, 30]
        #     group_list = []
        #     for i in range(self.hindcast_length):
        #         start = i *(self.forecast_length - self.forecast_overlap)
        #         end = (i + 1) * (self.forecast_length - self.forecast_overlap)
        #         group = first_step_feats[:, start:end].unsqueeze(-1)  # [batch_size, hindcast_length, 1]
        #         group_list.append(group)
        #     # ***********************************************/
        #     # 拼接所有组：得到 [batch_size, hindcast_length, forecast_length]
        #     x_transformed = torch.cat(group_list, dim=-1)
        #
        #     # 转置为 [hindcast_length, batch_size, forecast_length]
        #     x_d = x_transformed.transpose(0, 1)


        elif self.embedding_type == 'hindcast':
            data_type = 'x_h'
            x_d = data[data_type].transpose(0, 1)

        # x_d (24, 256,,5)
        if 'x_s' in data and 'x_one_hot' in data:
            x_s = torch.cat([data['x_s'], data['x_one_hot']], dim=-1)
        elif 'x_s' in data:
            x_s = data['x_s']
        elif 'x_one_hot' in data:
            x_s = data['x_one_hot']
        else:
            x_s = None

        # Don't run autoregressive inputs through the embedding layer. This does not work with NaN's
        if self._num_autoregression_inputs > 0:
            dynamics_out = self.dynamics_embedding(x_d[:, :, :-self._num_autoregression_inputs])
        else:
            dynamics_out = self.dynamics_embedding(x_d)

        statics_out = None
        if x_s is not None:
            statics_out = self.statics_embedding(x_s)

        if not concatenate_output:
            ret_val = dynamics_out, statics_out
        else:
            if statics_out is not None:
                statics_out = statics_out.unsqueeze(0).repeat(dynamics_out.shape[0], 1, 1)
                ret_val = torch.cat([dynamics_out, statics_out], dim=-1)
            else:
                ret_val = dynamics_out

            # Append autoregressive inputs to the end of the output.
            if self._num_autoregression_inputs:
                ret_val = torch.cat([ret_val, x_d[:, :, -self._num_autoregression_inputs:]], dim=-1)

        return ret_val

    def __getitem__(self, item: str) -> nn.Module:
        # required for dict-like access when freezing submodules' gradients in fine-tuning
        if item == "statics_embedding":
            return self.statics_embedding
        elif item == "dynamics_embedding":
            return self.dynamics_embedding
        else:
            raise KeyError(f"Cannot access {item} on InputLayer")
