from typing import Dict

import torch
import torch.nn as nn
from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config


class CrossBeforeLstm(BaseModel):
    """LSTM with Cross Attention model class.

    This class extends the original CudaLSTM to incorporate a Cross Attention mechanism. The Cross Attention module
    now operates before the LSTM, allowing the interaction between dynamic features (time-dependent inputs) and static
    features (region-specific embeddings) to influence the hidden representations fed into the LSTM.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """
    module_parts = ['embedding_net', 'cross_attention', 'lstm', 'head']

    def __init__(self, cfg: Config):
        super(CrossBeforeLstm, self).__init__(cfg=cfg)

        # Embedding network for static and dynamic features
        self.embedding_net = InputLayer(cfg)

        # Cross Attention module
        self.query_projection = nn.Linear(self.embedding_net.dynamics_output_size, cfg.hidden_size)  # Query: Dynamic features
        self.key_projection = nn.Linear(self.embedding_net.statics_output_size, cfg.hidden_size)  # Key: Static features
        self.value_projection = nn.Linear(self.embedding_net.statics_output_size, cfg.hidden_size)  # Value: Static features

        self.cross_attention = nn.MultiheadAttention(embed_dim=cfg.hidden_size, num_heads=4,batch_first=True)

        # LSTM for dynamic feature modeling
        self.lstm = nn.LSTM(input_size=cfg.hidden_size+self.embedding_net.dynamics_output_size, hidden_size=cfg.hidden_size)

        # Dropout layer
        self.dropout = nn.Dropout(p=cfg.output_dropout)

        # Output head
        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        if self.cfg.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[self.cfg.hidden_size:2 * self.cfg.hidden_size] = self.cfg.initial_forget_bias

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the Cross Attention + LSTM model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary:
                - `y_hat`: model predictions of shape [batch size, sequence length, number of target variables].
                - `h_n`: hidden state at the last time step of the sequence of shape [batch size, 1, hidden size].
                - `c_n`: cell state at the last time step of the sequence of shape [batch size, 1, hidden size].
        """
        # Embedding dynamic and static features
        x_d, x_s = self.embedding_net(data, concatenate_output=False)  # x_d: dynamic, x_s: static

        # Cross Attention
        # Dynamic features as Query, Static features as Key/Value
        query = self.query_projection(x_d)  # [batch_size, seq_len, hidden_size]
        key = self.key_projection(x_s).unsqueeze(1)  # [batch_size, 1, hidden_size]
        value = self.value_projection(x_s).unsqueeze(1)  # [batch_size, 1, hidden_size]

        # Reshape for MultiheadAttention ([seq_len, batch_size, hidden_size])
        query = query.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
        # key = key.permute(1, 0, 2)  # [1, batch_size, hidden_size]
        # value = value.permute(1, 0, 2)  # [1, batch_size, hidden_size]

        # Cross Attention computation
        attention_output, _ = self.cross_attention(query, key, value)  # [seq_len, batch_size, hidden_size]
        attention_output = attention_output.permute(1, 0, 2)  # [batch_size, seq_len, hidden_size]

        combined_input = torch.cat((x_d,attention_output), dim=-1)
        # LSTM forward pass
        lstm_output, (h_n, c_n) = self.lstm(input=combined_input)  # lstm_output: [seq_len, batch_size, hidden_size]

        # Transpose LSTM output to [batch_size, seq_len, hidden_size]
        lstm_output = lstm_output.transpose(0, 1)  # [batch_size, seq_len, hidden_size]
        h_n = h_n.transpose(0, 1)  # [batch_size, 1, hidden_size]
        c_n = c_n.transpose(0, 1)  # [batch_size, 1, hidden_size]

        # Apply dropout and pass through the output head
        pred = {'lstm_output': lstm_output, 'h_n': h_n, 'c_n': c_n}
        pred.update(self.head(self.dropout(lstm_output)))

        return pred