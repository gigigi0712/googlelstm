from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.gumbel import Gumbel

class GumbelMoeLSTM(BaseModel):
    module_parts = ['embedding_net', 'lstm', 'head']

    def __init__(self, cfg: Config):
        super(GumbelMoeLSTM, self).__init__(cfg=cfg)

        self.embedding_net = InputLayer(cfg)

        n_experts = cfg.exp_num
        hidden_dim = 128
        self.n_experts = n_experts

        # 门控网络（Gate Network）
        self.gate_network = nn.Sequential(
            nn.Linear(self.embedding_net.statics_output_size, hidden_dim),  # 输入静态特征 X_s
            nn.ReLU(),
            nn.Linear(hidden_dim, n_experts)  # 输出专家权重 logits
        )

        self.lstm = nn.LSTM(input_size=self.embedding_net.output_size, hidden_size=cfg.hidden_size)

        # 定义多个 LSTM 专家
        self.experts = nn.ModuleList([
            nn.LSTM(input_size=self.embedding_net.output_size, hidden_size=cfg.hidden_size, batch_first=True)
            for _ in range(n_experts)
        ])

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        # 输出层
        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

        self._reset_parameters()

        # 初始化温度参数（用于控制稀疏性）
        self.initial_temperature = 1  # 初始温度
        self.min_temperature = 0.15  # 最低温度
        self.temperature_decay = 0.95  # 温度衰减系数

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        if self.cfg.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[self.cfg.hidden_size:2 * self.cfg.hidden_size] = self.cfg.initial_forget_bias

    def gumbel_softmax(self, logits, temperature):
        """
        实现 Gumbel-Softmax 采样。
        :param logits: 原始 logits (未归一化的分数) [batch_size, n_experts]
        :param temperature: Gumbel-Softmax 的温度参数
        :return: 稀疏化概率分布 [batch_size, n_experts]
        """
        # 生成 Gumbel 噪声
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        gumbel_logits = (logits + gumbel_noise) / temperature
        return F.softmax(gumbel_logits, dim=-1)

    def forward(self, data: Dict[str, torch.Tensor], current_epoch: int = 0) -> Dict[str, torch.Tensor]:
        """
        前向传播。
        :param data: 输入数据
        :param current_epoch: 当前训练轮数，用于动态调整温度
        :return: 模型的预测输出
        """
        x_d, x_s = self.embedding_net(data, concatenate_output=False)  # x_d: dynamic, x_s: static
        x = self.embedding_net(data, concatenate_output=True)

        # 1. 门控网络：根据静态变量 X_s 生成每个专家的 logits
        expert_logits = self.gate_network(x_s)  # [batch_size, n_experts]

        # 动态调整温度参数
        temperature = max(
            self.min_temperature,
            self.initial_temperature * (self.temperature_decay ** current_epoch)
        )

        # Gumbel-Softmax 采样
        expert_weights = self.gumbel_softmax(expert_logits, temperature)  # [batch_size, n_experts]

        # 输入 LSTM 时确保 batch-first
        x = x.transpose(0, 1)  # 转换为 batch-first 格式

        # 2. 对每个专家的动态变量 X_d 进行处理
        expert_outputs = []
        for i, lstm in enumerate(self.experts):
            lstm_output, _ = lstm(x)  # [batch_size, seq_len, lstm_hidden_dim]
            expert_outputs.append(lstm_output[:, -1, :])  # 取最后一个时间步的输出

        # 将所有专家的输出堆叠起来
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, n_experts, lstm_hidden_dim]

        # 3. 根据专家权重对输出进行加权
        # 对专家权重增加一个维度以便广播 [batch_size, n_experts, 1]
        expert_weights = expert_weights.unsqueeze(-1)
        # 加权平均专家输出
        weighted_output = torch.sum(expert_weights * expert_outputs, dim=1)  # [batch_size, lstm_hidden_dim]

        # 4. 通过全连接层生成最终预测值
        pred = {'lstm_output': weighted_output,"expert_weights":expert_weights}
        pred.update(self.head(self.dropout(weighted_output)))
        pred["y_hat"] = pred["y_hat"].unsqueeze(1)
        return pred