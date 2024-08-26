from algorithms.s2s_attn.s2s_attn import Seq2seqAttention

import numpy as np

from common import constants


class S2SAttnForecast(Seq2seqAttention):
    def __init__(self, id, config, logger):
        super().__init__(id, config, logger)

        self.set_vars(constants.MODEL_S_S2S)
        self.init_config(config)

        self.algo_log = 'S2S_Attn'
        self.model_desc = 's2s_attn'
        self.progress_name = 'Seq2Seq-Attention'
        self.drop_rate = {'pred': 0.1, 'band': 0.25}

        self.pred_horizon = 30

    def init_param(self, config):
        super().init_param(config)
        self.attn_out = dict([(feat, hidden_size if hidden_size is None else hidden_size // 2) for feat, hidden_size in self.lstm_out.items()])

    def fill_serving_result(self, result, feat, lower_band, upper_band, pred_mu=None, pred_sigma=None, real=None):
        result[feat] = dict(zip(np.arange(1, self.pred_horizon + 1), np.maximum(0, pred_mu)))
        result[f"{feat}_lower"] = dict(zip(np.arange(1, self.pred_horizon + 1), lower_band))
        result[f"{feat}_upper"] = dict(zip(np.arange(1, self.pred_horizon + 1), upper_band))
