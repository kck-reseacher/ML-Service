from algorithms.s2s_attn.s2s_attn import Seq2seqAttention

import numpy as np

from common import constants


class S2SAttnDetect(Seq2seqAttention):
    def __init__(self, id, config, logger):
        super().__init__(id, config, logger)

        self.set_vars(constants.MODEL_S_SEQATTN)
        self.init_config(config)

        self.algo_log = 'SeqAttn'
        self.model_desc = 'seqattn'
        self.progress_name = 'SeqAttn'
        self.drop_rate = {'pred': 0.0, 'band': 0.3}

    def init_config(self, config):
        super().init_config(config)

        if config['parameter'].get('service', None) is not None:
            for feat, sigma_val in config['parameter']['service'][self.algo_str]['range'].items():
                self.sigma_coef[feat] = sigma_val

    def init_param(self, config):
        super().init_param(config)
        self.attn_out = self.get_hyper_param_values(self.params_s2s, 'attention_size')

    def update_band_width(self, parameter):
        range_dict = parameter['parameter']['service'][constants.MODEL_S_SEQATTN]['range']
        for feat in self.pred_feats:
            if feat not in range_dict.keys():
                self.logger.info(f"{feat} not found in range_dict received from server")
            else:
                prev_val = self.sigma_coef[feat]
                self.sigma_coef[feat] = range_dict[feat]
                self.logger.info(f"{feat} sigma_coef changed : {prev_val} => {self.sigma_coef[feat]}")

    def is_line_patterned(self, feat_data):
        feat_ma = np.convolve(feat_data, np.ones(10), mode='valid') / 10
        feat_vals = feat_data[-feat_ma.shape[0]:]
        return True if feat_ma[feat_ma == feat_vals].shape[0] / feat_ma.shape[0] >= 0.9 else False

    def fill_serving_result(self, result, feat, lower_band, upper_band, pred_mu=None, pred_sigma=None, real=None):
        result[f"{feat}_lower"], result[f"{feat}_upper"] = lower_band[0], upper_band[0]
        result[f"{feat}_real"], result[f"{feat}_pred"], result[f"{feat}_std"] = eval(real[-1]) if type(real[-1]) is str else real[-1], pred_mu[0], pred_sigma[0]

        if 'features' not in result.keys():
            result['features'] = []
        result['features'].append(feat)

    def check(self, pred_result, serving_data):
        result = []
        for feat in pred_result.get('features', []):
            try:
                if serving_data[feat] is not None \
                        and serving_data[feat] >= constants.DBSLN_CHECK_MIN \
                        and not pred_result[f"{feat}_lower"] <= serving_data[feat] <= pred_result[f"{feat}_upper"]:
                    item = {
                        'name': feat,
                        'value': serving_data[feat],
                        'lower': pred_result[f"{feat}_lower"],
                        'upper': pred_result[f"{feat}_upper"],
                        'deviation': (serving_data[feat] - pred_result[f"{feat}_pred"]),
                        'zscore': (serving_data[feat] - pred_result[f"{feat}_pred"]) / pred_result[f"{feat}_std"]
                    }
                    result.append(item)
            except:
                continue
        return result