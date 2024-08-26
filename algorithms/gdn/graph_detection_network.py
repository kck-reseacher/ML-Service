import copy

import datetime
from pathlib import Path
from typing import List, Dict, Union

import numpy as np
import pandas as pd
import torch


from algorithms import aimodel

from common.timelogger import TimeLogger
from common import constants
from common.module_exception import ModuleException
from common.redisai import REDISAI


class GraphDetectionNetwork(aimodel.AIModel):

    def __init__(self, id, config, logger):
        ####################
        self.logger = logger
        self.config = copy.deepcopy(config)

        self.algo_name = 'gdn'
        self.model_desc = 'graph based anomaly detection network'
        self.progress_name = 'GDN'

        ####################
        cfg = config['parameter']['train']['gdn']
        self.batch = 512  # cfg.get('batch_size', 1024)
        self.epoch = 100  # cfg.get('epoch', 100)
        self.edge_index_sets = None
        self.node_num = None
        self.slide_win = 30
        self.slide_stride = 1
        self.dim = 64
        self.topk = 0
        self.out_layer_inter_dim = 256,
        self.out_layer_num = 1
        self.val_ratio = 0.1
        self.decay = 0
        self.report = 'best'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.window_size = 30
        self.confidence_level = 98

        ####################
        removal_features = ["socket_count"]  # ["extcall_time", "extcall_time", "socket_count", "file_count", "fail_count", "extcall_count"]
        self.train_meta_feature_list = list(filter(lambda x: x not in removal_features, cfg.get('features')))

        self.train_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.feature_map = []
        self.save_path = config.get('model_dir')
        self.scaler = None
        self.cdf = None  # cumulative density func
        self.threshold = None
        self.model = None
        self.best_model = None

        self.logger.info("Success init GDN !!")

    def predict(self, input_df):
        path = str(Path(self.model_dir) / self.algo_name)
        time = pd.to_datetime(input_df.index[-1], format=constants.INPUT_DATETIME_FORMAT)
        time = time.strftime(constants.INPUT_DATETIME_FORMAT)
        cols_len = len(self.pred_feats)

        with TimeLogger(f"[{self.algo_name} Serving] elapsed time :", self.logger):

            if any(columns not in input_df.columns for columns in self.pred_feats):
                self.logger.error(f"The learned feature does not exist in the serving data.")
                raise ModuleException('E706')

            df = input_df[self.pred_feats]

            if df.isnull().sum().sum() > 0:
                feat_data = df.astype('float32').interpolate(limit_direction='both').fillna(method='ffill').fillna(
                    method='bfill')
            else:
                feat_data = df.astype('float32')

            real_data = df.values[-self.window_size:]
            feat_data = feat_data[-self.window_size:]

            bundle_key = str(Path(path) / f"{self.algo_name}_bundle")
            bundle_key = REDISAI.make_redis_model_key(bundle_key, "")
            onnx_feat_model_key = str(Path(path) / f"{self.algo_name}_model")
            onnx_feat_model_key = REDISAI.make_redis_model_key(onnx_feat_model_key, "")

            try:
                bundle = REDISAI.inference_pickle(bundle_key)
                scaler = bundle['scaler']
                score_table = bundle['score_table']
                feat_data_scaled_rep = scaler.transform(feat_data)
                feat_data_scaled_rep = feat_data_scaled_rep.T[np.newaxis, :, :]

                preds, attention_map, _ = REDISAI.inference_gdn(onnx_feat_model_key, feat_data_scaled_rep)

            except Exception as e:
                raise ModuleException('E705')

            loss = abs(feat_data_scaled_rep[:, :, -1] - preds)
            loss = np.squeeze(loss)
            anomaly_score, is_anomaly, anomaly_contribution = self.get_anomaly_result(loss, score_table)

            preds = scaler.inverse_transform(preds)
            preds = np.squeeze(preds)

            neighbor_attention_map = attention_map[:-cols_len].reshape(-1, cols_len - 1)
            self_attention_map = attention_map[-cols_len:].reshape(-1, 1)

            diagonal_attention_map = np.empty((cols_len, cols_len), dtype=object)
            for idx, value in enumerate(self_attention_map):
                diagonal_attention_map[idx] = np.insert(neighbor_attention_map[idx], idx, value)

            result = self.make_result_format(preds, real_data, diagonal_attention_map, is_anomaly, anomaly_score,
                                             anomaly_contribution)
            result['time'] = time

        return result

    def init_config(self, config: Dict):
        self.config = config

        self.params = config['results'][self.algo_name]
        self.pred_feats = self.params['features']
        # self.window_size = self.parmas['window_size']

    def make_result_format(self, preds: np.ndarray, real_data: np.ndarray, attention_map: List[Dict], is_anomaly: bool, anomaly_score: int, anomaly_contribution: int):
        result = {
            "time": "",
            "sys_id": self.config['sys_id'],
            "target_id": self.config['target_id'],
            "inst_type": self.config['inst_type'],
            "is_anomaly": is_anomaly,
            "normality_score": anomaly_score,
            "target_grade": self.evaluate_score(anomaly_score),
            "metric_list": []
        }

        for idx, column in enumerate(self.pred_feats):
            result["metric_list"].append({
                "real_value": real_data[-1][idx],
                "model_value": max(round(float(preds[idx]), 4), 0),
                "anomaly_contribution": round(float(anomaly_contribution[idx]), 1),
                "metric_name": column,
                "attention_map": self.get_attention_map(attention_map[idx]) if is_anomaly else None
            })

        self.sort_based_on_anomaly_contribution(result['metric_list'])

        return result

    def sort_based_on_anomaly_contribution(self, metric_list):
        metric_list.sort(key=lambda x: x['anomaly_contribution'], reverse=True)

    def get_attention_map(self, attention_map: np.ndarray):
        return {
            col: round(float(value),4) for col, value in zip(self.pred_feats, attention_map)
        }

    def evaluate_score(self, score):
        if score <= 30:
            return 2
        elif score <= 50:
            return 1
        else:
            return 0

    def get_anomaly_result(self, loss, score_table):
        score_list = []
        for i in range(len(loss)):
            table = score_table[i]
            feature_loss = loss[i]

            feature_score = table[table["loss"] <= feature_loss][-1:]["score"].values[0]

            score_list.append(feature_score)
        target_score = np.mean(score_list)

        if target_score < 50:
            is_anomaly = True
        else:
            is_anomaly = False

        anomaly_score = np.round(target_score, 1) if target_score > 5 else 5
        anomaly_contribution = self.get_anomaly_contribution(loss)

        return (anomaly_score, is_anomaly, anomaly_contribution)

    def calculate_anomaly_score(self, cdf, loss, is_anomaly):
        WORST_ANOMALY_SCORE = 5
        pos = np.where(cdf['loss'] > loss)

        if is_anomaly:
            score_of_default = 50
        else:
            score_of_default = 100
        if any(*pos):
            return max(np.round(score_of_default - (cdf['cdf'][min(*pos)] / 2) * 100, 1), WORST_ANOMALY_SCORE)

        return WORST_ANOMALY_SCORE

    def get_anomaly_contribution(self, losses):
        total_loss = losses.sum()
        return [
            100 * (loss/total_loss) for loss in losses
        ]

