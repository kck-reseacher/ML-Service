import os
import joblib
import numpy as np

import psutil
import concurrent
from algorithms import aimodel
from common.timelogger import TimeLogger
from common.redisai import REDISAI
from pathlib import Path


class LoadForecast(aimodel.AIModel):
    def __init__(self, id, config, logger):

        self.logger = logger
        self.config = config

        self.algo_log = 'S2S_Attn'
        self.model_desc = 's2s_attn'
        self.progress_name = 'Seq2Seq-Attention'
        self.window_size = {}

        self.drop_rate = {'pred': 0.1, 'band': 0.1}
        self.pred_horizon = 30

        self.multi_scalers = {}
        self.models = {}

        self.is_multiprocessing_mode = True
        self.is_multithread_mode = True
        self.number_of_child_processes = int(psutil.cpu_count(logical=True) * 0.3)

    def init_config(self, config):
        self.config = config
        self.model_dir = self.config['model_dir']

        parameters = config['parameter']['train']["seq2seq"]
        self.pred_feats = parameters['features']
        self.test_ratio = config["parameter"]["data_set"]["test"] if 'data_set' in self.config[
            'parameter'].keys() else None
        self.sigma_coef = dict([(feat, 3) for feat in self.pred_feats])
        self.init_param(parameters)

    def init_param(self, parameters):
        self.window_size = dict(
            [(feat, parameters["metric_hyper_params"][feat]["window_size"]) for feat in self.pred_feats])
        self.epochs = dict([(feat, parameters["metric_hyper_params"][feat]["epochs"]) for feat in self.pred_feats])
        self.batch_size = dict(
            [(feat, parameters["metric_hyper_params"][feat]["batch_size"]) for feat in self.pred_feats])
        self.lr = dict([(feat, parameters["metric_hyper_params"][feat]["learning_rate"]) for feat in self.pred_feats])
        self.lstm_out = dict(
            [(feat, parameters["metric_hyper_params"][feat]["hidden_unit_size"]) for feat in self.pred_feats])
        self.attn_out = dict([(feat, hidden_size if hidden_size is None else hidden_size // 2) for feat, hidden_size in
                              self.lstm_out.items()])

    def feature_predict(self, feat, df_dict, serving_target_list):
        results = {}
        input_data = []
        for target_id in serving_target_list:
            results[target_id] = {}
            target_df = df_dict[target_id]
            if feat not in self.multi_scalers[target_id].keys() or feat not in target_df.columns:
                continue
            feat_data = target_df[feat].fillna(value=np.nan).interpolate(limit_direction='both').values.astype(
                'float32')
            if feat_data.shape[0] < self.window_size[feat]:
                continue
            else:
                feat_data = feat_data[-self.window_size[feat]:]
            feat_data_scaled = self.multi_scalers[target_id][feat].transform(feat_data.reshape(-1, 1)).reshape(1, -1, 1)
            input_data.append(feat_data_scaled)
        if len(input_data) == 0:
            return results
        input_data = np.vstack(input_data)
        onnx_feat_model_key = str(Path(self.model_dir) / f"{self.model_desc}_{feat}")
        onnx_feat_model_key = REDISAI.make_redis_model_key(onnx_feat_model_key, "")
        preds = REDISAI.inference(onnx_feat_model_key, input_data)
        preds = np.squeeze(preds)

        if preds.ndim < 2:
            preds = preds[np.newaxis, :]

        for idx, target_id in enumerate(serving_target_list):
            target_preds = preds[idx]
            target_preds = self.multi_scalers[target_id][feat].inverse_transform(target_preds.reshape(-1, 1))
            results[target_id][feat] = {i + 1: np.maximum(0, value) for i, value in enumerate(target_preds.flatten())}

        return results

    # multi target batch prediction
    def predict(self, df_dict):
        total_result = {}
        with TimeLogger(f"[{self.algo_log} Serving] elapsed time :", self.logger):
            scaler_key = str(Path(self.model_dir) / f"{self.model_desc}_scaler")
            scaler_key = REDISAI.make_redis_model_key(scaler_key, "")

            self.multi_scalers = REDISAI.inference_joblib(scaler_key)

            serving_target_list = []
            for target_id in df_dict.keys():
                if target_id in self.multi_scalers.keys():
                    total_result[target_id] = {}
                    serving_target_list.append(target_id)

            self.logger.info(f"target {list(set(df_dict.keys()) - set(serving_target_list))} not fitted in Traning")

            if self.is_multithread_mode:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self.feature_predict, feat, df_dict, serving_target_list) for feat in
                               self.pred_feats]
                    concurrent.futures.wait(futures)
                    for future in futures:
                        feat_result = future.result()
                        for target_id, result_dict in feat_result.items():
                            total_result[target_id].update(result_dict)
            else:
                for feat in self.pred_feats:
                    feat_result = self.feature_predict(feat, df_dict, serving_target_list)
                    for target_id, result_dict in feat_result.items():
                        total_result[target_id].update(result_dict)

        return total_result

    def load(self, model_dir):

        # 구간별 일반화 모델
        for feat in self.pred_feats:
            feat_model_path = str(Path(model_dir) / f"{self.model_desc}_{feat}.h5")
            onnx_feat_model_path = str(Path(model_dir) / f"{self.model_desc}_{feat}.onnx")
            # feat_scaler_path = str(Path(model_dir) / f"{self.model_desc}_{feat}_scaler.pkl")
            self.logger.info(f"feat_model_path: {feat_model_path}")
            if os.path.exists(feat_model_path) or os.path.exists(onnx_feat_model_path):
                # self.models[feat] = ONNX.onnx_load(onnx_feat_model_path, feat_model_path)
                self.models[feat] = REDISAI.save_onnx_to_redis(onnx_feat_model_path)
                self.logger.info(f"load model {feat} success")

        scaler_path = str(Path(model_dir) / f"{self.model_desc}_scaler.pkl")
        if os.path.exists(scaler_path):
            self.multi_scalers = joblib.load(scaler_path)

        return True