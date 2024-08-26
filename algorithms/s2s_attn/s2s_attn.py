import os
import time
import joblib
import numpy as np

from abc import *
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn import exceptions

from algorithms import aimodel
from common.timelogger import TimeLogger
from common import constants

from common.memory_analyzer import MemoryUtil
from common.module_exception import ModuleException
from common.redisai import REDISAI


class Seq2seqAttention(aimodel.AIModel, metaclass=ABCMeta):
    def __init__(self, id, config, logger):
        self.mu = MemoryUtil(logger)

        self.logger = logger
        self.config = config
        self.model_dir = self.config['model_dir']

        self.business_list = config.get('business_list', None)
        self.except_failure_date_list = config.get('except_failure_date_list', [])
        self.except_business_list = config.get('except_business_list', [])

        self.algo_str = None
        self.params_s2s = None
        self.pred_feats = None

        self.algo_log = None
        self.model_desc = None
        self.progress_name = None
        self.drop_rate = {}
        self.model_id = None

        self.test_perc = None

        self.scalers = {}
        self.models = {}

        # model params
        self.params_s2s = None
        self.window_size = {}
        self.lstm_out = {}
        self.attn_out = {}
        self.batch_size = {}
        self.lr = {}
        self.epochs = {}

        # band params (const)
        self.n_band_iter = 50

    def set_vars(self, algo_str):
        self.default_features = self.config['parameter']['train'][constants.MODEL_S_DBSLN]['features']
        self.algo_str = algo_str
        self.params_s2s = self.config['parameter']['train'][self.algo_str]
        self.pred_feats = self.params_s2s['features']
        self.sigma_coef = {feat: 3 for feat in self.pred_feats}

    def init_config(self, config):
        self.config = config

        self.params_s2s = config['parameter']['train'][self.algo_str]
        self.pred_feats = self.params_s2s['features']
        self.test_perc = config['parameter']['data_set']['test'] if 'data_set' in config['parameter'].keys() else None

        self.init_param(config)

        self.scalers = dict([(feat, StandardScaler()) for feat in self.pred_feats])
        self.set_vars(self.algo_str)

    def get_hyper_param_values(self, params, hyper_param):
        if 'metric_hyper_params' in params.keys():
            metrics = params["metric_hyper_params"].keys()
            return dict([(metric, params["metric_hyper_params"][metric].get(hyper_param, None)) for metric in metrics])
        else:
            return dict()

    def init_param(self, config):
        self.params_s2s = config['parameter']['train'][self.algo_str]
        self.pred_feats = self.params_s2s['features']
        self.window_size = self.get_hyper_param_values(self.params_s2s, 'window_size')
        self.lstm_out = self.get_hyper_param_values(self.params_s2s, 'hidden_unit_size')
        self.batch_size = self.get_hyper_param_values(self.params_s2s, 'batch_size')
        self.lr = self.get_hyper_param_values(self.params_s2s, 'learning_rate')
        self.epochs = self.get_hyper_param_values(self.params_s2s, 'epochs')

    def is_line_patterned(self, feat_data):
        return None

    def predict(self, df, sbiz_df=None):
        try:
            result = {}
            path = str(Path(self.model_dir) / self.algo_str)

            # df[self.pred_feats] = df[self.pred_feats].astype('float32')
            with TimeLogger(f"[{self.algo_log} Serving] elapsed time :", self.logger):
                for i, feat in enumerate(self.pred_feats):
                    if feat in df.columns:
                        feat_data = df[feat].fillna(value=np.nan).interpolate(limit_direction='both').values.astype('float32')
                        if feat_data.shape[0] < self.window_size[feat]:
                            continue
                        else:
                            feat_real = df[feat].values[-self.window_size[feat]:]
                            feat_data = feat_data[-self.window_size[feat]:]

                        feat_scaler_key = str(Path(path) / f"{self.model_desc}_{feat}_scaler")
                        feat_scaler_key = REDISAI.make_redis_model_key(feat_scaler_key, "")
                        onnx_feat_model_key = str(Path(path) / f"{self.model_desc}_{feat}")
                        onnx_feat_model_key = REDISAI.make_redis_model_key(onnx_feat_model_key, "")

                        # get self.scalers[feat] from redisai
                        self.scalers[feat] = REDISAI.inference_joblib(feat_scaler_key)
                        feat_data_scaled_rep = np.tile(self.scalers[feat].transform(feat_data.reshape(-1, 1)).reshape(1, -1, 1), self.n_band_iter).T

                        # input_name = f"{onnx_feat_model_key}/in"
                        # output_name = f"{onnx_feat_model_key}/out"
                        preds = REDISAI.inference(onnx_feat_model_key, feat_data_scaled_rep)
                        preds = self.scalers[feat].inverse_transform(np.squeeze(preds))

                        pred_mu, pred_sigma = preds.mean(axis=0).round(2), preds.std(axis=0).round(2)
                        lower_band, upper_band = np.maximum(0, pred_mu - self.sigma_coef[feat] * pred_sigma), pred_mu + self.sigma_coef[feat] * pred_sigma
                        pred_mu = np.maximum(0, pred_mu)

                        self.fill_serving_result(result, feat, lower_band, upper_band, pred_mu, pred_sigma, feat_real)
        except exceptions.NotFittedError as e:
            raise ModuleException('E705')
        except KeyError as e:
            raise ModuleException('E705')

        return result

    @abstractmethod
    def fill_serving_result(self, result, feat, lower_band, upper_band, pred_mu=None, pred_sigma=None, real=None):
        pass

    def load(self, path):
        self.mu.print_memory()
        # self.mu.gc_info()

        path = str(Path(path) / self.algo_str)
        Path(path).mkdir(exist_ok=True, parents=True)

        try:
            load_start = time.time()
            self.models = {}
            for feat in self.pred_feats:
                loaded_objects = []
                self.logger.info(f"[{self.algo_log}] Loading {feat} ...")
                feat_model_path = str(Path(path) / f"{self.model_desc}_{feat}.h5")
                feat_scaler_path = str(Path(path) / f"{self.model_desc}_{feat}_scaler.pkl")
                onnx_feat_model_path = str(Path(path) / f"{self.model_desc}_{feat}.onnx")

                if os.path.exists(feat_model_path) or os.path.exists(onnx_feat_model_path):
                    self.models[feat] = REDISAI.save_onnx_to_redis(onnx_feat_model_path)
                    self.window_size[feat] = self.config["parameter"]["train"][self.algo_str]["metric_hyper_params"][feat]["window_size"]
                    loaded_objects.append('model')
                else:
                    self.logger.info(f"{feat} model not found")

                # scaler
                if os.path.exists(feat_scaler_path):
                    self.scalers[feat] = joblib.load(feat_scaler_path)
                    loaded_objects.append('scaler')

                self.logger.info(f"Loaded object(s) from {feat} : {', '.join(loaded_objects)}")
                if len(loaded_objects) < len(['model', 'scaler']):
                    return False

            # self.apply_bayesian_approx_to_model()
        except Exception as e:
            self.logger.error(f"[{self.algo_log}] Error log while Load() : {e}")
            return False

        load_end = time.time()
        self.logger.info(f"[{self.algo_log}] load() elapsed : {load_end - load_start:.3f}s")

        self.mu.print_memory()
        # self.mu.gc_info()

        return True
