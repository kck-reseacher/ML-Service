import os
import datetime
import numpy as np
import pandas as pd
import torch

from collections import defaultdict
from pathlib import Path

from dateutil.parser import parse

from algorithms.mtad_gat.args import hyper_parameter
from analyzer import aimodule
from common import aicommon, constants
from common.aicommon import Query
from common.aiserverAPI import ServerAPI
from common.constants import SystemConstants as sc
from common.error_code import Errors
from common.redisai import REDISAI
from common.timelogger import TimeLogger


class ExemAiopsEventFcstClf(aimodule.AIModule):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.target_logger = None
        self.serverAPI = ServerAPI(config, logger)
        self.db_query = Query.CreateQuery(self.config, self.logger)
        self.home = Path(config.get("home")) if config.get("home") else None
        self.sys_id = config["sys_id"]
        self.target_id = config["target_id"]
        self.inst_type = config["inst_type"]
        self.module_id = config["module"]
        self.log_dir = config["log_dir"]
        self.train_params = {}
        self.service_id = f"{self.module_id}_{self.inst_type}_{self.target_id}"
        # 서빙 시 이벤트 feature, msg 정보 담음
        self.event_cluster = None
        self.event_msg = None
        # codeGroup 서빙 시 이벤트 특정 위한 dict
        self.event_feature_dict = dict()

        # set model param
        hp = hyper_parameter
        # 학습에 이용할 데이터 조건
        self.least_train_days = hp["least_train_days"]
        self.least_nan_percent = hp["least_nan_percent"]
        # 에폭 배치 학습률은 대시보드 통한 입력으로 변경
        self.p_n_epochs = hp["p_n_epochs"]
        self.p_batch_size = hp["p_bs"]
        self.p_learning_rate = hp["p_init_lr"]
        # 이외의 param
        self.window_size = hp["lookback"]
        self.normalize = hp["normalize"]
        self.spec_res = hp["spec_res"]

        # Conv1D, GAT layers
        self.kernel_size = hp["kernel_size"]
        self.use_gatv2 = hp["use_gatv2"]
        self.feat_gat_embed_dim = hp["feat_gat_embed_dim"]
        self.time_gat_embed_dim = hp["time_gat_embed_dim"]
        # GRU
        self.gru_n_layers = hp["gru_n_layers"]
        self.gru_hid_dim = hp["gru_hid_dim"]

        self.fc_n_layers = hp["fc_n_layers"]
        self.fc_hid_dim = hp["fc_hid_dim"]

        self.recon_n_layers = hp["recon_n_layers"]
        self.recon_hid_dim = hp["recon_hid_dim"]

        self.alpha = hp["alpha"]

        self.val_split = hp["val_split"]
        self.shuffle_dataset = hp["shuffle_dataset"]
        self.dropout = hp["dropout"]
        self.use_cuda = hp["use_cuda"]
        self.device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"
        self.print_every = hp["print_every"]
        self.log_tensorboard = hp["log_tensorboard"]

        self.scale_scores = hp["scale_scores"]
        self.use_mov_av = hp["use_mov_av"]
        self.gamma = hp["gamma"]
        self.level = hp["level"]
        self.q = hp["q"]
        self.reg_level = hp["reg_level"]
        self.dynamic_pot = hp["dynamic_pot"]

        # Serving Param
        self.forecast_len = hp["forecast_len"]
        # Anomaly Forecast 시간 가중치
        self.weight_arr = None

        self.model_dir = self.config["model_dir"]

        self.models = {}
        self.bundle = {}
        self.model_name_list = list()

    def init_train(self):
        pass

    def init_param(self, config):
        pass

    def init_serve(self, reload=False):
        self.config = {}
        self.config["header"] = {"success": True}
        self.target_logger = self.create_logger("super")
        self.get_weight_arr()
        return True

    def get_weight_arr(self):
        time_index = list(range(self.window_size + self.forecast_len))
        weight_list = []
        for f in range(self.forecast_len):
            tmp_list = list()
            for w in range(self.window_size):
                tmp = (time_index[self.window_size + f] - time_index[0]) / (
                        time_index[self.window_size + f] - time_index[w])
                tmp_list.append(tmp)
            weight_list.append(tmp_list)
        self.weight_arr = np.array(weight_list)

    def create_logger(self, target_id):
        """
        logger를 만드는 함수.
        기존처럼 load 함수 내에서 로거를 만들면 동일한 로거가 만들어지고 Handler만 추가되므로 중복 로깅됨
        참고 : https://5kyc1ad.tistory.com/269
        Parameters
        ----------
        target_id : 타겟아이디
        Returns : 로거
        ----------
        """
        logger_dir = str(Path(self.log_dir) / sc.EXEM_AIOPS_EVENT_FCST_MULTI / self.inst_type)
        logger_name = f"{self.module_id}_{self.inst_type}_{self.sys_id}"

        logger = self.create_multi_logger(
            logger_dir, logger_name, self.sys_id, f"{constants.EXEM_AIOPS_EVENT_FCST}_{self.inst_type}")
        return logger

    def _load(self, model_name):
        try:
            fpath = os.path.join(self.config['model_dir'])
            # bundle
            bundle_PATH = str(Path(fpath) / f"{model_name}_bundle.pkl")
            bundle_key = REDISAI.make_redis_model_key(bundle_PATH, ".pkl")
            self.bundle[model_name] = REDISAI.inference_pickle(bundle_key)
        except Exception as ex:
            self.target_logger.exception(f"== Failed to load [{self.sys_id}_{self.inst_type}_{self.target_id}_{model_name}] : {ex}")
            return False

        self.target_logger.info(f"== Success to load [{self.sys_id}_{self.inst_type}_{self.target_id}_{model_name}]")
        return True

    def serve(self, header, input_df):
        def remove_prefix(s):
            return '_'.join(s.split('_')[1:])
        self.sys_id = header['sys_id']
        self.inst_type = header["inst_type"]
        self.target_id = header['target_id']

        self.target_logger.info(
            f"============ [{self.sys_id}_{self.inst_type}_{self.target_id}] Start Loading ============")

        self.model_name_list = self.config['results']['EventPredictor']['train_metrics'].keys()
        self.event_cluster = self.config['results']["event_cluster"]
        self.event_msg = self.config['results']["event_msg"]
        self.event_msg["ETC"] = "기타 지표로부터 이벤트가 예측됩니다."  # 임의로 Setting, 문구 표출 방법, 문구 협의 필요

        for target in self.model_name_list:
            self._load(target)

        res = {
            "keys": ["time", "predict_time", "sys_id", "group_type", "group_id", "event_prob", "explain", "event_doc", "event_code"]
        }
        self.target_logger.info(
            f"============ [{self.sys_id}_{self.inst_type}_{self.target_id}] Start Preprocessing ============")
        df_dict = self.preprocessing_for_serve(input_df)
        values_length = np.array([len(value) for value in df_dict.values()])
        self.target_logger.info(
            f"[{self.sys_id}_{self.inst_type}_{self.target_id}] check inst_target each length : {values_length}")
        self.target_logger.info(
            f"============ [{self.sys_id}_{self.inst_type}_{self.target_id}] Finish Preprocessing ============")

        if len(np.where(values_length < self.window_size)[0]) > 0:
            self.target_logger.exception(
                f"[{self.sys_id}_{self.inst_type}_{self.target_id}] at least one of the targets is less than {self.window_size}). target list = {[list(df_dict.keys())[i] for i in np.where(values_length < self.window_size)[0]]}")
            return None, None, Errors.E704.value, Errors.E704.desc
        self.target_logger.info(
            f"============ [{self.sys_id}_{self.inst_type}_{self.target_id}] Start Serving============")

        result_dict = defaultdict(dict)
        result_db_dict = defaultdict(dict)
        fpath = os.path.join(self.config['model_dir'])
        with ((TimeLogger(f"[{self.sys_id}_{self.inst_type}_{self.target_id}] elapsed time :", self.target_logger))):
            for target, df in df_dict.items():
                self.target_logger.info(f"[{self.sys_id}_{self.inst_type}_{self.target_id}] inst_target: {target}")
                # get bundle items
                scaler = self.bundle[target]["scaler"]
                columns = self.bundle[target]["columns"]
                threshold = self.bundle[target]["threshold"]
                anomaly_cdf = self.bundle[target]["anomaly_cdf"]
                # 이벤트 주요 지표 T/F arr
                inst = target.split("_")[0]
                is_event_feature_dict = dict()
                if inst == "db":
                    db_key = 'ORACLE'
                    db_key = next((value for key, value in constants.DB_KEY_MAPPING.items() if key in target),
                                  db_key)
                    for event_code, event_features in self.event_cluster[inst][db_key].items():
                        is_event_feature_dict[event_code] = np.array(
                            [True if col in event_features else False for col in columns])
                else:
                    for event_code, event_features in self.event_cluster[inst].items():
                        is_event_feature_dict[event_code] = np.array(
                            [True if col in event_features else False for col in columns])
                # sort column and scale
                data = np.asarray(df[columns], dtype=np.float32)
                x = scaler.transform(data)
                x = x.reshape((1,) + x.shape)

                # model load and inference
                onnx_model_path = str(Path(fpath) / f"{target}.onnx")
                onnx_model_key = REDISAI.make_redis_model_key(onnx_model_path, ".onnx")
                _, recon = REDISAI.event_clf_inference(onnx_model_key, x)

                # get anomaly score
                a_score = np.sqrt((recon - x) ** 2)
                a_score = np.squeeze(a_score, axis=0)

                # anomaly smoothing
                smoothing_window = int(self.window_size * 0.05)
                a_score = pd.DataFrame(a_score).ewm(span=smoothing_window).mean().values

                # Anomaly Forecast
                forecast_anomaly = []
                for i in range(30):
                    right = a_score[1:, :].T
                    left = a_score[:-1, :].T
                    w = np.array(self.weight_arr[i][1:])
                    d = (right / left) ** w
                    forecast_anomaly.append(a_score[-1] * np.prod(d, axis=1) ** (i / sum(self.weight_arr[i][1:])))
                forecast_anomaly_arr = np.array(forecast_anomaly)

                # Add ETC features
                glb_columns = np.array([True] * len(columns))
                glb_columns = {"ETC": glb_columns}
                glb_columns.update(is_event_feature_dict)
                is_event_feature_dict = glb_columns

                # Compare Event Anomaly Forecast (Avg of important features) with Threshold
                event_anomaly_forecast = dict()
                anomaly_arr = np.array([])
                anomaly_proba_arr = np.array([])
                predicted_event_columns = np.array([False] * len(columns))

                for key, value in threshold.items():
                    event_anomaly_forecast[key] = np.average(forecast_anomaly_arr.T[is_event_feature_dict[key]], axis=0)
                    # Compare with threshold
                    anomaly = event_anomaly_forecast[key] > value
                    # Check First Anomaly
                    if event_anomaly_forecast[key][0] < threshold[key]:
                        anomaly = np.array([False] * 30)
                    # Find proba from cpf
                    serve_index = np.ceil(event_anomaly_forecast[key] * 100).astype(int)
                    serve_index = np.where(serve_index > 999, 999, serve_index)
                    proba = np.round(anomaly_cdf[key][serve_index], 2)

                    anomaly_arr = np.append(anomaly_arr, anomaly)
                    anomaly_proba_arr = np.append(anomaly_proba_arr, proba)
                    if key != "ETC" and anomaly.sum() > 0:
                        predicted_event_columns = is_event_feature_dict[key] | predicted_event_columns

                anomaly_arr = anomaly_arr.reshape(-1, self.forecast_len).T
                anomaly_arr = anomaly_arr.astype(bool)
                anomaly_proba_arr = anomaly_proba_arr.reshape(-1, self.forecast_len).T

                # ETC 이벤트 기여 지표에 대한 로직, Choose Features which have a bigger anomaly than ETC Threshold
                predicted_event_columns = np.logical_not(predicted_event_columns)
                predicted_event_columns = predicted_event_columns & (
                            np.average(forecast_anomaly_arr, axis=0) > threshold["ETC"])
                is_event_feature_dict["ETC"] = predicted_event_columns

                # ETC threshold 넘지 못할 경우 기타 이벤트는 발생하지 않음
                if predicted_event_columns.sum() == 0:
                    anomaly_arr[:, 0] = np.array([False] * 30)
                    anomaly_proba_arr[:, 0] = np.array([0] * 30)

                contrib_top_n = 5
                contrib_list = list()
                for i in range(self.forecast_len):
                    contrib_event_list = list()
                    for key in threshold.keys():
                        # Filter Event features
                        forecast_anomaly_event_arr = forecast_anomaly_arr[i][is_event_feature_dict[key]]
                        if len(forecast_anomaly_event_arr) == 0:
                            contrib_event_list.append([(columns[0], 0)])  # dummy data
                        else:
                            event_columns_arr = np.array(columns)[is_event_feature_dict[key]]
                            # Get index of contrib features
                            sorted_idx = np.argsort(forecast_anomaly_event_arr)
                            idx = sorted_idx[-contrib_top_n:]
                            idx = idx[::-1]  # Reverse index
                            contrib_score = forecast_anomaly_event_arr[idx]
                            contrib_score_percent = contrib_score / np.sum(contrib_score)
                            contrib_feature = event_columns_arr[idx]
                            contrib_tuple = list(zip(contrib_feature, contrib_score_percent))
                            contrib_event_list.append(contrib_tuple)
                    contrib_list.append(contrib_event_list)

                # arr to list
                anomaly_proba_list = anomaly_proba_arr.tolist()
                anomaly_list = anomaly_arr.tolist()

                if anomaly_arr.sum() > 0:
                    self.target_logger.debug(f"[{self.sys_id}_{self.inst_type}_{self.target_id}] "
                                             f" Event is predicted from inst_target: {target}\n"
                                             f" Feature anomaly score: {sorted(zip(columns, a_score[-1] * 100), key=lambda x: x[1], reverse=True)}")

                if target.split('_')[0] == "db":
                    db_key = 'ORACLE'
                    db_key = next((value for key, value in constants.DB_KEY_MAPPING.items() if key in target),
                                  db_key)
                    result_db_dict[db_key][target] = anomaly_proba_list, anomaly_list, contrib_list
                else:
                    result_dict[target.split('_')[0]][target] = anomaly_proba_list, anomaly_list, contrib_list

            result_dict["db"] = result_db_dict
        self.target_logger.info(
            f"============ [{self.sys_id}_{self.inst_type}_{self.target_id}] ALL target Inferece Finish ============")

        # 리턴 포맷 생성
        result_values_list = list()
        # ETC 이벤트 포맷
        for i in range(self.forecast_len):
            event_prob_list = list()
            explain_list = list()
            for instance_key, instance_dict in result_dict.items():
                if instance_key == "db":
                    for db_type_key, db_instance_dict in result_dict[instance_key].items():
                        for t, (proba, anomaly, contrib) in db_instance_dict.items():
                            p, a, c = proba[i], anomaly[i], contrib[i]
                            # 이벤트 순서
                            event_prob_list.append({"inst_type": t.split("_")[0], "target_id": remove_prefix(t),
                                                    "event_prob": round(p[0], 2),
                                                    "event_result": a[0]})

                            explain_list.append([{"inst_type": t.split("_")[0], "target_id": remove_prefix(t),
                                                  "metric_name": c[0][j][0],
                                                  "contribution": round(float(c[0][j][1]), 2)} for j in
                                                 range(len(c[0]))])
                else:  # db 이외
                    for t, (proba, anomaly, contrib) in instance_dict.items():
                        p, a, c = proba[i], anomaly[i], contrib[i]
                        # 이벤트 순서
                        event_prob_list.append({"inst_type": t.split("_")[0], "target_id": remove_prefix(t),
                                                "event_prob": round(p[0], 2),
                                                "event_result": a[0]})

                        explain_list.append([{"inst_type": t.split("_")[0], "target_id": remove_prefix(t),
                                              "metric_name": c[0][j][0],
                                              "contribution": round(float(c[0][j][1]), 2)} for j in
                                             range(len(c[0]))])
            result_values_list.append(
                [header["predict_time"],
                 str(parse(header["predict_time"]) + datetime.timedelta(minutes=i + 1)),
                 self.sys_id, self.inst_type, self.target_id,
                 event_prob_list, explain_list, self.event_msg["ETC"],
                 "ETC"])

        # ETC 외의 이벤트 리턴 포맷
        for instance_key, instance_dict in result_dict.items():
            if instance_key == "db":
                for db_type_key, db_instance_dict in result_dict[instance_key].items():
                    event_code_list = list(self.event_cluster[instance_key][db_type_key].keys())
                    event_code_list.insert(0, "ETC")
                    for event_index in range(1, len(event_code_list)):
                        for i in range(self.forecast_len):
                            event_prob_list = list()
                            explain_list = list()
                            for t, (proba, anomaly, contrib) in db_instance_dict.items():
                                p, a, c = proba[i], anomaly[i], contrib[i]
                                # 이벤트 순서
                                event_prob_list.append({"inst_type": t.split("_")[0], "target_id": remove_prefix(t),
                                                        "event_prob": round(p[event_index], 2),
                                                        "event_result": a[event_index]})

                                explain_list.append([{"inst_type": t.split("_")[0], "target_id": remove_prefix(t),
                                                      "metric_name": c[event_index][j][0],
                                                      "contribution": round(float(c[event_index][j][1]), 2)} for j in
                                                     range(len(c[event_index]))])

                            result_values_list.append(
                                [header["predict_time"],
                                 str(parse(header["predict_time"]) + datetime.timedelta(minutes=i + 1)),
                                 self.sys_id, self.inst_type, self.target_id,
                                 event_prob_list, explain_list, self.event_msg[event_code_list[event_index]],
                                 event_code_list[event_index]])
            else:  # db 이외
                event_code_list = list(self.event_cluster[instance_key].keys())
                event_code_list.insert(0, "ETC")
                for event_index in range(1, len(event_code_list)):
                    for i in range(self.forecast_len):
                        event_prob_list = list()
                        explain_list = list()
                        for t, (proba, anomaly, contrib) in instance_dict.items():
                            p, a, c = proba[i], anomaly[i], contrib[i]
                            # 이벤트 순서
                            event_prob_list.append({"inst_type": t.split("_")[0], "target_id": remove_prefix(t),
                                                    "event_prob": round(p[event_index], 2),
                                                    "event_result": a[event_index]})

                            explain_list.append([{"inst_type": t.split("_")[0], "target_id": remove_prefix(t),
                                                  "metric_name": c[event_index][j][0],
                                                  "contribution": round(float(c[event_index][j][1]), 2)} for j in
                                                 range(len(c[event_index]))])

                        result_values_list.append(
                            [header["predict_time"],
                             str(parse(header["predict_time"]) + datetime.timedelta(minutes=i + 1)),
                             self.sys_id, self.inst_type, self.target_id,
                             event_prob_list, explain_list, self.event_msg[event_code_list[event_index]],
                             event_code_list[event_index]])

        result = dict(res, **{'values': result_values_list})
        total_result_dict = dict()
        total_result_dict["event_fcst"] = result
        self.target_logger.info(
            f"============ [{self.sys_id}_{self.inst_type}_{self.target_id}] ALL Serving Process Done ============")
        return None, total_result_dict, 0, None

    def preprocessing_for_serve(self, df):
        """
            서빙 데이터 그룹 별 분리
            결측 처리
        """
        df['full_name'] = df.apply(lambda row: row['inst_type'] + '_' + row['target_id'], axis=1)
        df = df.pivot_table(index=['time', 'full_name'], columns='name', values='real_value')
        df = df.reset_index()
        df = df.sort_values(['time'])

        # 타겟 별 과거 데이터 merge
        df_dict = dict()
        for target in set(df['full_name']):  # full_name = was_214, was_215 ...
            self.target_logger.info(f'[{self.sys_id}_{self.inst_type}_{self.target_id}] inst_target: {target}')
            df_t = df[df['full_name'] == target]
            df_t.index = pd.to_datetime(df_t["time"])

            e = df_t.index[-1]
            s = e - datetime.timedelta(minutes=59)
            t_index = pd.DatetimeIndex(pd.date_range(start=s, end=e, freq="1min"))

            each_target_df = df_t.reindex(t_index).interpolate(method="linear").fillna(method='ffill').fillna(
                method='bfill').fillna(0)

            # 필요한 컬럼 자르기
            inst_t = target.split('_')
            if inst_t[0] == 'db':
                db_key = 'ORACLE'
                db_key = next((value for key, value in constants.DB_KEY_MAPPING.items() if key in target),
                              db_key)
                each_target_df = each_target_df[
                    self.config['parameter']['train']['eventfcst']['features'][inst_t[0]][db_key]]
            else:
                each_target_df = each_target_df[self.config['parameter']['train']['eventfcst']['features'][inst_t[0]]]

            df_dict[target] = each_target_df

        return df_dict

    def end_serve(self):
        pass

    def get_debug_info(self):
        pass

