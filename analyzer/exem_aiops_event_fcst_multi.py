import datetime

from pprint import pformat
import pandas as pd
import numpy as np
from pathlib import Path
import datetime
from dateutil.parser import parse

from analyzer import aimodule
from common.constants import SystemConstants as sc
from common.module_exception import ModuleException
from common.redisai import REDISAI
from common import constants

class ExemAiopsEventFcstMulti(aimodule.AIModule):
    def __init__(self, config, logger):
        """
        :param config:
        :param logger:
        """
        # Config
        self.config = config
        self.module_id = self.config["module"]
        self.db_conn_str = self.config["db_conn_str"]
        self.model_dir = self.config["model_dir"]
        self.log_dir = self.config["log_dir"]
        self.sys_id = self.config["sys_id"]
        self.inst_type = self.config["inst_type"]
        # Log
        self.logger = logger
        self.logger.info(f"\t\t [Init] Create {pformat(config['inst_type'])} instance")
        self.serving_logger = None

    def init_load(self):
        target_id = 'super'

        logger_dir = str(Path(self.log_dir) / self.name / self.inst_type / target_id)
        logger_name = f"{self.module_id}_{self.inst_type}"

        logger = self.create_multi_logger(
            logger_dir, logger_name, "serving", f"{sc.EXEM_AIOPS_EVENT_FCST_MULTI}_{self.inst_type}"
        )
        self.serving_logger = logger

        Path(self.model_dir).mkdir(exist_ok=True, parents=True)

    def _get_root_key(self, group_type, group_id):
        return f"{self.sys_id}/{self.module_id}/{group_type}/{group_id}"

    def _validation_check(self, data_dict, target_id, serving_time, window_size):
        c1 = 3  # 최근 c1 분 데이터 검사, 데이터 받아야 할듯
        c2 = 5  # 피처당 c2개의 nan 허용, 데이터 받아야 할듯

        input_df = pd.DataFrame.from_dict(data_dict)
        input_df = input_df.loc[:, input_df.isnull().sum() != len(input_df)]
        input_df = input_df.where(pd.notnull(input_df), None)

        input_df['time'] = pd.to_datetime(input_df['time'])
        serving_time = pd.to_datetime(serving_time)

        # window size Data Frame 구성
        start_time = serving_time - pd.Timedelta(minutes=window_size-1)
        time_range = pd.date_range(start=start_time, end=serving_time, freq='T')
        time_df = pd.DataFrame(time_range, columns=['time'])
        input_df = pd.merge(time_df, input_df, on='time', how='left')

        feat_input_df = input_df.drop(columns=['time'])
        has_all_nans_last_c1 = feat_input_df.iloc[-c1:].isna().all(axis=1)
        has_all_nans_count_c2 = feat_input_df.isna().all(axis=1)
        if has_all_nans_last_c1.all():
            self.serving_logger.warning(f"{self.inst_type}_{target_id} - invalid dataset(최근 3분치 데이터 없음) \n"
                                        f"{has_all_nans_last_c1[has_all_nans_last_c1].index}")
            raise ModuleException("E704")
        elif has_all_nans_count_c2.sum() > c2:
            self.serving_logger.warning(f"{self.inst_type}_{target_id} - invalid dataset(6분 이상의 데이터 없음) \n"
                                        f"{has_all_nans_count_c2[has_all_nans_count_c2].index}")
            raise ModuleException("E704")
        else:
            # interpolate
            interpolated_df = input_df.interpolate(method='linear', limit_direction='both')
            self.serving_logger.debug(f"{self.inst_type}_{target_id} - valid dataset")

        return interpolated_df

    def serve(self, header, data_dict):
        group_type = header['group_type']
        group_id = header['group_id']
        target_id = header['target_id']  # 서빙 요청 타겟
        serving_time = header['predict_time']

        self.serving_logger.info(f"{self.sys_id}_{group_type}_{group_id}_{self.inst_type}_{target_id} - serving start")

        window_size = 60  #  고정된 값이라 파라미터 뺄 필요 없어 보임
        forecast_len = 30  #  고정된 값이라 파라미터 뺄 필요 없어 보임

        # 유효성 검사
        input_df = self._validation_check(data_dict, target_id, serving_time, window_size)

        # Make redis key
        root_key = self._get_root_key(group_type, group_id)
        model_root_key = f"{root_key}/{self.inst_type}_{target_id}"
        tsmixer_model_key = f"{model_root_key}_tsmixer"
        tsmixer_scaler_key = f"{model_root_key}_tsmixer_scaler"
        gat_model_key = f"{model_root_key}"
        bundle_key = f"{model_root_key}_bundle"

        # Model config, event_definition
        model_config_key = f"{root_key}/model_config"
        try:
            model_config = REDISAI.inference_json(model_config_key)
        except Exception as e:
            self.serving_logger.error(f"sys_id:[{self.sys_id}], target_id:[{target_id}] cannot load model_config from REDISAI")
            raise Exception(e)
        event_definition = model_config['event_definition']

        # tsmixer scaler
        try:
            tsmixer_scaler = REDISAI.inference_joblib(tsmixer_scaler_key)
        except Exception as e:
            self.serving_logger.error(f"sys_id:[{self.sys_id}], target_id:[{target_id}] cannot load tsmixer scaler from REDISAI")
            raise Exception(e)

        # Bundle
        try:
            bundle = REDISAI.inference_pickle(bundle_key)
        except Exception as e:
            self.serving_logger.error(f"sys_id:[{self.sys_id}], target_id:[{target_id}] cannot load bundle from REDISAI")
            raise Exception(e)

        gat_scaler = bundle["scaler"]
        columns_order = bundle["columns"]
        threshold = bundle["threshold"]
        anomaly_cdf = bundle["anomaly_cdf"]

        # ----------------- 부하 예측
        input_df = input_df[columns_order]
        feat_data = input_df.astype('float32')

        feat_data_scaled_rep = tsmixer_scaler.transform(feat_data)[np.newaxis, :, :]

        try:
            scaled_preds = REDISAI.inference(tsmixer_model_key, feat_data_scaled_rep)[0]
        except Exception as e:
            raise ModuleException('E705')

        scaled_90_preds = np.concatenate((feat_data_scaled_rep, scaled_preds), axis=1)
        scaled_60_preds = scaled_90_preds[:, 30:, :]
        x = np.round(np.maximum(tsmixer_scaler.inverse_transform(np.squeeze(scaled_60_preds)), 0), 2)

        # ---------------- 이벤트 예측
        x = gat_scaler.transform(x)
        x = x.reshape((1,) + x.shape)

        try:
            _, recon = REDISAI.event_clf_inference(gat_model_key, x)
        except Exception as e:
            raise ModuleException('E705')

        # get anomaly score
        a_score = np.sqrt((recon - x) ** 2)
        a_score = np.where(x > recon, a_score, 1e-7)  # 지표 크기가 감소 이벤트 아님
        a_score = np.squeeze(a_score, axis=0)

        # anomaly smoothing
        smoothing_window = int(window_size * 0.05)
        a_score = pd.DataFrame(a_score).ewm(span=smoothing_window).mean().values

        # 미래 forecast_len(30분) anomaly 남김
        forecast_anomaly_arr = a_score[forecast_len:,:]

        is_event_feature_dict = dict()
        if self.inst_type == "db":
            db_key = 'ORACLE'
            db_key = next((value for key, value in constants.DB_KEY_MAPPING.items() if key in target_id),
                          db_key)
            for event_code, event_info in event_definition.items():
                if db_key in event_info["features"][self.inst_type].keys():
                    is_event_feature_dict[event_code] = np.array(
                        [True if col in event_info["features"][self.inst_type][db_key] else False for col in columns_order])
        else:
            for event_code, event_info in event_definition.items():
                if self.inst_type in event_info["features"]:
                    is_event_feature_dict[event_code] = np.array(
                        [True if col in event_info["features"][self.inst_type] else False for col in columns_order])

        glb_columns = np.array([True] * len(columns_order))
        glb_columns = {"ETC": glb_columns}
        glb_columns.update(is_event_feature_dict)
        is_event_feature_dict = glb_columns

        # ETC 이벤트 구문 추가
        etc_msg = dict()
        etc_msg["msg"] = "기타 이벤트가 예측됩니다."
        event_definition["ETC"] = etc_msg

        # Compare Event Anomaly Forecast (Avg of important features) with Threshold
        event_anomaly_forecast = dict()
        anomaly_arr = np.array([])
        anomaly_proba_arr = np.array([])
        predicted_event_columns = np.array([False] * len(columns_order))

        for key, value in threshold.items():
            event_anomaly_forecast[key] = np.average(forecast_anomaly_arr.T[is_event_feature_dict[key]], axis=0)
            # Compare with threshold
            anomaly = event_anomaly_forecast[key] > value
            # Check First Anomaly
            if event_anomaly_forecast[key][0] < threshold[key]:
                anomaly = np.array([False] * forecast_len)
            # Find proba from cpf
            serve_index = np.ceil(event_anomaly_forecast[key] * 100).astype(int)
            serve_index = np.where(serve_index > 999, 999, serve_index)
            proba = np.round(anomaly_cdf[key][serve_index], 2)

            anomaly_arr = np.append(anomaly_arr, anomaly)
            anomaly_proba_arr = np.append(anomaly_proba_arr, proba)
            if key != "ETC" and anomaly.sum() > 0:
                predicted_event_columns = is_event_feature_dict[key] | predicted_event_columns

        anomaly_arr = anomaly_arr.reshape(-1, forecast_len).T
        anomaly_arr = anomaly_arr.astype(bool)
        anomaly_proba_arr = anomaly_proba_arr.reshape(-1, forecast_len).T

        # ETC 이벤트 기여 지표에 대한 로직, Choose Features which have a bigger anomaly than ETC Threshold
        predicted_event_columns = np.logical_not(predicted_event_columns)
        predicted_event_columns = predicted_event_columns & (
                    np.average(forecast_anomaly_arr, axis=0) > threshold["ETC"])
        is_event_feature_dict["ETC"] = predicted_event_columns

        # ETC threshold 넘지 못할 경우 기타 이벤트는 발생하지 않음
        if predicted_event_columns.sum() == 0:
            anomaly_arr[:, 0] = np.array([False] * forecast_len)
            anomaly_proba_arr[:, 0] = np.array([0] * forecast_len)

        contrib_top_n = 5
        contrib_list = list()
        for i in range(forecast_len):
            contrib_event_list = list()
            for key in threshold.keys():
                # Filter Event features
                forecast_anomaly_event_arr = forecast_anomaly_arr[i][is_event_feature_dict[key]]
                if len(forecast_anomaly_event_arr) == 0:
                    contrib_event_list.append([(columns_order[0], 0)])  # dummy data
                else:
                    event_columns_arr = np.array(columns_order)[is_event_feature_dict[key]]
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
        if anomaly_arr.sum() > 0:
            self.serving_logger.info(f"{self.sys_id}_{group_type}_{group_id}_{self.inst_type}_{target_id} - Event is predicted")
        # Return Format
        res = {
            "keys": ["time", "predict_time", "sys_id", "group_type", "group_id", "event_prob", "explain", "event_doc",
                     "event_code"]
        }

        result_values_list = list()
        for t in range(forecast_len):
            p, a, c = anomaly_proba_arr[t], anomaly_arr[t], contrib_list[t]
            for i, event_code in enumerate(is_event_feature_dict.keys()):
                event_prob_list = list()
                explain_list = list()
                event_prob_list.append({"inst_type": self.inst_type, "target_id": target_id,
                                        "event_prob": round(p[i], 2),
                                        "event_result": a[i]})

                explain_list.append([{"inst_type": self.inst_type, "target_id": target_id,
                                      "metric_name": c[i][j][0],
                                      "contribution": round(float(c[i][j][1]), 2)} for j in
                                     range(len(c[i]))])
                result_values_list.append(
                    [header["predict_time"],
                     str(parse(header["predict_time"]) + datetime.timedelta(minutes=t + 1)),
                     self.sys_id, group_type, group_id,
                     event_prob_list, explain_list, event_definition[event_code]['msg'],
                     event_code])

        result = dict(res, **{'values': result_values_list})
        total_result_dict = dict()
        total_result_dict["event_fcst"] = result
        self.serving_logger.info(f"{self.sys_id}_{group_type}_{group_id}_{self.inst_type}_{target_id} - serving done")

        return None, total_result_dict, 0, None