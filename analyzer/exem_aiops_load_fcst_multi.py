import datetime
import json
import os
import time
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd

from analyzer import aimodule
from common import aicommon, constants
from common.constants import SystemConstants as sc
from common.aiserverAPI import ServerAPI
from common.error_code import Errors
from common.module_exception import ModuleException
from common.redisai import REDISAI


class ExemAiopsLoadFcstMulti(aimodule.AIModule):
    def __init__(self, config, logger):
        """
        instance의 각 지표(metric)의 "이상 탐지"와 "예측"을 함
        기능
         - 학습
            - 입력 데이터 : 사용자에게 입력 받은 csv 형태의 pandas 의 지표별 시계열 데이터
            - 출력 데이터 : 데이터를 학습한 모델 파일과 학습 결과 파일
                - model_config.json, model_meta.json
         - 서빙
            - 입력 데이터 : 분당 t - 60 ~ t 까지의 각 인스턴스의 지표 데이터를 받아옴
            - 출력 데이터 : 이상 탐지 및 예측 결과를 dict 형태로 전달
                - 서버에서 db(postresql)로 전송함
        지원 알고리즘
         - 시계열 예측
             - Seq2Seq
        설정
        - 서버에서 학습 및 서빙 여부를 json 파라미터로 받아옴.
             - serverAPI
        메뉴
         - 학습 : 설정 -> 학습/서비스 -> 학습 -> 타입 (was, db, host(os), ... etc) -> 모듈 (이상탐지 / 부하예측)
         - 서비스 : 설정 -> 학습/서비스 -> 서비스 -> 타입 (was, db, host(os), ... etc) -> 모듈 (이상탐지 / 부하예측)
         - 차트 : 대시보드 -> 이상 탐지 모니터링 or 대시보드 -> 부하 예측 모니터링

        Parameters
        ----------
        config : 각종 설정 파일, 추가로 서버에서 설정 파일을 받아 옴.
        logger : 로그를 출력하기 위한 로거
        """
        self.config = config
        self.serving_logger = logger

        # naming rule에 따라 module/model에서 사용할 ID 생성
        # 이 ID를 이용해 csv 파일이나 model binary를 읽고 쓸때 필요한 이름 생성
        self.inst_type = config["inst_type"]
        self.module_id = config["module"]
        self.log_dir = config["log_dir"]
        self.model_dir = config["model_dir"]

        self.sys_id = config["sys_id"]

        self.serving_logger.info(f"config :{pformat(config)}")

        self.modelMap = {"super": {}}
        self.serverAPI = ServerAPI(config, logger)

        self.model_dict = {}

    def init_load(self, target_info, target_port, server_env: str):
        load_failed_target_list = list()

        target_list = target_info[self.inst_type]

        for target_id in target_list:
            load_res = False

            try:
                load_res = self.load(target_id)
            except Exception as e:
                self.serving_logger.exception(f"target_id ({str(target_id)}) load failed cause {e}")

            if not load_res:
                load_failed_target_list.append(target_id)

        if len(load_failed_target_list) > 0:
            self.serving_logger.error(f"port {target_port} model load error list : {str(load_failed_target_list)}")

    def create_logger(self, target_id):
        """
        logger를 만드는 함수.
        기존처럼 load 함수 내에서 로거를 만들면 동일한 로거가 만들어지고 Handler만 추가되므로 중복 로깅됨
        참고 : https://5kyc1ad.tistory.com/269
        Parameters
        ----------
        target_id : 타겟아이디
        Returns : 로거
        -------
        """
        logger_dir = str(Path(self.log_dir) / self.name / self.inst_type)
        logger_name = f"{self.module_id}_{self.inst_type}"

        logger = self.create_multi_logger(
            logger_dir, logger_name, "serving", f"{sc.EXEM_AIOPS_LOAD_FCST_MULTI}_{self.inst_type}"
        )

        return logger

    def _check_and_create_target_logger(self, target_id):
        is_logger = self.check_multi_logger(target_id, self.modelMap)

        if is_logger:
            self.modelMap['super'] = {}
            self.modelMap['super']["logger"] = self.create_logger(target_id)
        else:
            if not self.modelMap['super']["logger"].hasHandlers():
                self.modelMap['super']["logger"] = self.create_logger(target_id)

    def load(self, target_id=None, reload=False):

        if reload is True:
            if target_id in self.modelMap:
                self.modelMap['super']["logger"].info("[reload] start reload model")
            else:
                self.serving_logger.info(
                    f"[reload] fail, target_id: {target_id} is not exist!!"
                )

        self._check_and_create_target_logger(target_id)

        logger = self.modelMap['super'].get("logger")
        logger.info("[load] start loading model")

        config = {}
        config["sys_id"] = ""
        config["inst_type"] = self.inst_type
        config["module"] = self.module_id
        config["target_id"] = target_id
        config["parameter"] = {}  # parameter
        config["model_dir"] = str(Path(self.model_dir))
        config["log_dir"] = self.log_dir

        Path(config["model_dir"]).mkdir(exist_ok=True, parents=True)

        self.modelMap['super']["config"] = config
        self.modelMap['super']["logger"].info("[load] start model!")

        service_id = f"{self.module_id}_{config['target_id']}"

        self.modelMap['super']["logger"].info("[load] start creating instances and model loading")

        module_name = constants.ALGO_MODULE_CLASS_NAME[constants.MODEL_S_LOADFCST]['module_name']
        class_name = constants.ALGO_MODULE_CLASS_NAME[constants.MODEL_S_LOADFCST]['class_name']

        aimodule_path = Path(os.environ.get(sc.MLOPS_SERVING_PATH)) / "algorithms"
        path = str(aimodule_path)

        try:
            target_class = aicommon.Utils.get_module_class(module_name, class_name, path)
            self.modelMap['super'][constants.MODEL_S_LOADFCST] = target_class(service_id, config, logger)

        except MemoryError as error:
            self.modelMap['super']["logger"].exception(
                f"[Error] Unexpected memory error during serving : {error}"
            )
            aicommon.Utils.print_memory_usage(logger)
            raise error
        except Exception as exception:
            self.modelMap['super']["logger"].exception(
                f"[Error] Unexpected exception during serving : {exception}"
            )
            raise exception

        self.modelMap['super']["logger"].info("[load] finish creating instances and model loading")
        return True

    def end_train(self):
        pass

    # API for serving
    def init_serve(self, reload=False):
        res = self._load(reload)
        return res

    def _get_model_config_path(self, model_dir, sys_id=None, target_id=None):
        return str(Path(
            model_dir) / f"{sys_id}" / f"{self.module_id}" / f"{self.inst_type}" / f"{target_id}" / "model_config.json")

    def _update_model_config(self, model_config):
        self.modelMap['super']["config"]["target_id"] = model_config['target_id']
        self.modelMap['super']["config"]["sys_id"] = model_config['sys_id']
        self.modelMap['super']["config"]["parameter"] = model_config['parameter']
        self.modelMap['super']["config"]["model_dir"] = str(
            Path(self.model_dir) / f"{self.sys_id}" / f'{self.module_id}' / f'{self.inst_type}' / model_config[
                'target_id'])

        self.modelMap['super'][constants.MODEL_S_LOADFCST].init_config(self.modelMap['super']["config"])

    def serve(self, header, data_dict):
        target_id = header["target_id"]
        sys_id = header['sys_id']

        self.target_logger = self.modelMap['super']["logger"]
        self.target_logger.info("=========== Start Serving ===========")

        model_config_path = self._get_model_config_path(self.model_dir, sys_id, target_id)
        model_config_key = REDISAI.make_redis_model_key(model_config_path, "")
        model_config_key = model_config_key.replace(".json", "")
        model_config = REDISAI.inference_json(model_config_key)
        self._update_model_config(model_config)

        serving_data_dict = {}

        input_df = pd.DataFrame.from_dict(data_dict)
        input_df['target_id'] = input_df['target_id'].astype(str)
        for target_id in np.unique(input_df['target_id'].values):
            target_df = input_df[input_df['target_id'] == target_id]
            if len(target_df) < 60:
                self.target_logger.info(f"target data length not enough target_id: {target_id}, target_df: {target_df}")
                continue
            target_df = target_df.sort_values(by="time")
            target_df = target_df.drop_duplicates()
            serving_data_dict[str(target_id)] = target_df

        if len(serving_data_dict) <= 0:
            raise ModuleException("E704")

        input_dict = pd.DataFrame(input_df.iloc[-1]).to_dict()
        data_dict_cur = input_dict[list(input_dict.keys())[0]]
        serving_datetime_str = data_dict_cur["time"]

        # for business calendar serving
        business_list = header.get("business_list", None) if header is not None else None
        if business_list is None:
            self.target_logger.info(f"{Errors.E824.desc}")
            raise ModuleException("E824")

        seq2seq_pred_dict = {}
        pred_result = None
        error_code = None
        error_msg = None

        try:
            pred_result = self.modelMap['super'][constants.MODEL_S_LOADFCST].predict(serving_data_dict)
        except ModuleException as me:
            self.target_logger.exception(f"Algorithm ModuleException Cause: {me.error_msg}")
            error_code = me.error_code
            error_msg = me.error_msg
        except Exception as e:
            self.target_logger.exception(f"Unknown Algorithm Exception Cause: {e}")
            error_code = f"{Errors.E900.value}"
            error_msg = f"{Errors.E900.desc}"

        seq2seq_pred_dict["pred_result"] = pred_result
        seq2seq_pred_dict["error_code"] = error_code
        seq2seq_pred_dict["error_msg"] = error_msg

        # predict 결과가 없는 경우 빈 dict 형태 return
        res_seq2seq_perf = {}

        # default
        results = {
            "sys_id": sys_id,
            "target_id": target_id,
            "time": serving_datetime_str,
        }

        if bool(seq2seq_pred_dict):
            if seq2seq_pred_dict.get("error_code") is None:
                seq2seq_pred = seq2seq_pred_dict["pred_result"]
                res_seq2seq_perf = self.insert_serving_load_pred_performance(
                    results["time"], seq2seq_pred
                )
            else:
                res_seq2seq_perf = aicommon.Utils.set_except_msg(seq2seq_pred_dict)

        res = {
            "predict_seq2seq": res_seq2seq_perf,
        }

        # self.target_logger.info(f"res : {res}")
        self.target_logger.info("=========== End Serving =========")

        return None, res, None, None

    def end_serve(self):
        pass

    def insert_serving_load_pred_performance(self, time, load_pred):
        res = {
            "keys": ["time", "sys_id", "target_id", "type", "name", "predict_value"],
            "values": [],
        }

        if load_pred is None:
            return res

        tt = datetime.datetime.strptime(time, constants.INPUT_DATETIME_FORMAT)
        predict_time = [tt + datetime.timedelta(minutes=x) for x in range(31)]

        for target_id in load_pred.keys():
            feature_list = [
                feature
                for feature in list(load_pred[target_id].keys())
                if "upper" not in feature and "lower" not in feature
            ]
            for feature in feature_list:
                predict_list = list()
                for i in load_pred[target_id][feature].keys():
                    # time 필드 추가 dbsln_date 활용
                    predict_value = dict()
                    predict_value["time"] = predict_time[i].strftime(
                        constants.INPUT_DATETIME_FORMAT
                    )
                    predict_value["value"] = load_pred[target_id][feature][i]
                    predict_list.append(predict_value)

                if load_pred and feature in load_pred[target_id].keys():
                    res["values"].append(
                        [
                            time,
                            self.sys_id,
                            target_id,
                            self.inst_type,
                            feature,
                            {"data": predict_list},
                        ]
                    )
                else:
                    res["values"].append(
                        [
                            time,
                            self.sys_id,
                            target_id,
                            self.inst_type,
                            feature,
                            None,
                        ]
                    )

        return res
