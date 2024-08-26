import os
import datetime
import pandas as pd
import numpy as np

from pathlib import Path
from pprint import pformat
from analyzer import aimodule

from common import aicommon, constants
from common.redisai import REDISAI
from common.error_code import Errors
from common.aiserverAPI import ServerAPI
from common.memory_analyzer import MemoryUtil
from common.module_exception import ModuleException
from common.constants import SystemConstants as sc
np.set_printoptions(precision=2, suppress=True)

class ExemAiopsFcstTsmixerMulti(aimodule.AIModule):
    def __init__(self, config, logger):
        """
        instance의 "이상 탐지" 를 수행
        기능
         - 학습
            - 입력 데이터 : 사용자에게 입력 받은 csv 형태의 pandas 의 지표별 시계열 데이터
            - 출력 데이터 : 데이터를 학습한 모델 파일과 학습 결과 파일
                - model_config.json, model_meta.json

        부하 예측 알고리즘
        - TSMixer
            - 타겟(다변량) 부하 예측

        설정
        - 서버에서 학습 및 서빙 여부를 json 파라미터로 받아옴.
             - serverAPI
        메뉴
         - 학습 : 설정 -> 학습/서비스 -> 학습 -> 타입 (was, db, host(os), ... etc) -> 모듈 (이상탐지 / 부하예측)
         - 차트 : 대시보드 -> 이상 탐지 모니터링 or 대시보드 -> 부하 예측 모니터링

        Parameters
        ----------
        config : 각종 설정 파일, 추가로 서버에서 설정 파일을 받아 옴.
        logger : 로그를 출력하기 위한 로거
        """
        self.serving_logger = logger
        self.target_logger = None

        self.inst_type = config["inst_type"]
        self.module_id = config["module"]
        self.sys_id = config["sys_id"]
        self.log_dir = config["log_dir"]

        self.model_dir = config["model_dir"]
        self.temp_dir = config["temp_dir"]
        self.db_conn_str = config["db_conn_str"]
        self.anls_inst_template_path = config["anls_inst_template_path"]

        self.serving_logger.info(f"\t\t [Init] Create {pformat(config['inst_type'])} instance")

        self.modelMap = {'super': {}}
        self.serverAPI = ServerAPI(config, logger)
        self.mu = MemoryUtil(logger)


    def _get_model_config_path(self, model_dir, sys_id=None, target_id=None):
        return str(Path(
            model_dir) / f"{sys_id}" / f"{self.module_id}" / f"{self.inst_type}" / f"{target_id}" / "model_config.json")

    def _save(self):
        pass
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

        logger = self.create_multi_logger(logger_dir, logger_name, "serving", f"{sc.EXEM_AIOPS_LOAD_FCST_MULTI}_{self.inst_type}")

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
        """
        매 서빙마다 호출
        reload = False & modelMap[target_id] 경우, pass
        reload = True 인 경우, 업데이트 된 config 정보로 알고리즘 인스턴스 새로 생성
                              - 학습 종료 후 자동 배포
                              - 수동 배포
                              - 서비스 down → up
        """
        self.mu.print_memory()

        if reload is True:
            if target_id in self.modelMap:
                self.modelMap['super']["logger"].info("[reload] start reload model")
            else:
                self.serving_logger.info(f"[reload] fail, target_id: {target_id} is not exist!!")

        self._check_and_create_target_logger(target_id)

        logger = self.modelMap['super'].get("logger")
        logger.info("[load] start loading model")

        config = {}
        config["sys_id"] = self.sys_id
        config["inst_type"] = self.inst_type
        config["module"] = self.module_id
        config["target_id"] = target_id
        config["parameter"] = {} # parameter
        config["model_dir"] = str(Path(self.model_dir))
        config["temp_dir"] = self.temp_dir
        config["log_dir"] = self.log_dir
        config["db_conn_str"] = self.db_conn_str

        Path(config["model_dir"]).mkdir(exist_ok=True, parents=True)

        self.modelMap['super']["config"] = config
        self.modelMap['super']["logger"].info("[load] start model!")

        service_id = f"{self.module_id}_{config['target_id']}"

        # 개별 알고리즘 인스턴스 생성 및 모델 로딩
        self.modelMap['super']["logger"].info("[load] start creating instances and model loading")

        self.modelMap['super']["load_status"] = True
        module_name = constants.ALGO_MODULE_CLASS_NAME[constants.MODEL_S_TSMIXER]['module_name']
        class_name = constants.ALGO_MODULE_CLASS_NAME[constants.MODEL_S_TSMIXER]['class_name']

        aimodule_path = str(Path(os.environ.get(sc.AIMODULE_PATH)) / "algorithms")
        aimodule_path += f'/{module_name}'

        try:
            target_class = aicommon.Utils.get_module_class(module_name, class_name, f'{str(aimodule_path)}')
            self.modelMap['super'][constants.MODEL_S_TSMIXER] = target_class(service_id, config, logger)

        except MemoryError as error:
            self.modelMap['super']["logger"].exception(f"[Error] Unexpected memory error during serving : {error}")
            aicommon.Utils.print_memory_usage(logger)
            self.mu.print_memory()

            return False
        except Exception as exception:
            self.modelMap['super']["logger"].exception(f"[Error] Unexpected exception during serving : {exception}")

        self.modelMap['super']["logger"].info("[load] finish creating instances and model loading")

        return True

    def init_load(self, target_info, target_port, server_env: str):
        target_id = 'super'
        load_res = False

        try:
            load_res = self.load(target_id)
        except Exception as e:
            self.serving_logger.exception(f"load failed cause {e}")

        if not load_res:
            self.serving_logger.error(f"model load error")

    def _update_model_config(self, model_config):
        self.modelMap['super']["config"]["target_id"] = model_config['target_id']
        self.modelMap['super']["config"]["sys_id"] = model_config['sys_id']
        self.modelMap['super']["config"]["parameter"] = model_config['parameter']
        self.modelMap['super']["config"]["model_dir"] = str(Path(self.model_dir) / f"{self.sys_id}" / f'{self.module_id}' / f'{self.inst_type}' / model_config['target_id'])
        if model_config.get("business_list"):
            self.modelMap['super']["config"]["business_list"] = model_config["business_list"]
        if model_config.get("results"):
            self.modelMap['super']["config"]["results"] = model_config["results"]

        self.modelMap['super'][constants.MODEL_F_TSMIXER].init_config(self.modelMap['super']["config"])
    def init_param(self, config):
        # set parameters
        self.logger.info(f"=> init parameters:{config}")

        self.tsmixer.init_param(config)

        return True
    def train(self, train_logger):
        pass
    def end_train(self):
        pass

    # API for serving
    def init_serve(self, reload=False):
        pass

    def serve(self, header, data_dict):

        target_id = header['target_id']
        sys_id = header['sys_id']

        self.sys_id = sys_id
        self.target_logger = self.modelMap['super']["logger"]

        # ==== validation check ====
        c1 = 3 # 최근 c1 분 데이터 검사
        c2 = 5 # 피처당 c2개의 nan 허용

        input_df = pd.DataFrame.from_dict(data_dict)
        input_df = input_df.loc[:, input_df.isnull().sum() != len(input_df)]
        input_df = input_df.where(pd.notnull(input_df), None)

        has_all_nans_last_c1 = input_df.iloc[-c1:].isna().all(axis=1)
        has_all_nans_count_c2 = input_df.isna().all(axis=1)
        if has_all_nans_last_c1.all():
            self.target_logger.debug(f"[{sys_id}_{self.inst_type}_{target_id}] "
                                     f"{self.inst_type}_{target_id} - invalid dataset(최근 3분치 데이터 없음) \n"
                                     f"{has_all_nans_last_c1[has_all_nans_last_c1].index}")
            raise ModuleException("E704")
        elif has_all_nans_count_c2.sum() > c2:
            self.target_logger.debug(f"[{sys_id}_{self.inst_type}_{target_id}] "
                                     f"{self.inst_type}_{target_id} - invalid dataset(6분 이상의 데이터 없음) \n"
                                     f"{has_all_nans_count_c2[has_all_nans_count_c2].index}")
            raise ModuleException("E704")
        else:
            self.target_logger.debug(f"[{sys_id}_{self.inst_type}_{target_id}] "
                                     f"{self.inst_type}_{target_id} - valid dataset")

        model_config_path = self._get_model_config_path(self.model_dir, sys_id, target_id)
        model_config_key = REDISAI.make_redis_model_key(model_config_path, "")
        model_config_key = model_config_key.replace(".json", "")


        try:
            model_config = REDISAI.inference_json(model_config_key)
        except Exception as e:
            self.target_logger.error(f"sys_id:[{self.sys_id}], target_id:[{target_id}] cannot load model_config from REDISAI")
            raise Exception(e)

        try:
            self._update_model_config(model_config)
        except Exception as e:
            self.target_logger.error(f"target_id[{target_id}]: {e}")
            raise ModuleException("E904")

        self.target_logger.debug(f"{target_id} serving data length info {len(input_df)}")

        business_list = header.get("business_list", None) if header is not None else None
        if business_list is None:
            self.target_logger.info(f"{Errors.E824.desc}")
            raise ModuleException("E824")

        self.target_logger.info(f"{target_id}_input_shape : {input_df.shape}")

        time = input_df['time'].iloc[-1]
        feats = model_config['results']['tsmixer']['features']

        pred_result = None
        error_code = None
        error_msg = None

        self.target_logger.info(f"=========== [{target_id}] Start Serving ===========")

        try:
            pred_result = self.modelMap['super'][constants.MODEL_S_TSMIXER].predict(input_df)
        except ModuleException as me:
            self.target_logger.exception(f"Algorithm ModuleException Cause: {me.error_msg}")
            error_code = me.error_code
            error_msg = me.error_msg
        except Exception as e:
            self.target_logger.exception(f"Unknown Algorithm Exception Cause: {e}")
            error_code = f"{Errors.E900.value}"
            error_msg = f"{Errors.E900.desc}"


        tsmxier_pred_dict = self.make_result_format(time, pred_result, feats, target_id)

        res = {
            "pred_tsmixer": tsmxier_pred_dict,
            "errno": error_code,
            "errmsg": error_msg
        }

        self.target_logger.info(f"=========== [{target_id}] End Serving =========")

        return None, res, None, None

    def update_service_status(self, target_id=None, status="down"):
        if status == "down":  # service down
            if target_id in self.modelMap:
                self.modelMap['super']["logger"].info(f"[service down] target {target_id} down !!")
                self.serving_logger.info(f"[service down] target {target_id} !!")
                self.modelMap.pop(target_id)
                return True
            else:
                raise ModuleException("E713")
        else:  # service up
            self.serving_logger.info(f"[service up] target {target_id} !!")
            self.load(target_id)
            return True

    def end_serve(self):
        pass

    def get_debug_info(self):
        pass

    def make_result_format(self, time, preds, feats, target_id):
        res = {
            "keys": ["serving_time", "predict_time", "diff_minute", "sys_id", "inst_type", "target_id",  "metric", "predict_value"],
            "values": [],
        }

        if preds is None:
            return res

        tt = datetime.datetime.strptime(time, constants.INPUT_DATETIME_FORMAT)
        predict_time = [str(tt + datetime.timedelta(minutes=x+1)) for x in range(30)]


        for idx, feat in enumerate(feats):
            for i in range(preds.shape[0]):
                res['values'].append([time, predict_time[i], i+1, self.sys_id, self.inst_type, target_id, feat, np.maximum(preds[i][idx], 0)])
        return res