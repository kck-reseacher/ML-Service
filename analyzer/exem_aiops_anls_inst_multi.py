import datetime
import json
import os
import pandas as pd

from pathlib import Path
from pprint import pformat

from analyzer import aimodule
from common import aicommon, constants
from common.aiserverAPI import ServerAPI
from common.error_code import Errors
from common.memory_analyzer import MemoryUtil
from common.constants import SystemConstants as sc
from common.module_exception import ModuleException

from common.redisai import REDISAI


class ExemAiopsAnlsInstMulti(aimodule.AIModule):
    def __init__(self, config, logger):
        """
        instance의 각 지표(metric)의 "이상 탐지"와 "예측"을 함
        - 각 인스턴스(was, host(os), db, tp, web, ...) 마다 분석하고 인스턴스의 모든 타켓을 관리하여 처리함.
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
         - 이상 탐지
             - Dynamic Baseline
             - SeqAttn

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
        serving_logger : multi 모듈에서 serving.log 에 찍히는 로그
        ----------
        target_logger : multi 모듈 {AIMODULE_HOME}/logs/{sys_id}/{module_name}/{inst_type}/all/{target_id}/{module_id}_{inst_type}_{sys_id}_{target_id}.log
        """

        self.serving_logger = logger
        self.target_logger = None

        self.inst_type = config["inst_type"]
        self.module_id = config["module"]
        self.sys_id = ""
        self.log_dir = config["log_dir"]
        self.model_dir = config["model_dir"]
        self.temp_dir = config["temp_dir"]
        self.db_conn_str = config["db_conn_str"]
        self.anls_inst_template_path = config["anls_inst_template_path"]

        self.serving_logger.info(f"\t\t [Init] Create {pformat(config['inst_type'])} instance")

        self.modelMap = {'super':{}}

        self.mu = MemoryUtil(logger)

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
            logger_dir, logger_name, "serving", f"{sc.EXEM_AIOPS_ANLS_INST_MULTI}_{self.inst_type}"
        )

        return logger

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
        config["parameter"] = {} # parameter
        config["model_dir"] = str(Path(self.model_dir))
        config["temp_dir"] = self.temp_dir
        config["log_dir"] = self.log_dir
        config["db_conn_str"] = self.db_conn_str

        Path(config["model_dir"]).mkdir(exist_ok=True, parents=True)

        # train model config
        if config["model_dir"] is None:
            self.modelMap['super']["logger"].info(
                "[warning] model directory is not found: ", config["model_dir"]
            )
            return False
        else:
            model_config_path = Path(self.anls_inst_template_path)

            model_config = json.loads(Path(model_config_path).read_text(encoding="utf-8"))  # model_config.json 로드

            # 학습 지표 셋 설정
            if model_config.get("data_set"):
                config["data_set"] = model_config["data_set"]
            if model_config.get("weight"):
                config["weight"] = model_config["weight"]
            if model_config.get("business_list"):
                config["business_list"] = model_config["business_list"]
            if model_config.get("parameter"):
                config["parameter"] = model_config["parameter"]

        self.modelMap['super']["config"] = config
        self.modelMap['super']["logger"].info("[load] start model!")

        service_id = f"{self.module_id}_{config['target_id']}"

        # 개별 알고리즘 인스턴스 생성 및 모델 로딩
        self.modelMap['super']["logger"].info("[load] start creating instances and model loading")

        self.modelMap['super']["load_status"] = True
        for algorithm in constants.ANOMALY_TYPE_ALGO_LIST:
            if config["parameter"]["train"].get(algorithm, None) is not None:
                self.modelMap['super'][f"use_{algorithm}"] = config["parameter"]["train"][algorithm].get("use", False)

                if self.modelMap['super'][f"use_{algorithm}"]:
                    module_name = constants.ALGO_MODULE_CLASS_NAME[algorithm]['module_name']
                    class_name = constants.ALGO_MODULE_CLASS_NAME[algorithm]['class_name']

                    aimodule_path = Path(os.environ.get(sc.MLOPS_SERVING_PATH)) / "algorithms"
                    path = str(aimodule_path)

                    # gdn 경로
                    if algorithm in [constants.MODEL_S_GDN]:
                        path += '/gdn'

                    # 기존 경로에 s2s_attn 디렉토리 경로 추가
                    if algorithm in [constants.MODEL_S_S2S, constants.MODEL_S_SEQATTN]:
                        path += '/s2s_attn'

                    try:
                        target_class = aicommon.Utils.get_module_class(module_name, class_name, path)
                        self.modelMap['super'][algorithm] = target_class(service_id, config, logger)

                    except MemoryError as error:
                        self.modelMap['super']["logger"].exception(
                            f"[Error] Unexpected memory error during serving : {error}"
                        )
                        aicommon.Utils.print_memory_usage(logger)
                        self.mu.print_memory()

                        return False
                    except Exception as exception:
                        self.modelMap['super']["logger"].exception(
                            f"[Error] Unexpected exception during serving : {exception}"
                        )

            else:
                self.modelMap['super'][f"use_{algorithm}"] = False

        self.mu.print_memory()

        self.modelMap['super']["logger"].info("[load] finish creating instances and model loading")

        self.mu.print_memory()

        return True

    def _predict_by_algo(self, model, algo, target_id, **kwargs):
        pred_result = None
        error_code = None
        error_msg = None

        return_dict = {}

        try:
            if constants.MODEL_S_SEQATTN is algo:
                pred_result = model.predict(kwargs["input_df"])

            elif constants.MODEL_S_GDN is algo:
                pred_result = model.predict(kwargs["input_df"])

        except ModuleException as me:
            self.target_logger.exception(f"target {target_id} Algorithm ModuleException Cause ({algo}) : {me.error_msg}")
            error_code = me.error_code
            error_msg = me.error_msg

        except Exception as e:
            self.target_logger.exception(f"target {target_id} Unknown Algorithm Exception Cause ({algo}) : {e}")
            error_code = f"{Errors.E900.value}"
            error_msg = f"{Errors.E900.desc}"

        return_dict["pred_result"] = pred_result
        return_dict["error_code"] = error_code
        return_dict["error_msg"] = error_msg

        return return_dict

    def init_load(self, target_info, target_port, server_env: str):
        target_id = 'super'

        try:
            load_res = False
            load_res = self.load(target_id)
        except Exception as e:
            self.serving_logger.exception(f"load failed cause {e}")

        if not load_res:
            self.serving_logger.error(f"model load error")

    def _get_model_config_path(self, model_dir, sys_id=None, target_id=None):
        return str(Path(model_dir) / f"{sys_id}" / f"{self.module_id}" / f"{self.inst_type}" /f"{target_id}" / "model_config.json")

    def _update_model_config(self, model_config):
        self.modelMap['super']["config"]["target_id"] = model_config['target_id']
        self.modelMap['super']["config"]["sys_id"] = model_config['sys_id']
        self.modelMap['super']["config"]["parameter"] = model_config['parameter']
        self.modelMap['super']["config"]["model_dir"] = str(
            Path(self.model_dir) / f"{self.sys_id}" / f'{self.module_id}' / f'{self.inst_type}' / model_config[
                'target_id'])
        # 학습 지표 셋 설정
        if model_config.get("data_set"):
            self.modelMap['super']["config"]["data_set"] = model_config["data_set"]
        if model_config.get("weight"):
            self.modelMap['super']["config"]["weight"] = model_config["weight"]
        if model_config.get("business_list"):
            self.modelMap['super']["config"]["business_list"] = model_config["business_list"]
        if model_config.get("parameter"):
            self.modelMap['super']["config"]["parameter"] = model_config["parameter"]
        if model_config.get("results"):
            self.modelMap['super']["config"]["results"] = model_config["results"]

        for algorithm in constants.ANOMALY_TYPE_ALGO_LIST:
            self.modelMap['super'][f"use_{algorithm}"] = model_config["parameter"]["train"].get(algorithm, {}).get(
                "use", False)
            if self.modelMap['super'][f"use_{algorithm}"]:
                self.modelMap['super'][algorithm].init_config(self.modelMap['super']["config"])
                self.modelMap['super'][algorithm].model_dir = str(
                    Path(self.model_dir) / f"{self.sys_id}" / f'{self.module_id}' / f'{self.inst_type}' / model_config[
                        'target_id'])

    def serve(self, header, data_dict):

        target_id = header['target_id']  # self.target_id
        sys_id = header['sys_id']
        self.sys_id = sys_id
        self.target_logger = self.modelMap['super']["logger"]
        # ==== validation check ====
        c1 = 3  # 최근 c1 분 데이터 검사
        c2 = 5  # 피처당 c2개의 nan 허용
        e = pd.to_datetime(header['predict_time'])
        s = e - pd.Timedelta(minutes=59)
        t_index = pd.DatetimeIndex(pd.date_range(start=s, end=e, freq='T'))
        input_df = pd.DataFrame.from_dict(data_dict)
        input_df = input_df.sort_values(by="time")
        input_df.index = pd.to_datetime(input_df["time"])
        input_df = input_df.reindex(t_index)
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
        # ==========================

        model_config_path = self._get_model_config_path(self.model_dir, sys_id, target_id)
        model_config_key = REDISAI.make_redis_model_key(model_config_path, "")
        model_config_key = model_config_key.replace(".json", "")
        try:
            model_config = REDISAI.inference_json(model_config_key)
        except Exception as e:
            self.target_logger.error(f"sys_id:[{self.sys_id}], target_id:[{target_id}] cannot load model_config from REDISAI")
            raise Exception(e)

        # use_algorithm update
        # seq2seq init_config
        # dbsln init_config
        # config update

        ## update model_config
        try:
            self._update_model_config(model_config)
        except Exception as e:
            self.target_logger.error(f"target_id[{target_id}]: {e}")
            raise ModuleException("E904")

        self.target_logger.debug(f"{target_id} serving data length info {len(input_df)}")

        config = self.modelMap['super']["config"]

        # 모델 사용여부
        use_gdn = constants.USE_GDN and self.modelMap['super'].get("use_gdn", False)
        model_dict = {}

        # 모델
        for algo in constants.ANOMALY_TYPE_ALGO_LIST:
            model_dict[algo] = self.modelMap['super'].get(algo, None)

        self.target_logger.info(f"=========== [{target_id}] Start Serving ===========")

        input_dict = pd.DataFrame(input_df.iloc[-1]).to_dict()
        data_dict_cur = input_dict[list(input_dict.keys())[0]]

        self.target_logger.debug(f"serving with data:\n{data_dict_cur}")

        serving_datetime_str = data_dict_cur["time"]
        self.target_logger.info(f"serving data time: {serving_datetime_str}")

        gdn_pred_dict = {}

        for algo in constants.ANOMALY_TYPE_ALGO_LIST:
            model = model_dict[algo]

            if constants.MODEL_S_GDN is algo and use_gdn:
                gdn_pred_dict = self._predict_by_algo(model, algo, target_id, input_df=input_df)

        # predict 결과가 없는 경우 빈 dict 형태 return
        res_gdn_result = {}

        # =================== gdn start ===================
        if use_gdn and bool(gdn_pred_dict):
            if gdn_pred_dict.get("error_code") is None:
                res_gdn_result = gdn_pred_dict["pred_result"]
            else:
                res_gdn_result = aicommon.Utils.set_except_msg(gdn_pred_dict)
        # =================== gdn end ===================

        res = {
            "detect_gdn": res_gdn_result
        }

        self.target_logger.info(f"=========== [{target_id}] End Serving =========")

        return None, res, None, None

    def end_serve(self):
        pass

    def insert_serving_seqattn_performance(self, time, result, target_id):
        res = {'keys': ['time', 'sys_id', 'inst_type', 'target_id', 'name', 'real_value',
                        'predict_value', 'predict_lower', 'predict_upper', 'anomaly'],
               'values': []}

        if result is None:
            return res

        for feat in result['features']:
            try:
                is_anomaly = bool(result[f"{feat}_real"] is not None and (result[f"{feat}_real"] > result[f"{feat}_upper"] or result[f"{feat}_real"] < result[f"{feat}_lower"]))
                res['values'].append(
                    [time, self.sys_id, self.inst_type, target_id, feat, result[f"{feat}_real"], result[f"{feat}_pred"],
                     result[f"{feat}_lower"], result[f"{feat}_upper"], is_anomaly])
            except:
                continue
        return res

    def _check_and_create_target_logger(self, target_id):
        is_logger = self.check_multi_logger(target_id, self.modelMap)

        if is_logger:
            self.modelMap['super'] = {}
            self.modelMap['super']["logger"] = self.create_logger(target_id)
        else:
            if not self.modelMap['super']["logger"].hasHandlers():
                self.modelMap['super']["logger"] = self.create_logger(target_id)

    # API for training
    def train(self, train_logger):
        pass

    def test_train(self):
        pass

    def end_train(self):
        pass