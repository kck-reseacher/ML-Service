import json
import re
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from pprint import pformat
import pandas as pd
import psycopg2 as pg2
from confluent_kafka import Producer
import asyncio
import nest_asyncio

import algorithms.logseq.config.default_regex as basereg
from algorithms.ocdigcn.ocdigcn import OCDiGCN
from analyzer import aimodule
from analyzer.exem_aiops_anls_log_sparse import ExemAiopsAnlsLogSparse
from common.clickhouse_client import get_client, close_client
from common import aicommon, constants
from common.aiserverAPI import ServerAPI
from common.constants import SystemConstants as sc
from common.error_code import Errors
from common.memory_analyzer import MemoryUtil
from common.module_exception import ModuleException
from common.redisai import REDISAI
from common.system_util import SystemUtil
from resources.config_manager import Config

past_input_dict = dict()
_past_input_dict = dict()

class ExemAiopsAnlsLogMulti(aimodule.AIModule):
    def __init__(self, config, logger):
        """
        Parameters
        ----------
        config : 각종 설정 파일, 추가로 서버에서 설정 파일을 받아 옴.
        logger : multi 모듈에서 serving.log 에 찍히는 로그
        """

        self.serving_logger = logger
        self.target_logger = None

        self.inst_type = config["inst_type"]
        self.module_id = config["module"]
        self.sys_id = config["sys_id"]
        self.log_dir = config["log_dir"]
        self.model_dir = config["model_dir"]
        self.db_conn_str = config["db_conn_str"]
        self.logpresso_decode_config = config["logpresso"]
        self.key_prefix = None

        self.serving_logger.info(f"\t\t [Init] Create {pformat(config['inst_type'])} instance")

        self.modelMap = {'super':{}}
        self.serverAPI = ServerAPI(config, logger)

        self.mu = MemoryUtil(logger)
        self.logseq_window_size = constants.LOGSEQ_N_WINDOW
        self.default_anomaly_threshold = 5
        self.log_length = 300

        self.sparselog_analyzer = ExemAiopsAnlsLogSparse(config, logger)
        self.modelMap_sparse = {'super': {}}
        self.datetime_regex_dic = {}

    def create_logger(self, target_id):
        """
        logger를 만드는 함수.
        기존처럼 load 함수 내에서 로거를 만들면 동일한 로거가 만들어지고 Handler만 추가되므로 중복 로깅됨
        참고 : https://5kyc1ad.tistory.com/269
        Parameters
            target_id : 타겟아이디
        Returns : 로거
        -------

        """
        logger_dir = str(Path(self.log_dir) / self.inst_type / target_id)
        logger_name = f"{self.module_id}_{self.inst_type}_{self.sys_id}_{target_id}"
        logger = self.create_multi_logger(logger_dir, logger_name, "serving", f"{sc.EXEM_AIOPS_ANLS_LOG_MULTI}_{self.inst_type}")
        return logger

    def load(self, target_id=None, reload=False):
        # 이미 모델이 로드된 상태라면 pass
        if reload is False and target_id in self.modelMap and constants.MODEL_S_DIGCN in self.modelMap["super"].keys():
            return True

        if reload is True:
            if target_id in self.modelMap:
                self.modelMap["super"]["logger"].info("[reload] start reload model")
            else:
                self.serving_logger.info(f"[reload] fail, target_id: {target_id} is not exist!!")
                return False

        is_logger = self.check_multi_logger(target_id, self.modelMap)

        if is_logger:
            self.modelMap["super"] = {}
            self.modelMap["super"]["logger"] = self.create_logger(target_id)
        else:
            if not self.modelMap["super"]["logger"].hasHandlers():
                self.modelMap["super"]["logger"] = self.create_logger(target_id)

        logger = self.modelMap["super"].get("logger")
        logger.info("[load] start loading model")

        config = {}
        config["sys_id"] = self.sys_id
        config["inst_type"] = self.inst_type
        config["module"] = self.module_id
        config["target_id"] = target_id
        config["model_dir"] = str(Path(self.model_dir) / self.sys_id / self.module_id / self.inst_type / target_id)
        config["log_dir"] = self.log_dir
        config["db_conn_str"] = self.db_conn_str

        self.key_prefix = f"{self.sys_id}/{self.module_id}/{self.inst_type}/{target_id}"
        parameter_key = f"{self.key_prefix}/parameter"

        # REDIS에 키 없을 경우 --> parameter_kdy = 102/exem_aiops_anls_log/log/{target_id}/parameter
        if not REDISAI.exist_key(parameter_key):
            response = self.serverAPI.update_redis_parameter(self.sys_id, self.module_id, self.inst_type, target_id)
            if not response:
                return False
        parameter = REDISAI.get(parameter_key)
        config["parameter"] = json.loads(parameter)

        Path(config["model_dir"]).mkdir(exist_ok=True, parents=True)

        self.modelMap["super"]["config"] = config
        self.modelMap["super"]["logger"].info("[load] start model!")

        self._get_preset_from_pg(target_id)
        # digcn(패턴이상탐지) 분석 여부
        self.modelMap["super"]["use_digcn"] = config["parameter"][constants.MODEL_S_DIGCN]["use"]
        # 인스턴스 생성
        try:
            self.modelMap["super"][constants.MODEL_S_DIGCN] = OCDiGCN(config=config, logger=logger)

        except Exception as exception:
            self.modelMap["super"]["logger"].exception(f"[Error] Unexpected exception during serving : {exception}")
            raise ModuleException("E777")

        onnx_model_path = str(Path(config["model_dir"]) / f"{constants.MODEL_S_DIGCN}" / "digcn_model.onnx")
        onnx_model_key = REDISAI.make_redis_model_key(onnx_model_path, ".onnx")
        if not REDISAI.exist_key(onnx_model_key):
            msg = f"ONNX model {onnx_model_key} not found"
            self.modelMap["super"]["logger"].critical(msg)
            raise ModuleException("E850")

        self.modelMap["super"]["logger"].info("[load] end loading model")
        return True

    # 사용자에 의해 config 변경시 새로운 서버 알림으로 config로 초기화
    def init_param(self, target_id, config):
        self.modelMap["super"]["logger"].info(f"[init param] initialize target {target_id} : parameters:{config}")
        return True

    def query_data_from_ch(self, target_id, standard_datetime, log_type):
        df = pd.DataFrame()
        start_time = time.time()
        try:
            client = get_client()
            query_str = f"""select time, offset, left(message, {self.log_length})
                            from dm_{log_type}
                            where target_id = '{target_id}' and time >= '{standard_datetime}' and time < '{pd.to_datetime(standard_datetime) + timedelta(minutes=1)}' 
                        """
            result = client.query(query_str)
            df = pd.DataFrame(result.result_rows, columns=['time', 'line_no', 'msg'])
            if len(df) > 0:
                df['time'] = df['time'].dt.tz_localize(None)
        except Exception as ex:
            self.target_logger.exception(f"error occurred in ClickHouse query.  {ex}")
        if client:
            close_client()
        self.target_logger.info(f"target_id: {target_id}, serving time:{standard_datetime}, LP data length : {len(df)}, elapsed time: {time.time() - start_time}")
        return df

    def _get_preset_from_pg(self, target_id):
        conn, cursor = None, None
        regex_list, delimiters = [], " "

        try:
            if REDISAI.exist_key(f"{self.key_prefix}/regex_list") and REDISAI.exist_key(f"{self.key_prefix}/delimiters"):
                regex_list = json.loads(REDISAI.get(f"{self.key_prefix}/regex_list"))
                delimiters = json.loads(REDISAI.get(f"{self.key_prefix}/delimiters"))
            else:
                conn = pg2.connect(self.modelMap["super"]["config"]["db_conn_str"])
                cursor = conn.cursor(cursor_factory=pg2.extras.RealDictCursor)

                query = "select re.regset_id, re.delimiter, re.regex, re.replace_str " \
                        "from ai_config_log_regex re left join xaiops_config_log xl " \
                        "on re.regset_id = xl.regset_id " \
                        "where xl.target_id = %s"
                cursor.execute(query, (target_id,))
                query_result = [dict(record) for record in cursor]
                if len(query_result) > 0:
                    for item in query_result:
                        if item["delimiter"]:
                            delimiters = item["regex"]
                        else:
                            regex = [item["regex"], item["replace_str"]]
                            regex_list.append(regex)
                else:
                    regex_list = basereg.COMMON_REGEX
                    delimiters = basereg.COMMON_DELIMITER
                REDISAI.set(f"{self.key_prefix}/regex_list", regex_list)
                REDISAI.set(f"{self.key_prefix}/delimiters", delimiters)

            self.modelMap["super"]["config"]["preset"] = regex_list
            self.modelMap["super"]["config"]["delimiters"] = delimiters
            self.modelMap["super"]["logger"].info(f"preset after ===={self.modelMap['super']['config']['preset']}")

        except Exception as ex:
            self.target_logger.exception(f"unable to get regex from the database : {ex}\n\n{traceback.format_exc()}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def _update_model_config(self, model_config):
        self.modelMap['super']["config"]["target_id"] = model_config['target_id']
        self.modelMap['super']["config"]["sys_id"] = model_config['sys_id']
        self.modelMap['super']["config"]["parameter"] = model_config['parameter']
        self.modelMap['super']["config"]["model_dir"] = str(Path(self.model_dir) / f"{self.sys_id}" / f'{self.module_id}' / f'{self.inst_type}' / model_config['target_id'])
        self.modelMap['super']["config"][constants.MODEL_S_DIGCN] = model_config['results'][constants.MODEL_S_DIGCN]

        self.modelMap['super'][constants.MODEL_S_DIGCN].init_config(self.modelMap['super']["config"])
        return model_config['results']['log_type']

    def _get_model_config_path(self, model_dir, sys_id=None, target_id=None):
        return str(Path(
            model_dir) / f"{sys_id}" / f"{self.module_id}" / f"{self.inst_type}" / f"{target_id}" / "model_config")

    def replace_preset_and_remove_number(self, input_df):
        start_time = time.time()
        input_df['msg_transform'] = input_df['msg']
        self.target_logger.debug(f"===== original msg >>>>>: {input_df['msg_transform'].iloc[0]}")

        if self.modelMap["super"]['config']['preset']:
            self.target_logger.debug(f"preset_list: {self.modelMap['super']['config']['preset']}")
            for preset in self.modelMap["super"]['config']['preset']:
                input_df['msg_transform'] = list(map(lambda x: re.sub(preset[0], preset[1], x), input_df['msg_transform']))
            self.target_logger.debug(f"===== after preset replacing >>>>>  {input_df['msg_transform'].iloc[0]}")

        input_df['msg_transform'] = list(map(lambda x: re.sub(r'[^\w\s]', ' ', x), input_df['msg_transform']))   # 특수문자 제거
        input_df['msg_transform'] = list(map(lambda x: re.sub(r'\d', '', x), input_df['msg_transform']))   # 숫자 제거
        self.target_logger.info(f"===== after number removing  >>>>> {input_df['msg_transform'][0]}")
        self.target_logger.info(f"replace_preset_and_remove_number ===> elapsed time: {time.time() - start_time}")
        return input_df

    async def serve(self, header, data_dict):
        self.sys_id, self.inst_type = header["sys_id"], header["inst_type"]
        target_id, standard_datetime = header["target_id"],  header["predict_time"]   # 실제 서빙 데이터에 들어있는 DM 시간

        self.load(target_id)
        self.sparselog_analyzer.load(target_id)

        self.target_logger = self.modelMap["super"]["logger"]
        self.target_logger.info("=========== Start Serving ===========")

        model_config_path = self._get_model_config_path(self.model_dir, self.sys_id, target_id)
        model_config_key = REDISAI.make_redis_model_key(model_config_path, "")
        model_config = REDISAI.inference_json(model_config_key)
        log_type = self._update_model_config(model_config)

        # Read Redis
        past_input_key = f"{self.key_prefix}/past_input_df"
        if REDISAI.exist_key(past_input_key):
            _past_input_df = REDISAI.inference_pickle(past_input_key)
        else:
            _past_input_df = pd.DataFrame()

        data_dict = self.query_data_from_ch(target_id, standard_datetime, log_type)

        if len(data_dict) == 0:
            res = self.make_empty_body(target_id)
            header = self.make_log_serving_header_dict(target_id, standard_datetime, data_dict)
            self.target_logger.info(f"no serving data, LP data length : {len(data_dict)}")
            self.target_logger.info("=========== End Serving ===========\n")
            return header, res, None, None

        data_dict = self.replace_preset_and_remove_number(data_dict)

        try:
            input_df = aicommon.Utils.make_log_serving_data(_past_input_df, data_dict)
        except Exception as ServingDataError:
            self.target_logger.error(f"{ServingDataError}")
            self.target_logger.error(f"{Errors.E710.desc}")
            raise ModuleException("E710")

        if len(input_df) <= self.logseq_window_size:
            past_input_df = input_df
            REDISAI.save_body_to_redis(past_input_key, past_input_df)
            self.target_logger.error(f"serving data not enough {len(input_df)}, logseq_window_size {self.logseq_window_size}")
            raise ModuleException("E704")
        else:
            past_input_df = input_df.iloc[-self.logseq_window_size:]   # window 사이즈만큼 저장
            REDISAI.save_body_to_redis(past_input_key, past_input_df)

        self.target_logger.debug(f"make_log_serving_window_data info {len(input_df)}")
        self.target_logger.debug(f"modelMap[{target_id}].keys(): {self.modelMap['super'].keys()}")

        tmp_input_df = input_df.copy()
        digcn_result, digcn_header, sparse_result, sparse_header = await self.multi_serve_process(target_id, data_dict, standard_datetime, tmp_input_df)
        self.KafkaMessageProducer(log_type, target_id, standard_datetime, digcn_header, digcn_result, sparse_result)

        res = {"detect_digcn": digcn_result}
        self.target_logger.debug(f"digcn_header : {str(digcn_header)}")
        self.target_logger.info("=========== End Serving ===========\n")

        return digcn_header, res, None, None

    async def multi_serve_process(self, target_id, data_dict, standard_datetime, tmp_input_df):
        start_all = time.time()
        digcn_result, digcn_header, sparse_result, sparse_header = None, None, None, None
        '''predict digcn detection'''
        digcn_serve = self.modelMap["super"][constants.MODEL_S_DIGCN].predict(standard_datetime, tmp_input_df)

        if self.sparselog_analyzer.modelMap_sparse["super"]['use_sparselog']:
            ''' predict sparselog detection '''
            sparse_serve = self.sparselog_analyzer.modelMap_sparse["super"][constants.MODEL_S_SPARSELOG].predict(standard_datetime, data_dict)
            results = await asyncio.gather(digcn_serve, sparse_serve)
            digcn_result, digcn_header, sparse_result, sparse_header = results[0][0], results[0][1], results[1][0], results[1][1]

            self.insert_sparse_keyword_result_to_pg(standard_datetime, target_id, sparse_header, sparse_result)

        else:
            self.target_logger.info(f"[sparselog] analysis is not active.")
            results = await asyncio.gather(digcn_serve)
            digcn_result, digcn_header = results[0][0], results[0][1]

        self.target_logger.info(f'[Asynchronous] time taken: {round(time.time() - start_all, 5)}')
        if 'error_code' in digcn_result.keys():
            digcn_result = aicommon.Utils.set_except_msg(digcn_result)
        else:
            self.insert_digcn_result_to_pg(standard_datetime, digcn_header, digcn_result)

        return digcn_result, digcn_header, sparse_result, sparse_header

    def KafkaMessageProducer(self, log_type, target_id, standard_datetime, digcn_header, digcn_result, sparse_result, topic_name="alarm_log"):
        os_env = SystemUtil.get_environment_variable()
        py_config = Config(os_env[sc.MLOPS_SERVING_PATH], os_env[sc.AIMODULE_SERVER_ENV]).get_config()
        bootstrap_servers = f"{py_config['kafka']['host']}:{py_config['kafka']['port']}"  # Kafka broker address
        producer_conf = {'bootstrap.servers': bootstrap_servers}  # Kafka consumer configuration

        message = {
            "inst_type": log_type,
            "target_id": target_id,
            "time": standard_datetime,
            "data": {}
        }

        # digcn_result에 error_code 키가 있을 경우 = 그래프 데이터 없거나 redis 키 정상 로드 안됐을 경우
        # 그 외 정상 predict 수행했을 경우 = digcn_result['values'] 존재
        if 'error_code' not in digcn_result.keys():
            if digcn_header:
                message["data"]["digcnLog"] = []
                for digcn_val in digcn_result['values']:
                    message["data"]["digcnLog"].append({"content": digcn_val[7],
                                                        "line_num": digcn_val[5],
                                                        "anomaly_score": digcn_val[8]})
            else:
                self.target_logger.debug(f"""digcn_result['values'] is not exist. {digcn_result}""")

        if len(sparse_result['values']) > 0:
            for sparse_val in sparse_result['values']:
                log_info = {
                    "log_path": sparse_val[1],
                    "content": sparse_val[2],
                    "line_num": sparse_val[3],
                    "probability": sparse_val[4],
                }
                detection_type = sparse_val[7]
                if detection_type == 1 or detection_type == 3:  # keyword
                    if "keywordLog" not in message["data"].keys():
                        message["data"]["keywordLog"] = []
                    log_info["keyword"] = sparse_val[6]
                    message["data"]["keywordLog"].append(log_info)

                if detection_type == 1 or detection_type == 2:  # sparse
                    if "sparseLog" not in message["data"].keys():
                        message["data"]["sparseLog"] = []
                    message["data"]["sparseLog"].append(log_info)

        try:
            if len(message['data'].keys()) > 0:    # digcn, sparse, keyword 중 하나라도 탐지 결과 있을 경우
                producer = Producer(producer_conf)
                producer.produce(topic=topic_name, value=json.dumps(message, default=str))
                producer.flush()  # topic에 message가 들어갈 때까지 대기
        except Exception as ex:
            self.target_logger.exception(f"failed to produce message with kafka. : {ex}\n\n{traceback.format_exc()}")

    def insert_digcn_result_to_pg(self, standard_datetime, digcn_header, digcn_perf):
        conn, cursor = None, None
        insert_list = [(standard_datetime, pd.to_datetime(tmp[0]).strftime("%Y-%m-%d %H:%M:%S"), tmp[3],
                        int(tmp[5]), str(tmp[7]), float(tmp[8]))
                       for tmp in digcn_perf["values"]]

        query_digcn = 'INSERT INTO ai_result_log_digcn_anomaly'\
                        '("standard_time","real_time","target_id","line_no","content","anomaly_score")'\
                        'VALUES %s;'

        query_digcn_summary = "INSERT INTO ai_result_log_digcn_summary" \
                               '("time","target_id","anomaly_count","total_count","anomaly","anomaly_threshold")' \
                               "VALUES (%s, %s, %s, %s, %s, %s)"
        try:
            if digcn_header and int(digcn_header["anomaly_count"]) >= 1:
                conn = pg2.connect(self.db_conn_str)
                cursor = conn.cursor(cursor_factory=pg2.extras.DictCursor)

                pg2.extras.execute_values(cursor, query_digcn, insert_list)
                cursor.execute(query_digcn_summary,
                               (digcn_header["time"].strftime("%Y-%m-%d %H:%M"), digcn_header["target_id"], int(digcn_header["anomaly_count"]),
                                int(digcn_header["total_count"]), digcn_header["anomaly"], int(digcn_header["anomaly_threshold"])),)
                conn.commit()
                self.target_logger.info(f"success digcn result to pg! module_serving_result_log_digcn's row cnt: {len(insert_list)}, summary's anomaly cnt: {int(digcn_header['anomaly_count'])}")

        except Exception as ex:
            self.target_logger.exception(f"fail insertion digcn result to pg : {ex}\n\n{traceback.format_exc()}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def insert_sparse_keyword_result_to_pg(self, standard_datetime, target_id, sparse_header, sparse_result):
        conn, cursor = None, None
        sparse_list, keyword_list = [], []
        for val in sparse_result['values']:  # 0:real_time, 1:log_path, 2:content, 3:line_no, 4:probability, 6:keyword
            sparse_val = (standard_datetime, val[0], target_id, val[1], val[3], val[2], val[4])
            keyword_val = (standard_datetime, val[0], target_id, val[1], val[3], val[2], val[6])
            if val[7] == 2:  # only sparse log detection
                sparse_list.append(sparse_val)
            elif val[7] == 3:  # only keyword log detection
                keyword_list.append(keyword_val)
            else:  # both sparse and keyword log detection
                sparse_list.append(sparse_val)
                keyword_list.append(keyword_val)

        query_sparse_anomaly = 'INSERT INTO ai_result_log_sparse_anomaly '\
                               '("standard_time","real_time","target_id","log_path","line_no","content","probability")'\
                               'VALUES %s;'

        query_sparse_summary = "INSERT INTO ai_result_log_sparse_summary " \
                               '("time","target_id","occurrences_cnt","prop_cnt","anomaly")' \
                               "VALUES (%s, %s, %s, %s, %s)"

        query_keyword_anomaly = 'INSERT INTO ai_result_log_keyword_anomaly ' \
                               '("standard_time","real_time","target_id","log_path","line_no","content","keyword")' \
                               'VALUES %s;'

        query_keyword_summary = "INSERT INTO ai_result_log_keyword_summary " \
                               '("time","target_id","occurrences_cnt")' \
                               "VALUES (%s, %s, %s)"

        try:
            conn = pg2.connect(self.db_conn_str)
            cursor = conn.cursor(cursor_factory=pg2.extras.DictCursor)

            if len(sparse_list) > 0:
                cursor.execute(query_sparse_summary, (standard_datetime, target_id, len(sparse_list), sparse_header['prop_cnt'], len(sparse_list) > sparse_header['prop_cnt']),)
                pg2.extras.execute_values(cursor, query_sparse_anomaly, sparse_list)
                self.target_logger.info(f"success sparselog result to pg! row cnt: {len(sparse_list)}")

            if len(keyword_list) > 0:
                cursor.execute(query_keyword_summary, (standard_datetime, target_id, len(keyword_list)),)
                pg2.extras.execute_values(cursor, query_keyword_anomaly, keyword_list)
                self.target_logger.info(f"success keyword result to pg! row cnt: {len(keyword_list)}")

            conn.commit()

        except Exception as ex:
            self.target_logger.exception(f"fail insertion sparselog & keyword result to pg : {ex}\n\n{traceback.format_exc()}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def make_log_serving_header_dict(self, target_id, standard_datetime, data_dict, anomaly_count=0):
        header = {}
        serving_time = standard_datetime
        header["time"] = datetime.strptime(serving_time[:16], '%Y-%m-%d %H:%M') if len(serving_time) >= 16 else serving_time
        header["sys_id"] = self.sys_id
        header["target_id"] = target_id
        header["total_count"] = max(0, len(data_dict) - self.logseq_window_size)
        header["anomaly_count"] = anomaly_count
        if self.modelMap["super"]["use_digcn"]:
            header["anomaly_threshold"] = self.modelMap["super"]["config"][constants.MODEL_S_DIGCN]['anomaly_threshold']
        else:
            header["anomaly_threshold"] = self.default_anomaly_threshold
        header["anomaly"] = bool(anomaly_count > header["anomaly_threshold"])
        return header

    def make_empty_body(self, target_id):
        res = dict()
        if self.modelMap["super"]["use_digcn"]:
            res[f"{constants.ANOMALY_RES_KEY_PREFIX + constants.MODEL_S_DIGCN}"] = dict()
        return res


    def end_serve(self):
        pass

    def train(self, train_logger):
        pass

    def test_train(self):
        pass

    def end_train(self):
        pass
