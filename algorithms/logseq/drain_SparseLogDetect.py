try:
    import re2 as re
except ImportError as e:
    import re
import datetime
import json
import time
import traceback
from pathlib import Path

import joblib
import pandas as pd
import psutil
import psycopg2 as pg2

from common import constants
from common.redisai import REDISAI

USE_MP = False
MP_MIN_ACTIVATE_LINES = 10000
MP_CPU_PORTION = 0.5
if USE_MP:
    from pathos import multiprocessing


class Drain:
    def __init__(self, target_id, logger=None, config=dict()):
        self.config = config
        self.logger = logger
        self.target_id = target_id

        self.model_dir = None
        self.model_path = self.get_model_path()
        self.model_dir_key = REDISAI.make_redis_model_key(self.model_dir)

        self.target_config = {}
        self.target_config_key = f"{self.model_dir_key}/config"
        if REDISAI.exist_key(self.target_config_key):
            self.target_config = json.loads(REDISAI.get(self.target_config_key))
        else:
            self._get_log_meta_from_pg()
            REDISAI.set(self.target_config_key, self.target_config)
        self.target_config.update(self.config["parameter"])
        self.rareRate_threshold = self.target_config['rare_rate']

        # counts for rare rate calculation and logging
        self.lasted_mins = int(time.time()/60)
        self.save_period = 5

        if self.target_config.get("alert_threshold") and self.target_config.get("alert_threshold") > 0:
            self.alert = True
            self.alert_threshold = self.target_config["alert_threshold"]
            self.logger.info(f"* realtime alert : {self.alert} ({self.alert_threshold} per minute)")
        self.logger.info(f"target_config:{self.target_config}")
        self.error_keyword_list = ['error', 'fatal', 'fail', 'exception', 'critical']

    def _get_log_meta_from_pg(self):
        """
        서빙 타겟에 해당하는 meta 정보 조회 후 target_config 업데이트
        """
        conn, cursor = None, None
        try:
            conn = pg2.connect(self.config["db_conn_str"])
            cursor = conn.cursor(cursor_factory=pg2.extras.RealDictCursor)

            query = 'SELECT log_id, target_id, log_path, regset_id ' \
                    'FROM xaiops_config_log ' \
                    'WHERE "target_id" = %s'
            cursor.execute(query, (self.target_id,))
            query_result = cursor.fetchone()
            if query_result is None:
                err_msg = f"log_id : {self.target_id} does not exist."
                raise Exception(err_msg)
            query_result = dict(query_result)
            self.logger.info(f"get_log_meta_from_pg ==> {query_result}")

            self.target_config.update(query_result)
        except Exception as ex:
            self.logger.exception(f"unable to get configuration from the database : {ex}\n\n{traceback.format_exc()}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def get_model_path(self):
        sys_id = self.config['sys_id']

        self.model_dir = str(Path(self.config["model_dir"]) / str(sys_id) / self.config['module'] / self.config['inst_type'] / self.target_id / constants.MODEL_S_SPARSELOG )
        model_file_path = f"{self.model_dir}/template_miner.pkl"

        return model_file_path

    def preprocess_log(self, log_data):
        """
        로그프레소 쿼리로 전달 받은 로그메세지 한 줄을 분석하여 키워드 검출 및 전처리 진행

        참고) logging 혹은 print 출력 사용 시 입출력 시간 지연으로 인해 multiprocessing의 이점을 취할 수 없으므로, 사용하지 않도록 한다.

        Args:
            log_data : Logpresso query로 전달받은 한 개 row의 로그메세지 정보(dict)
        """
        log_data = log_data[1].to_dict()
        logmessageL_raw = log_data["msg_transform"].strip()

        # 사용자 정의, 제외 문구 체크
        is_userkeyword, is_exclude_keyword = False, False
        matched_keywords, matched_exclude_keywords = [], []
        keyword_str, exclude_keyword_str = None, None

        user_keywords_list = set(self.target_config.get("include_keyword") + self.error_keyword_list)
        if user_keywords_list:
            for keyword in user_keywords_list:
                if keyword.lower() in logmessageL_raw.lower():
                    matched_keywords.append(keyword)
                    is_userkeyword = True  # insert user defined keyword matched log message into Postgres
            if is_userkeyword:
                keyword_str = ",".join(matched_keywords)
                self.logger.debug(f"\t*keyword [{keyword_str}] detected : {logmessageL_raw}")

        exclude_keywords_list = self.target_config.get("exclude_keyword")
        if exclude_keywords_list:
            for exclude_keyword in exclude_keywords_list:
                if exclude_keyword.lower() in logmessageL_raw.lower():
                    matched_exclude_keywords.append(exclude_keyword)
                    is_exclude_keyword = True  # insert user defined keyword matched log message into Postgres
            if is_exclude_keyword:
                exclude_keyword_str = ",".join(matched_exclude_keywords)
                self.logger.debug(f"\t*keyword [{exclude_keyword_str}] detected : {logmessageL_raw}")

        logmessage_lines = logmessageL_raw.split("\n")
        if len(logmessage_lines) > 1: # 최초 문장만 대표로 희소로그 분석 진행
            logmessageL_raw = logmessage_lines[0]

        # length 0 logs, which are anomaly cases
        if len(log_data["msg_transform"]) == 0:
            log_data["success"] = False
            return log_data

        # 후처리용 정보 저장
        log_data["success"] = True
        log_data["log_preprocessed"] = logmessageL_raw
        log_data["keyword"] = keyword_str
        log_data["exclude_keyword"] = exclude_keyword_str

        return log_data

    def process_drain(self, log_data):
        """
        similarity layer, output layer에서 Drain 트리 업데이트.
        신규 클러스터일 때 새로운 log cluster와 output cell을 생성하여 트리에 등록하고,
        매칭된 클러스터일 때 해당 클러스터를 업데이트 한다.

        Args:
            - log_data : 전처리 및 트리탐색이 완료된 로그데이터 dict
                - log_preprocessed : 전처리된 로그메세지. list of string
                - log_raw : 전처리 전 로그메세지

        Returns:
            - is_sparse : 입력된 로그메세지의 희소로그 판별 결과 bool
            - rare_rate : 해당 로그메세지가 속한 로그 클러스터의 희소율
            - matched_template : 해당 로그메세지가 속한 로그 클러스터의 템플릿
        """

        # initialize return values
        is_sparse = False
        rare_rate = 0

        log_preprocessed = log_data["log_preprocessed"]
        log_raw = log_data["msg"].strip()

        # search matching cluster using drain3
        match_result = self.template_miner.add_log_message(log_preprocessed)

        if match_result["change_type"] == 'cluster_created':   #change_type = ['cluster_created', 'none', 'cluster_template_changed']
            rare_rate = 1 / self.template_miner.drain.get_total_cluster_size()
            is_sparse = True
            self.logger.info(f"   new template detected. || original log msg: {log_raw}")
        else:  # match_cluster is not None
            rare_rate = match_result["cluster_size"] / self.template_miner.drain.get_total_cluster_size()
            if rare_rate < self.rareRate_threshold:
                is_sparse = True
                self.logger.debug(f"\t<TEMPLATE> {match_result['template_mined']}")

        matched_template = match_result["template_mined"]
        return is_sparse, rare_rate, matched_template

    def load_model_bin(self):
        """
        Loads model from a binary pickle file.
        Included log cluster data : root node / log cluseter list / outputcell list / pointer

        Args:
            - model_file_path: 모델 파일(.pkl) 경로
        """
        sparse_model_key = REDISAI.make_redis_model_key(self.model_path,'.pkl')
        if REDISAI.exist_key(sparse_model_key):
            self.template_miner = REDISAI.inference_joblib(sparse_model_key)
            self.logger.info(f"[sparselog] load_model_bin() already model exists. path: {self.model_path}")

        else:
            # 패턴이상탐지(log2template) 학습 시 저장됨, 이미 존재할 예정
            self.logger.warning(f"[sparselog] Model file not found. path: {self.model_path}")

    def save_model_bin(self):
        try:
            joblib.dump(self.template_miner, self.model_path)
            self.logger.info(f"[sparseLog] template_miner trained for {self.lasted_mins} minutes has been saved.")
        except Exception as e:
            self.logger.exception(f"unable to save template_miner model. path:{self.model_path}. {e}")


    def post_analysis(self, log_data):
        if log_data["success"] is False:
            return

        raw_log = log_data["msg"].strip()
        is_userkeyword = True if log_data["keyword"] is not None else False

        # Drain 분석 및 트리 업데이트 진행
        is_sparse, rare_rate, matched_template = self.process_drain(log_data)

        if self.alert and is_sparse:
            self.cache_sparse_cnt += 1

        # detection type info for inserting to PG
        detection_type = 0  # 검출된 희소로그 또는 키워드 없음
        if is_userkeyword is True and is_sparse is True:
            detection_type = 1  # type 1 : 희소로그 & 키워드 동시 검출
        elif is_sparse is True:
            detection_type = 2  # type 2 : 희소로그만 검출
        elif is_userkeyword is True:
            detection_type = 3  # type 3 : 키워드만 검출
        res = []
        if detection_type != 0:
            if log_data["exclude_keyword"] is None or (log_data["exclude_keyword"] is not None and log_data["exclude_keyword"] not in raw_log):
                self.logger.debug(f"log_data['line_no']: {log_data['line_no']}, rare_rate: {rare_rate}, raw_log: {raw_log}")
                log_time = log_data["time"]
                if log_time is None:
                    log_time = datetime.datetime.now()  # 로그 시간을 명시 하지 않으면, 현재 시간 입력
                res = [
                    pd.to_datetime(log_time).strftime("%Y-%m-%d %H:%M:%S"),
                    self.target_config["log_path"],
                    raw_log,
                    log_data["line_no"],
                    rare_rate*100,
                    self.target_config["log_id"],
                    log_data["keyword"],
                    detection_type
                ]
            else:
                self.logger.info(f"exclude_keyword: {log_data['exclude_keyword']} in raw_log")

        return self.cache_sparse_cnt, is_sparse, rare_rate, matched_template, res

    async def predict(self, standard_datetime, input_df):
        self.logger.info(f"sparselog's serving starts.")
        start_sparse = time.time()
        self.cache_sparse_cnt = 0
        result = {"keys": ["time", "log_path", "content", "line_num", "probability", "log_id", "keyword", "detection_type"],
                  "values": []}
        if USE_MP and len(input_df) >= MP_MIN_ACTIVATE_LINES:
            cores = int(psutil.cpu_count(logical=False) * MP_CPU_PORTION)
            if cores == 0:
                cores = 1
            pool = multiprocessing.Pool(cores)

            # get preprocessed result
            prep_result = pool.map_async(self.preprocess_log, input_df)
            prep_result = prep_result.get()
            pool.close()
            pool.join()

            for log_data in prep_result:
                # check sparselog & update model tree & post process & insert result to pg
                sparse_cnt, is_sparse, rare_rate, matched_template, res = self.post_analysis(log_data)
                if res != []:
                    result["values"].append(res)

        # 적은 수량의 로그는 오버헤드를 줄이기 위해 일반 루프로 수행
        else:
            log_cnt = 0
            for log_data in input_df.iterrows():
                # get preprocessed result
                prep_result = self.preprocess_log(log_data)
                # check sparselog & update model tree & post process & insert result to pg
                sparse_cnt, is_sparse, rare_rate, matched_template, res = self.post_analysis(prep_result)
                log_cnt += 1
                if res != []:
                    result["values"].append(res)

        if self.config["is_master_server"] and self.lasted_mins % self.save_period == 0:
            self.save_model_bin()

        # model 및 count 관련 parameter redis 저장
        self.save_sparse_log_to_redis()

        # 대시보드 알림 체크
        header = None
        if self.alert and self.cache_sparse_cnt > 0:
            header = self.post_processing_result(self.cache_sparse_cnt, pd.to_datetime(standard_datetime))
        self.logger.info(f'[Asynchronous-sparse] time taken: {round(time.time() - start_sparse, 5)}')
        return result, header

    def post_processing_result(self, sparse_cnt, log_time):
        # make sparse serving header
        header = {}
        header["time"] = datetime.datetime.strftime(log_time, "%Y-%m-%d %H:%M")
        header["occurrences_cnt"] = sparse_cnt
        header["prop_cnt"] = self.alert_threshold
        return header

    def save_sparse_log_to_redis(self):
        try:
            # model_bin to redis
            REDISAI.save_body_to_redis(f"{self.model_dir_key}/template_miner", self.template_miner)
            self.logger.debug(f"save model_bin to redis Done")

        except Exception as e:
            self.logger.error(f"Unexpected exception '{e}' occured while saving models and parameters")