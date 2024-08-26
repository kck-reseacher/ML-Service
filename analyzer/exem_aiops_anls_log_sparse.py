import datetime
import gc
import signal
import time
import json
from pathlib import Path
import jnius_config
jnius_config.add_options('-Xmx512m')

from analyzer import aimodule
from common.aiserverAPI import ServerAPI
import algorithms.logseq.config.default_regex as basereg
from algorithms.logseq.drain_SparseLogDetect import Drain
from common import constants
from common.constants import SystemConstants as sc
from common.module_exception import ModuleException
from common.redisai import REDISAI

global base_regL, base_delimiters
# default regular expressions
base_regL = basereg.COMMON_REGEX
base_delimiters = basereg.COMMON_DELIMITER


class ExemAiopsAnlsLogSparse(aimodule.AIModule):
    exit_now = False  # 종료 signal 탐지 시 메인프로세스 종료 플래그
    def __init__(self, config, logger):
        # SIGINT, SIGTERM signal handler 등록
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        self.running = False  # 현재 메인프로세스가 실행 중인 지 여부
        self.config = config
        self.alert = False  # 대시보드 실시간 알림 여부
        self.logger = logger
        self.serverAPI = ServerAPI(config, logger)

        self.save_period = self.config.get("save_period", 300)
        self.lasted_mins = 0
        self.today = datetime.date.today()

        self.inst_type = config["inst_type"]
        self.module_id = config["module"]
        self.sys_id = config["sys_id"]
        self.log_dir = config["log_dir"]
        self.modelMap_sparse = {'super':{}}

    def exit_gracefully(self, signum, frame):
        self.exit_now = True # (deprecated) for local file analyze
        if self.running:
            total_process_time = self.closing()
            self.logger.info(
                f"The running time of Drain is {total_process_time}.")
        self.logger.info("Terminate signal detected. Terminating Drain.")
        exit()

    def closing(self):
        self.running = False
        if self.config["save_model"]:
            self.drain.save_model_bin()

        total_process_time = time.time() - self.process_start_time
        self.logger.info(f"this process takes {total_process_time}")
        gc.collect()
        return total_process_time

    def create_logger(self, target_id):
        logger_dir = str(Path(self.log_dir) / self.inst_type / target_id)
        logger_name = f"{self.module_id}_{self.inst_type}_{self.sys_id}_{target_id}"
        logger = self.create_multi_logger(logger_dir, logger_name, self.sys_id, f"{sc.EXEM_AIOPS_ANLS_LOG_MULTI}_{self.inst_type}")

        return logger

    def load(self, target_id=None, reload=False):
        if reload is False and target_id in self.modelMap_sparse and constants.MODEL_S_SPARSELOG in self.modelMap_sparse["super"].keys():
            return True

        if reload is True:
            if target_id in self.modelMap_sparse:
                self.modelMap_sparse["super"]["logger"].info("[reload] start reload model")
            else:
                self.serving_logger.info(f"[reload] fail, target_id: {target_id} is not exist!!")

        is_logger = self.check_multi_logger(target_id, self.modelMap_sparse)

        if is_logger:
            self.modelMap_sparse["super"] = {}
            self.modelMap_sparse["super"]["logger"] = self.create_logger(target_id)
        else:
            if not self.modelMap_sparse["super"]["logger"].hasHandlers():
                self.modelMap_sparse["super"]["logger"] = self.create_logger(target_id)

        logger = self.modelMap_sparse["super"].get("logger")
        logger.info("[sparselog] start loading model")

        parameter_key = f"{self.sys_id}/{self.module_id}/{self.inst_type}/{target_id}/parameter"
        # REDIS에 키 없을 경우 --> parameter_kdy = 102/exem_aiops_anls_log/log/{target_id}/parameter
        if not REDISAI.exist_key(parameter_key):
            response = self.serverAPI.update_redis_parameter(self.sys_id, self.module_id, self.inst_type, target_id)
            if not response:
                return False
        parameter = json.loads(REDISAI.get(parameter_key))
        self.config["parameter"] = parameter[constants.MODEL_S_SPARSELOG]

        self.modelMap_sparse["super"]["use_sparselog"] = self.config["parameter"]["use"]

        try:
            if self.modelMap_sparse["super"]["use_sparselog"]:
                # 실제 분석을 수행할 Drain 인스턴스 정의
                self.modelMap_sparse["super"][constants.MODEL_S_SPARSELOG] = Drain(target_id, config=self.config, logger=logger)
                self.modelMap_sparse["super"][constants.MODEL_S_SPARSELOG].load_model_bin()
                self.modelMap_sparse["super"]["logger"].info(
                    f"====loaded SparseLog==target_id:{target_id}==(rareRate_threshold: {self.modelMap_sparse['super'][constants.MODEL_S_SPARSELOG].rareRate_threshold}, "
                    f"total {self.modelMap_sparse['super'][constants.MODEL_S_SPARSELOG].template_miner.drain.get_total_cluster_size()} lines of log message in the model.)")
            else:
                logger.info("[sparselog] analysis is not active.")
                return True, self.modelMap_sparse

        except Exception as exception:
            self.modelMap_sparse["super"]["logger"].exception(f"[Error] Unexpected exception during serving : {exception}")
            raise ModuleException("E777")

        return True, self.modelMap_sparse