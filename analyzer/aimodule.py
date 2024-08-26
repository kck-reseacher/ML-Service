import json
import logging
import os
import time

import pandas as pd

from pathlib import Path
from common import aicommon, constants
from common.constants import SystemConstants as sc
from common.system_util import SystemUtil
from resources.logger_manager import Logger
from resources.config_manager import Config


class AIModule:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.name = __name__

    # API for training
    def init_train(self):
        pass

    def train(self, stat_logger):
        pass

    def test_train(self):
        pass

    def end_train(self):
        pass

    # API for serving
    def init_serve(self, reload):
        pass

    def serve(self, input_df):
        pass

    def end_serve(self):
        pass

    def estimate(self, serving_date: list = None, input_df: pd.DataFrame = None, sbiz_df: pd.DataFrame = None):
        pass

    # API for dev and debug
    def get_debug_info(self):
        pass

    @staticmethod
    def check_multi_logger(target_id, modelMap):
        return target_id not in modelMap or modelMap[target_id].get("logger", None) is None

    @staticmethod
    def create_multi_logger(logger_dir, logger_name, sys_id, module_name):
        os_env = SystemUtil.get_environment_variable()
        py_config = Config(os_env[sc.MLOPS_SERVING_PATH], os_env[sc.AIMODULE_SERVER_ENV]).get_config()

        # module_error_dir
        error_log_dict = dict()
        error_log_dict["log_dir"] = str(
            Path(os_env[sc.AIMODULE_LOG_PATH]) / "serving" / sc.ERROR_LOG_DEFAULT_PATH
        )
        error_log_dict["file_name"] = module_name

        logger = Logger().get_default_logger(
            logdir=logger_dir, service_name=logger_name, error_log_dict=error_log_dict,
        )

        if py_config["use_integration_log"]:
            module = module_name.rsplit("_", 1)[0]
            inst_type = module_name.rsplit("_", 1)[1]
            integration_log_dir = str(Path(os_env[sc.AIMODULE_LOG_PATH]) / "integration" / str(sys_id) / module / inst_type)
            integration_error_log_dict = {
                "log_dir": str(Path(os_env[sc.AIMODULE_LOG_PATH]) / "integration" / str(sys_id) / sc.ERROR_LOG_DEFAULT_PATH / sc.SERVING),
                "file_name": module_name,
            }
            Logger().get_default_logger(
                logdir=integration_log_dir, service_name=f"integration_{logger_name}",
                error_log_dict=integration_error_log_dict,
            )

        return logger

    @staticmethod
    def default_module_status():
        return {constants.PROGRESS: 0, constants.DURATION_TIME: None}
