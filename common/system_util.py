import os
import sys
import argparse

from common.constants import SystemConstants as sc
from common import aicommon


class SystemUtil:
    @staticmethod
    def get_run_function_name() -> str:
        name = ""
        try:
            name = sys._getframe().f_code.co_name
        except SystemError:
            return name
        return name

    @staticmethod
    def get_class_name(cls) -> str:
        return type(cls).__name__

    @staticmethod
    def get_environment_variable():
        os_env = dict()
        # AIMODULE_HOME
        home = os.environ.get(sc.AIMODULE_HOME)
        if home is None:
            print("plz export AIMODULE_HOME")
            home = os.path.dirname(os.path.abspath(__file__))
        else:
            os_env[sc.AIMODULE_HOME] = home

        # AIMODULE_LOG_PATH
        log_path = os.environ.get(sc.AIMODULE_LOG_PATH)
        if log_path is None:
            print("plz export AIMODULE_LOG_PATH")
            log_path = os.path.dirname(os.path.abspath(__file__))
        else:
            os_env[sc.AIMODULE_LOG_PATH] = log_path

        # AIMODULE_PATH
        py_path = os.environ.get(sc.AIMODULE_PATH)
        if py_path is None:
            print("plz export AIMODULE_PATH")
            py_path = os.path.dirname(os.path.abspath(__file__))
        else:
            os_env[sc.AIMODULE_PATH] = py_path

        # MLOPS_SERVING_PATH
        mlops_serving_path = os.environ.get(sc.MLOPS_SERVING_PATH)
        if mlops_serving_path is None:
            print("plz export MLOPS_SERVING_PATH")
        else:
            os_env[sc.MLOPS_SERVING_PATH] = mlops_serving_path.lower()

        # AIMODULE_SERVER_ENV
        server_env = os.environ.get(sc.AIMODULE_SERVER_ENV)
        if server_env is None:
            print("plz export AIMODULE_SERVER_ENV")
            py_path = os.path.dirname(os.path.abspath(__file__))
        else:
            os_env[sc.AIMODULE_SERVER_ENV] = server_env

        # MLOPS_SERVER_ENV
        mlops_server_env = os.environ.get(sc.MLOPS_SERVER_ENV)
        if mlops_server_env is None:
            print("plz export MLOPS_SERVER_ENV")
        else:
            os_env[sc.MLOPS_SERVER_ENV] = mlops_server_env.lower()

        # USE_SLAVE_SERVER
        use_slave_server = os.environ.get(sc.USE_SLAVE_SERVER)
        if use_slave_server is None:
            print("plz export USE_SLAVE_SERVER")
        else:
            os_env[sc.USE_SLAVE_SERVER] = use_slave_server.lower() == "true" if use_slave_server else False

        return os_env

    @staticmethod
    def get_server_start_param(module_name):
        # 입력 인자 설정
        if len(sys.argv) < 2:
            aicommon.Utils.usage()
            sys.exit()

        target_flag = False if "multi" in module_name else True

        parser = argparse.ArgumentParser(prog=module_name, description=module_name, add_help=True)
        parser.add_argument("-m", "--module", help="module name.", required=True)
        parser.add_argument("-p", "--port", help="port number", required=True)

        args = parser.parse_args()
        module_name = args.module
        target_port = args.port

        return module_name, target_port
