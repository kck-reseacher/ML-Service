import datetime
import json
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware

from analyzer.analyzer_factory import AnalyzerFactory
from api.configuration.configurations import MultiServingAPIConfigurations
from api.router import routers
from common.aiserverAPI import ServerAPI
from common.base64_util import Base64Util
from common.constants import SystemConstants as sc
from common.constants import MLServingConstants as mc
from common.system_util import SystemUtil
from resources.config_manager import Config
from resources.logger_manager import Logger


def create_app(*args, **kwargs):
    print(f"gunicorn server entry point start")

    args_dict = {}
    for k in kwargs:
        if k == 's':
            args_dict["sys_id"] = str(kwargs[k])
        elif k == 'm':
            args_dict["module_name"] = str(kwargs[k])
        elif k == 'i':
            args_dict["inst_type"] = str(kwargs[k])
        elif k == 't':
            args_dict["target_id"] = str(kwargs[k])
        elif k == 'p':
            args_dict["master_port"] = str(kwargs[k])
        elif k == 'w':
            args_dict["worker"] = str(kwargs[k])
        else:
            print(f"{k} args key is invalid")

    return main_process(args_dict)

def get_database_connection(py_config):
    try:
        pg_decode_config = Base64Util.get_config_decode_value(py_config[sc.POSTGRES])
    except Exception as e:
        print('base64 decode error, config: ' + str(py_config[sc.POSTGRES]))
        pg_decode_config = py_config[sc.POSTGRES]

    db_conn_str = (
        f"host={pg_decode_config['host']} "
        f"port={pg_decode_config['port']} "
        f"dbname={pg_decode_config['database']} "
        f"user={pg_decode_config['id']} "
        f"password={pg_decode_config['password']}"
    )
    return db_conn_str

def get_logpresso_connection(py_config):
    try:
        logpresso_decode_config = py_config["logpresso"]
        logpresso_decode_config = Base64Util.get_config_decode_value(logpresso_decode_config)
    except Exception as e:
        print('base64 decode error, config: ' + str(py_config["logpresso"]))

    logpresso = {
        "host": logpresso_decode_config['host'],
        "port": logpresso_decode_config['port'],
        "id": logpresso_decode_config['id'],
        "password": logpresso_decode_config['password']}
    return logpresso


def main_process(args_dict=None):
    # get environment_variable (AIMODULE_HOME, AIMODULE_PATH)
    os_env = SystemUtil.get_environment_variable()

    # get config (pg, api_server, flask_server)
    py_config = Config(os_env[sc.MLOPS_SERVING_PATH], os_env[sc.AIMODULE_SERVER_ENV]).get_config()

    # get server start parameter
    if args_dict is None:
        module_name, master_port = SystemUtil.get_server_start_param(
            os.path.basename(__file__))
        worker = 1
    else:
        module_name, master_port, worker = args_dict["module_name"], args_dict["master_port"], args_dict["worker"]

    sys_id = "all"
    target_id = "all"
    inst_type = "all"

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # not use GPU

    param = {
        "module": module_name,
        "sys_id": py_config["sys_id"],
        "inst_type": inst_type,
        "temp_dir": f"{os_env[sc.AIMODULE_HOME]}/temp",
        "model_dir": str(Path(os_env[sc.AIMODULE_HOME]) / "model" / "system" / module_name),
        "log_dir": str(Path(os_env[sc.AIMODULE_LOG_PATH]) / "serving" / module_name),
        "integration_log_dir": str(Path(os_env[sc.AIMODULE_LOG_PATH]) / "integration" / "system" / module_name),
        "db_conn_str": get_database_connection(py_config),
        "api_server": py_config["api_server"],
        "mlc_docker": py_config["mlc_docker"],
        "home_dir": f"{os_env[sc.AIMODULE_HOME]}"
    }

    module_params = {
        "exem_aiops_anls_inst_multi": {
            "trained_module": "exem_aiops_anls_inst",
            "type_list": ["was", "db", "os", "web", "tp", "network", "code",  "service"],
            "target_info": {"was": ["super"],
                            "os": ["super"],
                            "db": ["super"],
                            "tp": ["super"],
                            "web": ["super"],
                            "network": ["super"],
                            "code": ["super"]}
        },

        "exem_aiops_anls_service_multi": {
            "trained_module": "exem_aiops_anls_service",
            "type_list": ["service"],
            "target_info": {
                "service": ["super"]}

        },
        "exem_aiops_anls_log_multi": {
            "trained_module": "exem_aiops_anls_log",
            "type_list": ["log"],
            "target_info": {
                "log": ["super"]}
        },
        "exem_aiops_load_fcst_multi": {
            "trained_module": "exem_aiops_load_fcst",
            "type_list": ["was", "db", "os", "web", "tp", "network", "code",  "service"],
            "target_info": {"was": ["super"],
                            "os": ["super"],
                            "db": ["super"],
                            "tp": ["super"],
                            "web": ["super"],
                            "network": ["super"],
                            "service" : ["super"]
                            }
        },
        "exem_aiops_event_fcst_multi": {
            "trained_module": "exem_aiops_event_fcst",
            "type_list": ["was", "db", "os", "web", "tp", "network", "code"],
            "target_info": {"was": ["super"],
                            "os": ["super"],
                            "db": ["super"],
                            "tp": ["super"],
                            "web": ["super"],
                            "network": ["super"],
                            "code": ["super"]}
        },
        "exem_aiops_fcst_tsmixer_multi": {
            "trained_module": "exem_aiops_fcst_tsmixer",
            "type_list": ["was", "db", "os", "web", "tp", "network", "code", "service"],
            "target_info": {"was": ["super"],
                            "os": ["super"],
                            "db": ["super"],
                            "tp": ["super"],
                            "web": ["super"],
                            "network": ["super"],
                            "service" : ["super"]
                            }
        }
    }
    module_list = module_params.keys()
    # logger
    error_log_dict = {
        "log_dir": str(Path(os_env[sc.AIMODULE_LOG_PATH]) / "serving" / sc.ERROR_LOG_DEFAULT_PATH),
        "file_name": module_name,
    }

    logger = Logger().get_default_logger(logdir=param["log_dir"], service_name=sc.SERVING,
                                         error_log_dict=error_log_dict)

    if py_config["use_integration_log"]:
        integration_error_log_dict = {
            "log_dir": str(Path(os_env[sc.AIMODULE_LOG_PATH]) / "integration" / "system" / sc.ERROR_LOG_DEFAULT_PATH / sc.SERVING),
            "file_name": module_name,
        }
        Logger().get_default_logger(logdir=param["integration_log_dir"], service_name="integration_serving",
                                    error_log_dict=integration_error_log_dict)

    # module train/serving parameters
    serverAPI = ServerAPI(param, logger)

    # get analyzer instance
    anal_module_path = str(Path(os_env[sc.MLOPS_SERVING_PATH]) / "analyzer")
    anal_factory = AnalyzerFactory(logger)

    for module in module_list:
        if module == "exem_aiops_anls_inst_multi":
            trained_module = module_params[module]['trained_module']
            target_info = module_params[module]["target_info"]
            type_list = module_params[module]['type_list']

            params = {t: param.copy() for t in type_list}
            for t in type_list:
                params[t]["model_dir"] = str(Path(os_env[sc.AIMODULE_HOME]) / "model")
                params[t]["log_dir"] = str(Path(os_env[sc.AIMODULE_LOG_PATH]) / "serving")
                params[t]["module"] = trained_module
                params[t]["inst_type"] = t
                params[t]["temp_dir"] = str(Path(os_env[sc.AIMODULE_HOME]) / "temp")
                params[t]["anls_inst_template_path"] = os_env[sc.MLOPS_SERVING_PATH] + "/resources/anls_inst_template.json"

            instance_was = anal_factory.get_analyzer(anal_module_path, params["was"], module)
            instance_was.name = module
            instance_was.init_load(target_info, master_port, os_env[sc.AIMODULE_SERVER_ENV])

            instance_db = anal_factory.get_analyzer(anal_module_path, params["db"], module)
            instance_db.name = module
            instance_db.init_load(target_info, master_port, os_env[sc.AIMODULE_SERVER_ENV])

            instance_os = anal_factory.get_analyzer(anal_module_path, params["os"], module)
            instance_os.name = module
            instance_os.init_load(target_info, master_port, os_env[sc.AIMODULE_SERVER_ENV])

            instance_web = anal_factory.get_analyzer(anal_module_path, params["web"], module)
            instance_web.name = module
            instance_web.init_load(target_info, master_port, os_env[sc.AIMODULE_SERVER_ENV])

            instance_tp = anal_factory.get_analyzer(anal_module_path, params["tp"], module)
            instance_tp.name = module
            instance_tp.init_load(target_info, master_port, os_env[sc.AIMODULE_SERVER_ENV])

            instance_network = anal_factory.get_analyzer(anal_module_path, params["network"], module)
            instance_network.name = module
            instance_network.init_load(target_info, master_port, os_env[sc.AIMODULE_SERVER_ENV])

            instance_code = anal_factory.get_analyzer(anal_module_path, params["code"], module)
            instance_code.name = module
            instance_code.init_load(target_info, master_port, os_env[sc.AIMODULE_SERVER_ENV])

        elif module == "exem_aiops_anls_log_multi":
            trained_module = module_params[module]['trained_module']
            type_list = module_params[module]['type_list']

            params = {t: param.copy() for t in type_list}
            for t in type_list:
                params[t]["module"] = trained_module
                params[t]["inst_type"] = t
                params[t]["model_dir"] = str(Path(os_env[sc.AIMODULE_HOME]) / "model")
                params[t]["log_dir"] = str(Path(os_env[sc.AIMODULE_LOG_PATH]) / "serving" / module)
                params[t]["logpresso"] = get_logpresso_connection(py_config)
                params[t]["is_master_server"] = os_env[sc.MLOPS_SERVER_ENV] == sc.MASTER
            instance_log = anal_factory.get_analyzer(anal_module_path, params['log'], module)
            instance_log.name = module

        elif module == "exem_aiops_load_fcst_multi":
            trained_module = module_params[module]['trained_module']
            target_info = module_params[module]["target_info"]
            type_list = module_params[module]['type_list']

            params = {t: param.copy() for t in type_list}
            for t in type_list:
                params[t]["model_dir"] = str(Path(os_env[sc.AIMODULE_HOME]) / "model")
                params[t]["log_dir"] = str(Path(os_env[sc.AIMODULE_LOG_PATH]) / "serving")
                params[t]["module"] = trained_module
                params[t]["inst_type"] = t

            instance_loadfcst_was = anal_factory.get_analyzer(anal_module_path, params["was"], module)
            instance_loadfcst_was.name = module
            instance_loadfcst_was.init_load(target_info, master_port, os_env[sc.AIMODULE_SERVER_ENV])

            instance_loadfcst_db = anal_factory.get_analyzer(anal_module_path, params["db"], module)
            instance_loadfcst_db.name = module
            instance_loadfcst_db.init_load(target_info, master_port, os_env[sc.AIMODULE_SERVER_ENV])

            instance_loadfcst_os = anal_factory.get_analyzer(anal_module_path, params["os"], module)
            instance_loadfcst_os.name = module
            instance_loadfcst_os.init_load(target_info, master_port, os_env[sc.AIMODULE_SERVER_ENV])

            instance_loadfcst_web = anal_factory.get_analyzer(anal_module_path, params["web"], module)
            instance_loadfcst_web.name = module
            instance_loadfcst_web.init_load(target_info, master_port, os_env[sc.AIMODULE_SERVER_ENV])

            instance_loadfcst_tp = anal_factory.get_analyzer(anal_module_path, params["tp"], module)
            instance_loadfcst_tp.name = module
            instance_loadfcst_tp.init_load(target_info, master_port, os_env[sc.AIMODULE_SERVER_ENV])

            instance_loadfcst_network = anal_factory.get_analyzer(anal_module_path, params["network"], module)
            instance_loadfcst_network.name = module
            instance_loadfcst_network.init_load(target_info, master_port, os_env[sc.AIMODULE_SERVER_ENV])

            instance_loadfcst_service = anal_factory.get_analyzer(anal_module_path, params["service"], module)
            instance_loadfcst_service.name = module
            instance_loadfcst_service.init_load(target_info, master_port, os_env[sc.AIMODULE_SERVER_ENV])

        elif module == "exem_aiops_event_fcst_multi":
            trained_module = module_params[module]['trained_module']
            target_info = module_params[module]["target_info"]
            type_list = module_params[module]['type_list']

            params = {t: param.copy() for t in type_list}
            for t in type_list:
                params[t]["model_dir"] = str(Path(os_env[sc.AIMODULE_HOME]) / "model")
                params[t]["log_dir"] = str(Path(os_env[sc.AIMODULE_LOG_PATH]) / "serving")
                params[t]["module"] = trained_module
                params[t]["inst_type"] = t
                params[t]["temp_dir"] = str(Path(os_env[sc.AIMODULE_HOME]) / "temp")
                params[t]["anls_inst_template_path"] = os_env[sc.AIMODULE_PATH] + "/resources/anls_inst_template.json"

            instance_event_fcst_was = anal_factory.get_analyzer(anal_module_path, params["was"], module)
            instance_event_fcst_was.name = module
            instance_event_fcst_was.init_load()

            instance_event_fcst_db = anal_factory.get_analyzer(anal_module_path, params["db"], module)
            instance_event_fcst_db.name = module
            instance_event_fcst_db.init_load()

            instance_event_fcst_os = anal_factory.get_analyzer(anal_module_path, params["os"], module)
            instance_event_fcst_os.name = module
            instance_event_fcst_os.init_load()

            instance_event_fcst_web = anal_factory.get_analyzer(anal_module_path, params["web"], module)
            instance_event_fcst_web.name = module
            instance_event_fcst_web.init_load()

            instance_event_fcst_tp = anal_factory.get_analyzer(anal_module_path, params["tp"], module)
            instance_event_fcst_tp.name = module
            instance_event_fcst_tp.init_load()

            instance_event_fcst_network = anal_factory.get_analyzer(anal_module_path, params["network"], module)
            instance_event_fcst_network.name = module
            instance_event_fcst_network.init_load()

            instance_event_fcst_code = anal_factory.get_analyzer(anal_module_path, params["code"], module)
            instance_event_fcst_code.name = module
            instance_event_fcst_code.init_load()

        elif module == "exem_aiops_fcst_tsmixer_multi":
            trained_module = module_params[module]['trained_module']
            target_info = module_params[module]["target_info"]
            type_list = module_params[module]['type_list']

            params = {t: param.copy() for t in type_list}
            for t in type_list:
                params[t]["model_dir"] = str(Path(os_env[sc.AIMODULE_HOME]) / "model")
                params[t]["log_dir"] = str(Path(os_env[sc.AIMODULE_LOG_PATH]) / "serving")
                params[t]["module"] = trained_module
                params[t]["inst_type"] = t
                params[t]["temp_dir"] = str(Path(os_env[sc.AIMODULE_HOME]) / "temp")
                params[t]["anls_inst_template_path"] = os_env[sc.AIMODULE_PATH] + "/resources/anls_inst_template.json"

            instance_fcst_tsmixer_was = anal_factory.get_analyzer(anal_module_path, params["was"], module)
            instance_fcst_tsmixer_was.name = module
            instance_fcst_tsmixer_was.init_load(target_info, master_port, os_env[sc.AIMODULE_SERVER_ENV])

            instance_fcst_tsmixer_db = anal_factory.get_analyzer(anal_module_path, params["db"], module)
            instance_fcst_tsmixer_db.name = module
            instance_fcst_tsmixer_db.init_load(target_info, master_port, os_env[sc.AIMODULE_SERVER_ENV])

            instance_fcst_tsmixer_os = anal_factory.get_analyzer(anal_module_path, params["os"], module)
            instance_fcst_tsmixer_os.name = module
            instance_fcst_tsmixer_os.init_load(target_info, master_port, os_env[sc.AIMODULE_SERVER_ENV])

            instance_fcst_tsmixer_web = anal_factory.get_analyzer(anal_module_path, params["web"], module)
            instance_fcst_tsmixer_web.name = module
            instance_fcst_tsmixer_web.init_load(target_info, master_port, os_env[sc.AIMODULE_SERVER_ENV])

            instance_fcst_tsmixer_tp = anal_factory.get_analyzer(anal_module_path, params["tp"], module)
            instance_fcst_tsmixer_tp.name = module
            instance_fcst_tsmixer_tp.init_load(target_info, master_port, os_env[sc.AIMODULE_SERVER_ENV])

            instance_fcst_tsmixer_network = anal_factory.get_analyzer(anal_module_path, params["network"], module)
            instance_fcst_tsmixer_network.name = module
            instance_fcst_tsmixer_network.init_load(target_info, master_port, os_env[sc.AIMODULE_SERVER_ENV])

            instance_fcst_tsmixer_service = anal_factory.get_analyzer(anal_module_path, params["service"], module)
            instance_fcst_tsmixer_service.name = module
            instance_fcst_tsmixer_service.init_load(target_info, master_port, os_env[sc.AIMODULE_SERVER_ENV])
        else:
            logger.error("no module name")

    # post model load
    '''
    multi 모듈 초기 기동 시 port 별로 target_list 정보를 받아 로딩
    현재는 inst_multi 모듈만 적용, 추후 다른 multi 모듈도 필요하다면 적용 예정
    '''

    logger.info("============== Start Serving Process ==============")

    ############### 라우터 구성 방식 api/router/routers.py 공통 모듈 사용 ###############
    # set fastapi app

    app = FastAPI(
        title=MultiServingAPIConfigurations.title,
        description=MultiServingAPIConfigurations.description,
        version=MultiServingAPIConfigurations.version,
    )

    """
    router에 depends로 들어가는 함수 (DI method)
    request 전문을 디버깅하기 위한 용도
    serving의 경우 request 전문이 길어질수 있어서 길이 1000개 까지 로깅, 오류 시 무시
    """

    async def log_request_info(request: Request):
        try:
            if request.method == "POST":
                request_body = await request.json()
                request_body_str = str(request_body) if len(str(request_body)) < 1000 else str(request_body)[:1000]
                logger.debug(f"REQUEST JSON : {request_body_str}")
        except Exception:
            pass

    ## set app's router
    """
    모델 파일 reload를 위한 설정값
    reload를 signal 방식으로 하기 위해서 router 변수에 미리 필요한 값을 설정해줌
    """
    router = routers
    router.module_name = module_name

    router.instance_was = instance_was
    router.instance_db = instance_db
    router.instance_os = instance_os
    router.instance_web = instance_web
    router.instance_tp = instance_tp
    router.instance_network = instance_network
    router.instance_code = instance_code
    router.instance_log = instance_log
    router.instance_loadfcst_was = instance_loadfcst_was
    router.instance_loadfcst_db = instance_loadfcst_db
    router.instance_loadfcst_os = instance_loadfcst_os
    router.instance_loadfcst_web = instance_loadfcst_web
    router.instance_loadfcst_tp = instance_loadfcst_tp
    router.instance_loadfcst_network = instance_loadfcst_network
    router.instance_loadfcst_service = instance_loadfcst_service

    router.instance_event_fcst_was = instance_event_fcst_was
    router.instance_event_fcst_db = instance_event_fcst_db
    router.instance_event_fcst_os = instance_event_fcst_os
    router.instance_event_fcst_web = instance_event_fcst_web
    router.instance_event_fcst_tp = instance_event_fcst_tp
    router.instance_event_fcst_network = instance_event_fcst_network
    router.instance_event_fcst_code = instance_event_fcst_code

    router.instance_fcst_tsmixer_was = instance_fcst_tsmixer_was
    router.instance_fcst_tsmixer_db = instance_fcst_tsmixer_db
    router.instance_fcst_tsmixer_os = instance_fcst_tsmixer_os
    router.instance_fcst_tsmixer_web = instance_fcst_tsmixer_web
    router.instance_fcst_tsmixer_tp = instance_fcst_tsmixer_tp
    router.instance_fcst_tsmixer_network = instance_fcst_tsmixer_network
    router.instance_fcst_tsmixer_service = instance_fcst_tsmixer_service

    router.serverAPI = serverAPI
    router.logger = logger
    router.server_env = os_env[sc.AIMODULE_SERVER_ENV]
    router.target_port = master_port
    router.inst_type = inst_type
    # for each request
    app.include_router(router.router, prefix="", dependencies=[Depends(log_request_info)])

    """
    API 함수에서 사용될 refer 전달
    router에서 request.app 에서 받아서 사용
    """

    app.instance_was = instance_was
    app.instance_db = instance_db
    app.instance_os = instance_os
    app.instance_web = instance_web
    app.instance_tp = instance_tp
    app.instance_network = instance_network
    app.instance_code = instance_code

    app.instance_log = instance_log
    app.instance_loadfcst_was = instance_loadfcst_was
    app.instance_loadfcst_db = instance_loadfcst_db
    app.instance_loadfcst_os = instance_loadfcst_os
    app.instance_loadfcst_web = instance_loadfcst_web
    app.instance_loadfcst_tp = instance_loadfcst_tp
    app.instance_loadfcst_network = instance_loadfcst_network
    app.instance_loadfcst_service = instance_loadfcst_service

    app.instance_event_fcst_was = instance_event_fcst_was
    app.instance_event_fcst_db = instance_event_fcst_db
    app.instance_event_fcst_os = instance_event_fcst_os
    app.instance_event_fcst_web = instance_event_fcst_web
    app.instance_event_fcst_tp = instance_event_fcst_tp
    app.instance_event_fcst_network = instance_event_fcst_network
    app.instance_event_fcst_code = instance_event_fcst_code

    app.instance_fcst_tsmixer_was = instance_fcst_tsmixer_was
    app.instance_fcst_tsmixer_db = instance_fcst_tsmixer_db
    app.instance_fcst_tsmixer_os = instance_fcst_tsmixer_os
    app.instance_fcst_tsmixer_web = instance_fcst_tsmixer_web
    app.instance_fcst_tsmixer_tp = instance_fcst_tsmixer_tp
    app.instance_fcst_tsmixer_network = instance_fcst_tsmixer_network
    app.instance_fcst_tsmixer_service = instance_fcst_tsmixer_service

    app.logger = logger
    app.inst_type = inst_type
    app.target_id = target_id
    app.module_name = module_name
    app.param = param
    app.serverAPI = serverAPI
    app.localhost = py_config["serving_flask"]["host"]
    app.master_host = py_config["redis_server"]["master"]["host"]
    app.target_port = master_port
    app.nginx_port = mc.PORTS["nginx"]
    app.worker = worker

    ########################################################################

    # set cors
    origins = ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_event():
        print(f"[{datetime.datetime.now()}] - MLOPS SERVING START....")

    @app.on_event("shutdown")
    async def shutdown_event():
        print(f"[{datetime.datetime.now()}] - MLOPS SERVING SHUTDOWN....")

    # gunicorn 실행될때
    if args_dict is not None:
        return app
    else:
        return logger, os_env[sc.MLOPS_SERVING_PATH], app, py_config, master_port


if __name__ == "__main__":
    logger, py_path, app, py_config, master_port = main_process()

    logger.info("============== Run FastAPI app ==============")

    try:
        uvicorn_access_log_dict = json.load(
            open(Path(py_path + sc.LOGGER_FILE_PATH) / sc.UVICORN_LOGGER_CONFIG_FILE)
        )
    except Exception:
        logger.error("uvicorn_access_log_dict ERROR")

    uvicorn.run(
        app,
        host=py_config["serving_flask"]["host"],
        port=int(master_port),
        access_log=True,
        reload=False,
        log_level="info",
        log_config=uvicorn_access_log_dict,
        workers=1,
    )
