import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter
from fastapi import Request, Response

from api.request_model.serving_model import *
from common import aicommon
from common.constants import SystemConstants as sc
from common.error_code import Errors
from common.module_exception import ModuleException
from common.redisai import REDISAI
from common.system_util import SystemUtil
from resources.config_manager import Config

router = APIRouter()
"""
    FastAPI router

    requestparam : backend 에서 요청오는 request 전문 
                    각 요청에 맞는 형태를 구성해서 validation 확인과 openapi에서 사용할수 있음
    request : fastapi_serving.py (app)으로 접근 가능함
            - app.instance : 각 analyzer의 인스턴스
            - app.logger : 각 모듈 별 logger  

"""
os_env = SystemUtil.get_environment_variable()
py_config = Config(os_env[sc.MLOPS_SERVING_PATH], os_env[sc.AIMODULE_SERVER_ENV]).get_config()
master_mlc = f"http://{py_config['redis_server']['master']['host']}:{py_config['module_api_server']['port']}"

temp_dict = dict()

sys_id = None
module_name = None
inst_type = None

instance = None
instance_was = None
instance_db = None
instance_os = None
instance_web = None
instance_tp = None
instance_network = None
instance_code = None
instance_service = None
instance_instanceGroup = None
instance_hostGroup = None
instance_codeGroup = None
instance_log = None

serverAPI = None
logger = None
server_env = None
target_port = None


#####################################################
###               XAIOPS SERVING API              ###
#####################################################


def service_serving(url, target_serv, body_dict, headers):
    target_serv['body'] = [body_dict]
    res = requests.post(url=url, data=json.dumps(target_serv), headers=headers)
    target_result = json.loads(res.text)
    return target_result['header'], body_dict['tx_code'], target_result['body'][body_dict['tx_code']]


async def _log_internal(requestparam: ServingRequestModel, request: Request, instance_type: str):
    start_time = time.time()
    instance = getattr(request.app, instance_type)
    logger = request.app.logger

    header = requestparam.header.dict()
    header['predict_time'] = requestparam.standard_datetime
    body = requestparam.body
    logger.info(
        f"[Serving {header['sys_id']}_{header['inst_type']}_{header['target_id']} start] input datetime: {header['predict_time']}")

    _header = None
    _body = None
    errno = None
    errmsg = None
    results = {}

    try:
        _header, _body, errno, errmsg = await instance.serve(header, body)

    except MemoryError as error:
        logger.exception(f"[Error] Unexpected memory error during serving : {error}")
        aicommon.Utils.print_memory_usage(logger)
        errno = f"{Errors.E707.value}"
        errmsg = f"{Errors.E707.desc}"
    except ModuleException as me:
        logger.exception(
            f"[Error] Unexpected ModuleException during serving : [{header['sys_id']}_{header['inst_type']}_{header['target_id']}] {me.error_msg}")
        errno = f"{me.error_code}"
        errmsg = f"{me.error_msg}"
    except Exception as exception:
        logger.exception(
            f"[Error] Unexpected exception during serving : [{header['sys_id']}_{header['inst_type']}_{header['target_id']}] {exception}")
        errno = f"{Errors.E777.value}"
        errmsg = f"{Errors.E777.desc}"
    finally:
        pass

    if errno is not None:
        results["errno"] = errno
    if errmsg is not None:
        results["errmsg"] = errmsg
    if _header is not None:
        results["header"] = _header
    if _body is not None:
        results["body"] = _body

    response = Response(json.dumps(results, cls=aicommon.JsonEncoder))
    response.headers['Content-type'] = 'application/json'
    elapsed_time = time.time() - start_time
    logger.info(
        f"[Serving {header['sys_id']}_{header['inst_type']}_{header['target_id']} finish] input datetime: {requestparam.standard_datetime}, elapsed time: {elapsed_time:.2f}, status: {response.status_code}")
    if elapsed_time > 45:
        logger.error(
            f"[{header['sys_id']}_{header['inst_type']}_{header['target_id']}] Serving Request 45s Timeout occurred!!")

    return response


def _serve_internal(requestparam: ServingRequestModel, request: Request, instance_type: str):
    start_time = time.time()
    instance = getattr(request.app, instance_type)
    logger = request.app.logger

    header = requestparam.header.dict()
    header['predict_time'] = requestparam.standard_datetime
    body = requestparam.body
    logger.info(f"[Serving {header['sys_id']}_{header['inst_type']}_{header['target_id']} start] input datetime: {header['predict_time']}")

    _header = None
    _body = None
    errno = None
    errmsg = None
    results = {}

    try:
        _header, _body, errno, errmsg = instance.serve(header, body)

    except MemoryError as error:
        logger.exception(f"[Error] Unexpected memory error during serving : {error}")
        aicommon.Utils.print_memory_usage(logger)
        errno = f"{Errors.E707.value}"
        errmsg = f"{Errors.E707.desc}"
    except ModuleException as me:
        logger.exception(
            f"[Error] Unexpected ModuleException during serving : [{header['sys_id']}_{header['inst_type']}_{header['target_id']}] {me.error_msg}")
        errno = f"{me.error_code}"
        errmsg = f"{me.error_msg}"
    except Exception as exception:
        logger.exception(
            f"[Error] Unexpected exception during serving : [{header['sys_id']}_{header['inst_type']}_{header['target_id']}] {exception}")
        errno = f"{Errors.E777.value}"
        errmsg = f"{Errors.E777.desc}"
    finally:
        pass

    if errno is not None:
        results["errno"] = errno
    if errmsg is not None:
        results["errmsg"] = errmsg
    if _header is not None:
        results["header"] = _header
    if _body is not None:
        results["body"] = _body

    response = Response(json.dumps(results, cls=aicommon.JsonEncoder))
    response.headers['Content-type'] = 'application/json'
    elapsed_time = time.time() - start_time
    logger.info(
        f"[Serving {header['sys_id']}_{header['inst_type']}_{header['target_id']} finish] input datetime: {requestparam.standard_datetime}, elapsed time: {elapsed_time:.2f}, status: {response.status_code}")
    if elapsed_time > 45:
        logger.error(
            f"[{header['sys_id']}_{header['inst_type']}_{header['target_id']}] Serving Request 45s Timeout occurred!!")

    return response


@router.post("/serving/anomaly-load/was", tags=['ITMA'], summary="이상탐지/부하예측 WAS type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_was")


@router.post("/serving/anomaly-load/db", tags=['ITMA'], summary="이상탐지/부하예측 DB type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_db")


@router.post("/serving/anomaly-load/os", tags=['ITMA'], summary="이상탐지/부하예측 OS type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_os")


@router.post("/serving/anomaly-load/tp", tags=['ITMA'], summary="이상탐지/부하예측 TP type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_tp")


@router.post("/serving/anomaly-load/web", tags=['ITMA'], summary="이상탐지/부하예측 WEB type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_web")


@router.post("/serving/anomaly-load/network", tags=['ITMA'], summary="이상탐지/부하예측 NETWORK type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_network")


@router.post("/serving/anomaly-load/code", tags=['ITMA'], summary="이상탐지/부하예측 CODE type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_code")


@router.post("/serving/anomaly/service", tags=['ITMA'], summary="only 이상탐지 SERVICE type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_service")

@router.post("/serving/anls-log/log", tags=['ITMA'], summary="로그 이상 탐지 log type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return await _log_internal(requestparam, request, "instance_log")


@router.post("/serving/load-fcst/was", tags=['ITMA'], summary="부하 예측(RMC) WAS type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_loadfcst_was")


@router.post("/serving/load-fcst/db", tags=['ITMA'], summary="부하 예측(RMC) DB type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_loadfcst_db")


@router.post("/serving/load-fcst/os", tags=['ITMA'], summary="부하 예측(RMC) OS type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_loadfcst_os")


@router.post("/serving/load-fcst/web", tags=['ITMA'], summary="부하 예측(RMC) WEB type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_loadfcst_web")


@router.post("/serving/load-fcst/tp", tags=['ITMA'], summary="부하 예측(RMC) TP type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_loadfcst_tp")


@router.post("/serving/load-fcst/network", tags=['ITMA'], summary="부하 예측(RMC) NETWORK type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_loadfcst_network")


@router.post("/serving/load-fcst/service", tags=['ITMA'], summary="부하 예측(RMC) SERVICE type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_loadfcst_service")


@router.post("/serving/event-fcst/was", tags=['ITMA'], summary="event-fcst WAS type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_event_fcst_was")


@router.post("/serving/event-fcst/db", tags=['ITMA'], summary="event-fcst DB type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_event_fcst_db")


@router.post("/serving/event-fcst/os", tags=['ITMA'], summary="event-fcst OS type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_event_fcst_os")


@router.post("/serving/event-fcst/web", tags=['ITMA'], summary="event-fcst WEB type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_event_fcst_web")


@router.post("/serving/event-fcst/tp", tags=['ITMA'], summary="event-fcst TP type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_event_fcst_tp")


@router.post("/serving/event-fcst/network", tags=['ITMA'], summary="event-fcst NETWORK type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_event_fcst_network")


@router.post("/serving/event-fcst/code", tags=['ITMA'], summary="event-fcst CODE type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_event_fcst_code")

@router.post("/serving/fcst-tsmixer/was", tags=['ITMA'], summary="부하 예측(tsmixer) WAS type")
async def serve(requestparam: ServingRequestModel, request: Request):
    print('--')
    return _serve_internal(requestparam, request, "instance_fcst_tsmixer_was")


@router.post("/serving/fcst-tsmixer/db", tags=['ITMA'], summary="부하 예측(tsmixer) DB type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_fcst_tsmixer_db")


@router.post("/serving/fcst-tsmixer/os", tags=['ITMA'], summary="부하 예측(tsmixer) OS type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_fcst_tsmixer_os")


@router.post("/serving/fcst-tsmixer/web", tags=['ITMA'], summary="부하 예측(tsmixer) WEB type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_fcst_tsmixer_web")


@router.post("/serving/fcst-tsmixer/tp", tags=['ITMA'], summary="부하 예측(tsmixer) TP type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_fcst_tsmixer_tp")


@router.post("/serving/fcst-tsmixer/network", tags=['ITMA'], summary="부하 예측(tsmixer) NETWORK type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_fcst_tsmixer_network")


@router.post("/serving/fcst-tsmixer/service", tags=['ITMA'], summary="부하 예측(tsmixer) SERVICE type")
async def serve(requestparam: ServingRequestModel, request: Request):
    return _serve_internal(requestparam, request, "instance_fcst_tsmixer_service")

##########################################################
#                     Multi Port API                     #
##########################################################
@router.post("/question", tags=['CHATBOT'], response_model=None, summary="챗봇 질문에 대한 답변 수행")
async def serve(request: Request):
    start_time = time.time()
    body_req = await request.json()
    instance = request.app.instance
    logger = request.app.logger

    logger.info(f"[Serving session(key={body_req.get('sessionKey', None)}) start] input datetime: {body_req['lastServingTime']}")

    body_res = None

    errno = None
    errmsg = None

    try:
        body_res = instance.serve(body_req)

    except MemoryError as error:
        logger.exception(f"[Error] Unexpected memory error during serving : {error}")
        aicommon.Utils.print_memory_usage(logger)
        errno = f"{Errors.E707.value}"
        errmsg = f"{Errors.E707.desc}"
    except ModuleException as me:
        logger.exception(f"[Error] Unexpected ModuleException during serving : [{body_req.get('sessionKey', None)}] {me.error_msg}")
        errno = f"{me.error_code}"
        errmsg = f"{me.error_msg}"
    except Exception as exception:
        logger.exception(f"[Error] Unexpected exception during serving : [{body_req.get('sessionKey', None)}] {exception}")
        errno = f"{Errors.E777.value}"
        errmsg = f"{Errors.E777.desc}"
    finally:
        pass

    logger.info(f"result = {body_res}")

    response = Response(json.dumps(body_res, cls=aicommon.JsonEncoder))
    response.headers['Content-type'] = 'application/json'

    elapsed_time = time.time() - start_time
    logger.info(f"[Serving session(key={body_req.get('sessionKey', None)}) finish] input datetime: {body_req['lastServingTime']}, elapsed time: {elapsed_time:.2f}, status: {response.status_code}")

    return response

##########################################################
#                   Internal Function                    #
##########################################################

def _request_validation(data, target_id, logger):
    horizon = "1min"
    logger.debug("============== Start request validation ==============")
    try:
        df = pd.DataFrame(data)

        # 서빙 데이터 time column
        if 'time' not in df.columns:
            logger.warning("dataframe 'time' column is not exist")
            return

        df.index = pd.to_datetime(df.time)

        # 서빙 데이터 길이
        if len(df) != 60 and len(df) != 1:
            logger.warning("'serving data' len is not 60 ")
            return

        # 서빙 데이터 중복 또는 누락
        new_df = df.resample(horizon).count()
        if not np.array_equal(np.ones(new_df.shape), new_df):
            logger.warning("'serving data' duplicated or missed (NaN)")
            return

        global temp_dict
        if target_id not in temp_dict.keys():
            temp_dict[target_id] = pd.DataFrame()

        if df.index[-1] == temp_dict[target_id].index:
            logger.warning(f"'serving data' duplicated at the same time {df.index[-1]}")
            return

        temp_dict[target_id] = df.iloc[[-1]]  # df 의 마지막 row 를 저장
        logger.debug(f"============== last serving data time : {temp_dict[target_id].index[0]} ==============")
    except Exception:
        pass

