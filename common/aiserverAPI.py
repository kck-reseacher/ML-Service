import json
import sys

import requests

from common import aicommon, constants


class ServerAPI:
    def __init__(self, config, logger):
        """
        서버와 api 통신을 하기 위한 클래스

        설정 및 데이터 이동 방식
            - 대시보드 UI -> 백엔드 서버 -> 모델링 모듈
        사용자 화면에서 관련 설정을 변경 할 경우 모델링 모듈에서 변경된 설정을 가져옴.

        Parameters
        ----------
        config : 설정
        logger : 로그
        """

        self.config = config
        self.logger = logger

        self.dockerServerURL = f"http://{self.config['mlc_docker']['host']}:{self.config['mlc_docker']['port']}"

        # 배포 후 학습전 기존 model_config에 host로 저장된 정보 사용을 위해 분기 처리
        if self.config['api_server'] is None:
            self.serverURL = f"http://{self.config['host']}:{self.config['ports']['dashboard']}"
            self.adhocURL = f"http://{self.config['host']}:{self.config['ports']['adhoc']}"

        # adhoc 은 따로 구분없이 사용해도 된다고 확인
        else:
            self.serverURL = f"http://{self.config['api_server']['host']}:{self.config['api_server']['port']}"
            self.adhocURL = f"http://{self.config['api_server']['host']}:{self.config['api_server']['port']}"

    def _request_post(self, url, data, headers=None):
        response = None
        try:
            if headers:
                response = requests.post(url=url, data=data, headers=headers, timeout=10)
            else:
                response = requests.post(url=url, data=data, timeout=10)
        except requests.exceptions.Timeout:
            self.logger.error(f"Request Time Out : {url}")
        except Exception as e:
            self.logger.error(f"{e}")

        return response

    def send_anomaly_alarm(self, data):
        json_data = json.dumps(data, cls=aicommon.JsonEncoder)

        url = f"{self.serverURL}/send-alarm-notification"
        response = self._request_post(
            url=url, data=json_data, headers={"Content-type": "application/json"}
        )

        self._logging_result("send_anomaly_alarm", json_data, response)

    def send_anomalies_alarm(self, data):
        json_data = json.dumps(data, cls=aicommon.JsonEncoder)

        url = f"{self.serverURL}/v2/send-alarm-notification"
        response = self._request_post(url=url, data=json_data, headers={"Content-type": "application/json"})

        self._logging_result("send_anomalies_alarm", json_data, response)

    def send_predict_alarm(self, data):
        json_data = json.dumps(data, cls=aicommon.JsonEncoder)
        url = f"{self.serverURL}/send-alarm-notification"
        response = self._request_post(
            url=url, data=json_data, headers={"Content-type": "application/json"}
        )

        self._logging_result("send_predict_alarm", json_data, response)

    def get_adhoc_data(self, data):
        json_data = json.dumps(data, cls=aicommon.JsonEncoder)

        url = f"{self.adhocURL}/adhoc"
        response = self._request_post(url, data=json_data)

        self.logger.info(f"get_adhoc_data param:{json_data}")
        self.logger.debug(f"get_adhoc_data response:{response}")
        self._logging_result("get_adhoc_data", json_data, response)

        return response

    def update_redis_parameter(self, sys_id, module_name: str, inst_type, target_id):
        if module_name in str(constants.REQUIRE_SERVICE_API_MODULE_LIST):
            url = (f"{self.dockerServerURL}/mlc/rsa/update/parameter/{module_name}/{inst_type}/{target_id}")
            try:
                response = requests.patch(url)
            except Exception as e:
                self.logger.exception(f"{e}")
                self.logger.error(f"API server invalid.. {module_name} module service api required.. server down..")
                sys.exit(1)
            if response.status_code != 200:
                self._logging_error_result("update_service_parameter()", url, response.text)
                self.logger.error(f"Service API response format invalid.. inquire backend manager.. server down..")
                sys.exit(1)
            else:
                return True
        else:
            self.logger.error(f"{module_name} module service api not provided")
            return False

    def get_fail_condition(self, sys_id, module_name, target_id):
        url = (
            f"{self.serverURL}/mdl/module-failure-detection-info/"
            f"{str(sys_id)}/{module_name}/{str(target_id)}"
        )

        response = requests.get(url)

        self._logging_result("get_fail_condition", url, response.json()["data"])

        return json.loads(response.json()["data"])

    def get_failure_detection(self, sys_id):
        url = f"{self.serverURL}/failure-detection/service/{str(sys_id)}/list"

        response = requests.get(url)

        self._logging_result("get_failure_detection", url, response.json()["data"])

        return response.json()["data"]

    def _logging_result(self, func, json_data, response):
        self.logger.debug(func + f" request data : {json_data}")
        self.logger.info(func + f" response data: {response}")

    def _logging_error_result(self, func, json_data, response):
        self.logger.debug(func + f" request data : {json_data}")
        self.logger.error(func + f" response data: {response}")
