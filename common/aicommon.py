import datetime
import importlib.util
import json
import os
from pathlib import Path
from time import sleep
import time

import numpy as np
import pandas as pd
import pandas.io.sql as psql
import psutil
import psycopg2 as pg2
import psycopg2.extras

from common import constants as bc


def sMAPE(y_true, y_pred, inverse=False):
    # symmetric Mean Absolute Percentage Error
    # y_true and y_pred must be inverse_scaled 1-D data
    def inverse_sMAPE(smape_val):
        return 100 - np.clip(smape_val, 0, 100)

    data = np.vstack([y_true.reshape(-1), y_pred.reshape(-1)]).T
    data[(data[:, 0] == 0) & (data[:, 1] == 0)] = 1e-7  # same as keras.epsilon
    smape = 100 * np.mean(np.abs(data[:, 0] - data[:, 1]) / (np.abs(data[:, 0]) + np.abs(data[:, 1])))

    return inverse_sMAPE(smape) if inverse else smape


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.__str__()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super(JsonEncoder, self).default(obj)


class Utils:
    @staticmethod
    def wait_file_exist(fpath, sec=60):
        if os.path.exists(fpath):
            return True

        for i in range(sec):
            if not os.path.exists(fpath):
                sleep(1)
            else:
                return True

        return False

    @staticmethod
    def make_future_dbsln_json(feature_name, dbsln_result):
        result = {}
        result["data"] = []
        result["total"] = 0
        data_df = dbsln_result.drop(dbsln_result.index[[0]])  # current time 제외
        data_df = data_df.reset_index()
        data_df = data_df.round(2)

        for i in data_df.index:
            result["data"].append(
                {
                    "time": data_df.at[i, "time"].strftime(bc.INPUT_DATETIME_FORMAT),
                    "upper": data_df.at[i, f"{feature_name}_upper"],
                    "lower": data_df.at[i, f"{feature_name}_lower"],
                }
            )

        return json.dumps(result)

    @staticmethod
    def get_module_from_file(module_name, file_path):
        spec = importlib.util.spec_from_file_location(
            module_name, str(Path(file_path) / f"{module_name}.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def usage():
        print("usage: command [OPTIONS] [MODULE]")
        print("    -m, --module   module name")
        print("    -t, --target   target name")
        print("    -p, --port     port number")
        print("    -s, --sys_id   system id")
        print("--inst_type        instance type [db, os, was] ")

    @staticmethod
    def get_class(module_name, class_name):
        m = __import__(module_name)
        m = getattr(m, class_name)
        return m

    @staticmethod
    def get_module_class(module_name, class_name, path):
        m = Utils.get_module_from_file(module_name, path)
        c = getattr(m, class_name)
        return c

    @staticmethod
    def to_camel_case(snake_str):
        components = snake_str.split("_")
        return "".join(x.title() for x in components[0:])

    @staticmethod
    def print_memory_usage(logger):
        """Prints current memory usage stats.
        See: https://stackoverflow.com/a/15495136

        :return: None
        """
        mega = 1024 * 1024
        svmem = psutil.virtual_memory()
        total, available, used, free = (
            svmem.total / mega,
            svmem.available / mega,
            svmem.used / mega,
            svmem.free / mega,
        )
        proc = psutil.Process(os.getpid()).memory_info()[1] / mega
        logger.warning(
            "process = %sMB total = %sMB available = %sMB used = %sMB free = %sMB percent = %s%%",
            proc,
            total,
            available,
            used,
            free,
            svmem.percent,
        )


    @staticmethod
    def min_max_scaler(data):
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)

        return numerator / (denominator + 1e-7)

    # calculate training mode
    @staticmethod
    def calc_training_mode(train_days):
        if train_days < 2:
            return bc.TRAINING_MODE_ERROR
        elif train_days < 7:
            return bc.TRAINING_MODE_DAILY
        elif train_days < 14:
            return bc.TRAINING_MODE_WORKINGDAY
        elif train_days < 56:  # 8 weeks
            return bc.TRAINING_MODE_WEEKDAY
        else:
            return bc.TRAINING_MODE_BIZDAY

    # calculate training mode for service
    @staticmethod
    def calc_training_mode_for_service(train_days):
        if train_days < 2:
            return bc.TRAINING_MODE_ERROR
        elif train_days < 7:
            return bc.TRAINING_MODE_DAILY
        else:
            return bc.TRAINING_MODE_WORKINGDAY

    @staticmethod
    def window_serving_data(target_id, past_input_dict, data_dict):
        '''
        past_input_dict : 서빙 데이터 60분치를 저장하는 백업용 객체
            - target_id를 key로 past_input_df를 value로 하는 dictionary 객체 \
        past_input_df : 특정 타겟의 서빙 데이터 60분치를 갖고있는 Pandas DataFrame
        data_dict : 실시간 서빙 데이터
        '''
        past_input_df = past_input_dict[target_id]
        # 서빙 데이터가 60분치 초과 되는 경우 예외 처리 (비정상 케이스), 최신 60분 치만 인덱싱
        if len(past_input_df) > 60:
            past_input_df = past_input_df.iloc[len(past_input_df) - 60:len(past_input_df)]
        # 서빙 데이터가 60분치로 들어오는 경우 (정상 케이스), 가장 과거의 데이터를 제거
        if len(past_input_df) == 60:
            past_input_df = past_input_df.set_index("time")
            past_input_df = past_input_df.drop(past_input_df.iloc[0].name)
            past_input_df = past_input_df.reset_index()
        # 실시간 서빙 데이터 처리 로직
        input_df = pd.DataFrame.from_dict(data_dict)
        input_df = input_df.sort_values(by="time")
        input_df = input_df.drop_duplicates()
        # input_df 객체는 실시간 서빙 데이터(1분 치)로 인덱싱
        if len(past_input_df) == 59 and len(input_df) == 60:
            input_df = input_df.iloc[-1]
        # 실시간 서빙 데이터인 input_df를 past_input_df에 추가 후 input_df 객체로 할당
        # past_input_dict 객체에 input_df DataFrame 저장
        if len(past_input_df) < 60:
            input_df = past_input_df.append(input_df)
        past_input_dict[target_id] = input_df.copy()

        return past_input_dict, input_df

    @staticmethod
    def make_log_serving_data(past_input_df, data_dict):
        input_df = pd.DataFrame.from_dict(data_dict)

        past_input_df = past_input_df.loc[:, ~past_input_df.columns.duplicated()]
        input_df = input_df.loc[:, ~input_df.columns.duplicated()]

        return pd.concat([past_input_df, input_df])

    @staticmethod
    def set_except_msg(result_dict):
        return {"errno":result_dict["error_code"],"errmsg":result_dict["error_msg"]}

    @staticmethod
    def insert_anomaly_serving_dbsln_performance(
            m_dbsln,
            sys_id,
            inst_type,
            target_id,
            time,
            data_dict_cur,
            dbsln_pred,

    ):
        res = {
            "keys": [
                "time",
                "sys_id",
                "target_id",
                "type",
                "name",
                "real_value",
                "dbsln_value",
                "dbsln_lower",
                "dbsln_upper",
                "ma_value",
                "future_dbsln",
            ],
            "values": [],
        }

        d = dbsln_pred.loc[time]  # Series 변환

        for feature_name in m_dbsln.features:
            if feature_name in data_dict_cur.keys():
                future_dbsln = Utils.make_future_dbsln_json(
                    feature_name, dbsln_pred
                )
                res["values"].append(
                    [
                        time,
                        sys_id,
                        target_id,
                        inst_type,
                        feature_name,
                        np.array(data_dict_cur[feature_name]).astype(np.float64),
                        d[feature_name],
                        d[f"{feature_name}_lower"],
                        d[f"{feature_name}_upper"],
                        d[f"{feature_name}_avg"],
                        future_dbsln,
                    ]
                )
        return res

    @staticmethod
    def insert_anomaly_serving_dbsln_result(
            m_dbsln,
            sys_id,
            inst_type,
            results,
            data_dict_cur,
            dbsln_pred,
    ):
        time = results["time"]
        target_id = results["target_id"]
        anomalies = results["anomalies"]

        res = {
            "keys": [
                "time",
                "sys_id",
                "target_id",
                "type",
                "name",
                "real_value",
                "dbsln_value",
                "dbsln_lower",
                "dbsln_upper",
                "ma_value",
                "future_dbsln",
                "is_anomaly",
                "avg",
                "std",
                "deviation",
                "zscore",
                "ascore",
                "score",
                "level",
            ],
            "values": [],
        }

        d = dbsln_pred.loc[time]  # Series 변환

        for feature_name in m_dbsln.features:
            if feature_name in data_dict_cur.keys() and feature_name in d.index:
                future_dbsln = Utils.make_future_dbsln_json(
                    feature_name, dbsln_pred
                )

                is_anomaly = False
                anomaly_dict = {}
                performance_list = []
                anomaly_list = []

                if len(anomalies) > 0:
                    for anomaly in anomalies:
                        if anomaly["name"] == feature_name:
                            is_anomaly = True
                            anomaly_dict = anomaly
                            break

                performance_list = [
                    time,
                    sys_id,
                    target_id,
                    inst_type,
                    feature_name,
                    np.array(data_dict_cur[feature_name]).astype(np.float32).round(2) if data_dict_cur[feature_name] is not None else None,
                    d[feature_name],
                    d[f"{feature_name}_lower"],
                    d[f"{feature_name}_upper"],
                    d[f"{feature_name}_avg"],
                    future_dbsln,
                    is_anomaly
                ]

                anomaly_list = Utils.make_anomaly_dbsln_list(is_anomaly, anomaly_dict)

                res["values"].append(
                    performance_list + anomaly_list
                )

        return res

    @staticmethod
    def insert_anomaly_serving_seqattn_result(inst_type, results, data_dict_cur, seqattn_pred):
        res = {'keys': ['time', 'sys_id', 'target_id', 'type', 'name', 'real_value', 'seqattn_lower', 'seqattn_upper', 'is_anomaly', 'deviation', 'zscore', 'ascore', 'score', 'level'],
               'values': []}

        anomaly_feats = {anomaly['name']: anomaly for anomaly in results['seqattn_anomalies']}
        feat2anomaly_list = {feat: [True, anomaly_feats[feat]['deviation'], anomaly_feats[feat]['zscore'], anomaly_feats[feat]['ascore'], anomaly_feats[feat]['score'], anomaly_feats[feat]['level']] if feat in anomaly_feats else [False, '', '', '', '', ''] for feat in seqattn_pred['features']}
        res['values'] = [[time, results['sys_id'], results['target_id'], inst_type, feat, data_dict_cur[feat], seqattn_pred[f"{feat}_lower"], seqattn_pred[f"{feat}_upper"]] + feat2anomaly_list[feat] for feat in seqattn_pred['features']]

        return res

    @staticmethod
    def make_anomaly_dbsln_list(is_anomaly, anomaly_dict):
        anomaly_values_cnt = 7

        if is_anomaly:
            anomaly_list = [
                anomaly_dict["avg"],
                anomaly_dict["std"],
                anomaly_dict["deviation"],
                anomaly_dict["zscore"],
                anomaly_dict["ascore"],
                anomaly_dict["score"],
                anomaly_dict["level"],
            ]
        else:
            anomaly_list = [''] * anomaly_values_cnt

        return anomaly_list

    @staticmethod
    def decimal_point_discard(n, point):
        if "float" not in str(type(n)):
            return n
        return np.floor(n * pow(10, point)) / pow(10, point)

class Query:
    def __init__(self, db_conn_str, logger):
        """
        pg에 접속과 접속 종료 (close)를 관리하는 클래스
        프로그램 흐름 : __init__() -> connect() -> make_cursor() -> query()
        생성자에서 접속을 생성하고 커서를 만듬
        소멸자에서 접속을 닫고 커서를 reloase함.

        Examples
        ----------
        Query.CreateQuery() : factory 패턴으로 인스턴스를 생성험.

        Parameters
        ----------
        db_conn_str : ip, port, user, password 정보가 있는 postgresql 접속 정보
        logger : 로거
        """
        self.logger = logger
        self.conn = None
        self.cursor = None
        self.db_conn_str = db_conn_str
        self._init()

    def _init(self):
        self._connect()
        self._make_cursor()

    def _connect(self):
        try:
            self.conn = pg2.connect(self.db_conn_str)
            self.logger.info("[Query] connection get")
        except Exception as e:
            self.logger.info(
                f"[{type(self).__name__}] Unexpected exception occurred while connect() - {e}"
            )

    def _make_cursor(self):
        try:
            self.cursor = self.conn.cursor(cursor_factory=pg2.extras.DictCursor)
            self.logger.info("[Query] cursor get")
        except Exception as e:
            self.logger.info(
                f"[{type(self).__name__}] Unexpected exception occurred while make_cursor() - {e}"
            )

    def query(self, sql):
        result = None
        try:
            result = psql.read_sql(sql, self.conn)
            self.logger.info("[Query] read_sql()")
        except Exception as e:
            self.logger.info(
                f"[{type(self).__name__}] Unexpected exception occurred while  psql.read_sql() - {e}"
            )
        finally:
            return result

    def cursor_execute(self, data_insert_query, input_tuple=None):
        try:
            start_time = datetime.datetime.now()
            if input_tuple is None:
                self.cursor.execute(data_insert_query)
            else:
                self.cursor.execute(data_insert_query, input_tuple)
            self.conn.commit()
            self.logger.info("[Query] complete insertion")
            end_time = datetime.datetime.now()
            self.logger.info(
                f"[Query] cursor_execute finished. it took {(end_time - start_time).total_seconds()} seconds"
            )
            return True
        except Exception as e:
            self.logger.info(
                f"[{type(self).__name__}] Unexpected exception occurred while  cursor_execute() - {e}"
            )

    def __del__(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    @staticmethod
    def CreateQuery(config, logger):
        # db
        db_conn_str = config.get("db_conn_str", None)
        if db_conn_str is None:
            pq_config = config["postgres"]
            db_conn_str = (
                f"host={pq_config['host']} "
                f"port={pq_config['port']} "
                f"dbname={pq_config['database']} "
                f"user={pq_config['id']} "
                f"password={pq_config['password']}"
            )
        db_query = Query(db_conn_str, logger)
        return db_query

    @staticmethod
    def get_alarm_level_info_from_db(db_conn_str, sys_id, inst_type):
        conn = None
        cursor = None
        data = None
        try:
            conn = pg2.connect(db_conn_str)
            cursor = conn.cursor(cursor_factory=pg2.extras.DictCursor)

            statement = (
                "SELECT attention_level, warning_level, critical_level "
                "FROM aiops_alarm_level_info_ "
                "WHERE sys_id = %(sys_id)s "
                "and alarm_type =  %(inst_type)s"
            )

            cursor.execute(statement, {"sys_id": sys_id, "inst_type": inst_type})

            row = cursor.fetchall()

            data = pd.DataFrame(
                row, columns=["attention_level", "warning_level", "critical_level"]
            )

            conn.commit()

        except Exception as e:
            row = [['1,3', '4,5', '6,7']]
            data = pd.DataFrame(
                row, columns=["attention_level", "warning_level", "critical_level"]
            )
            # print(f"[Error] Unexpected exception during serving : {e}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

        return data

    @staticmethod
    def update_module_status_by_training_id(db_conn_str, train_history_id, data: dict, train_process, progress, process_start_time=None):
        for k, v in data.items():
            if k == train_process:
                data[k][bc.PROGRESS] = progress
                data[k][bc.DURATION_TIME] = process_start_time if process_start_time is not None else None
                break

        Query.update_training_progress(db_conn_str, train_history_id, data)

    @staticmethod
    def update_training_progress(db_conn_str, train_history_id, data):
        conn = None
        cursor = None

        try:
            conn = pg2.connect(db_conn_str)
            cursor = conn.cursor(cursor_factory=pg2.extras.DictCursor)

            statement = (
                "UPDATE ai_history_train "
                "SET module_status = %(data)s "
                "WHERE history_train_id = %(train_history_id)s "
            )

            cursor.execute(statement, {"data": json.dumps(data), "train_history_id": train_history_id})

            conn.commit()

        except Exception as e:
            print(f"[Error] Unexpected exception during serving : {e}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
