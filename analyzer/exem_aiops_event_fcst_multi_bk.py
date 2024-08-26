from pprint import pformat
import pandas as pd
from pathlib import Path
from analyzer import aimodule
from analyzer.exem_aiops_event_fcst_clf import ExemAiopsEventFcstClf

from common import constants
from common.constants import SystemConstants as sc
from common.redisai import REDISAI
from common.error_code import Errors
from common.module_exception import ModuleException
from resources.logger_manager import Logger


class ExemAiopsEventFcstMulti(aimodule.AIModule):
    def __init__(self, config, logger):
        """
        initialize instance attributes using JSON config information and Confluence documentation. AI Server inputs the JSON config information.
        :param config:
        :param logger:
        """
        self.config = config
        self.model_dir = self.config["model_dir"]
        self.log_dir = self.config["log_dir"]
        self.logger = logger
        self.target_logger = None
        self.logger.info(f"\t\t [Init] Create {pformat(config['inst_type'])} instance")
        self.event_fcst_clf = ExemAiopsEventFcstClf(self.config, self.logger)

    def init_train(self):
        pass

    def train(self, stat_logger):
        pass

    def end_train(self):
        pass

    def init_serve(self, reload=False):
        clf_res = self.event_fcst_clf.init_serve(reload)
        return clf_res

    def serve(self, header, body):
        #config update
        sys_id = header['sys_id']
        inst_type = header['inst_type']
        target_id = header['target_id']
        last_time = pd.to_datetime(header['predict_time'])
        config = self._get_config(sys_id, target_id)
        self.target_logger = self.create_logger(config)
        self.event_fcst_clf.config = config

        _body = {"event_fcst": {
            "keys": ["time", "predict_time", "sys_id", "group_type", "group_id", "event_prob", "explain", "event_doc", "event_code"]}}
        _body_clf = {'event_fcst': {'values': []}}

        try:
            self.target_logger.info(
                f"============ [{sys_id}_{inst_type}_{target_id}] Start data validation ============")
            body = self.preprocessing(header, body, last_time)
            body = self.df_to_dict(body, config, last_time)
            body = self.dict_to_df(body)
            self.target_logger.info(
                f"============ [{sys_id}_{inst_type}_{target_id}] Finish data validation ============")
            _header, _body_clf, errno, errmsg = self.event_fcst_clf.serve(header, body)

            # merge parsed serving result
            _body['event_fcst']['values'] = _body_clf['event_fcst']['values']

        except ModuleException as me:
            self.logger.error(f"{sys_id}_{inst_type}_{target_id}: {me}")
            return None, _body, me.error_code, me.error_msg

        except Exception as e:
            self.logger.error(f"{sys_id}_{inst_type}_{target_id}: {e}")
            return None, _body, Errors['E777'].value, Errors['E777'].desc

        finally:
            pass

        return None, _body, 0, None

    def preprocessing(self, header, body, last_time):
        body_key = f"{header['sys_id']}/exem_aiops_event_fcst/{header['inst_type']}/{header['target_id']}/body"
        input_df = pd.DataFrame(body)
        input_df['time'] = pd.to_datetime(input_df['time'], format='%Y-%m-%d %H:%M:%S')

        if REDISAI.exist_key(body_key):
            saved_df = REDISAI.inference_pickle(body_key)
            input_df = pd.concat([input_df, saved_df])
        time_filter2 = self._time_filter(input_df['time'], last_time, timedelta=5)
        saved_df = input_df.loc[time_filter2].drop_duplicates().sort_values(by='time', ascending=False, ignore_index=True)
        REDISAI.save_body_to_redis(body_key, saved_df)
        time_filter = self._time_filter(saved_df['time'], last_time)
        return saved_df.loc[time_filter].sort_values(by='time', ascending=False, ignore_index=True)

    @staticmethod
    def _time_filter(input_data, last_time, timedelta=0):
        '''
        Parameters
        ----------
        input_data: Timestamp type의 데이터
        last_time: header의 predict_time이나 standard_time
        timedelta: buffer 길이(min)

        Returns
        -------
        '''

        first_time = last_time - pd.Timedelta(minutes=(60 + timedelta))
        last_time = last_time + pd.Timedelta(minutes=timedelta)
        time_filter = (input_data > first_time) & (input_data <= last_time)
        return time_filter

    def _get_config(self, sys_id, target_id):
        model_config_path = self._get_model_config_path(self.model_dir, sys_id, target_id)
        model_config_key = REDISAI.make_redis_model_key(model_config_path, ".json")
        model_config = REDISAI.inference_json(model_config_key)
        return model_config

    def _get_model_config_path(self, model_dir, sys_id=None, target_id=None):
        return str(Path(
            model_dir) / f"{sys_id}" / f"{self.config['module']}" / f"{self.config['inst_type']}" / f"{target_id}" / "model_config.json")

    # df to dict
    def df_to_dict(self, df, config, last_time):
        df = df.pivot_table(index=['time', 'inst_type', 'target_id'], columns='name', values='real_value')
        df = df.reset_index()
        df = df.sort_values(by=['time'])
        c1 = 3 # 최근 c1 분 데이터 검사
        c2 = 5 # 피처당 c2개의 nan 허용

        df_dict = {}
        e = last_time
        self.target_logger.debug(f"[{config['sys_id']}_{config['inst_type']}_{config['target_id']}] standard time: {e}")
        s = e - pd.Timedelta(minutes=59)
        t_index = pd.DatetimeIndex(pd.date_range(start=s, end=e, freq='T'))
        grouped = df.groupby(['inst_type', 'target_id'])

        for group_key, group_df in grouped:
            inst_type, target_id = group_key
            group_df.index = pd.to_datetime(group_df["time"])
            each_target_df = group_df.reindex(t_index)

            # 필요한 컬럼만 자르기
            if inst_type == 'db':
                db_key = 'ORACLE'
                db_key = next((value for key, value in constants.DB_KEY_MAPPING.items() if key in target_id), db_key)
                columns_to_keep = config['parameter']['train']['eventfcst']['features'][inst_type][db_key]
            else:
                columns_to_keep = config['parameter']['train']['eventfcst']['features'][inst_type]
            temp_df = pd.DataFrame(columns = columns_to_keep)
            each_target_df = pd.concat([temp_df, each_target_df])
            each_target_df = each_target_df[columns_to_keep]

            # validate
            has_all_nans_last_c1 = each_target_df.iloc[-c1:].isna().all(axis=1)
            has_all_nans_count_c2 = each_target_df.isna().all(axis=1)
            if has_all_nans_last_c1.all():
                self.target_logger.info(f"[{config['sys_id']}_{config['inst_type']}_{config['target_id']}] "
                                         f"{inst_type}_{target_id} - invalid dataset(최근 3분치 데이터 없음) \n"
                                         f"{has_all_nans_last_c1[has_all_nans_last_c1].index}")
            elif has_all_nans_count_c2.sum() > c2:
                self.target_logger.info(f"[{config['sys_id']}_{config['inst_type']}_{config['target_id']}] "
                                         f"{inst_type}_{target_id} - invalid dataset(6분 이상의 데이터 없음) \n"
                                         f"{has_all_nans_count_c2[has_all_nans_count_c2].index}")
            # 그룹 내 타겟 데이터 중 특정 지표 전체 null인 경우, 서빙 데이터에서 해당 타겟 데이터 제외 후 나머지 타겟은 정상 서빙
            elif each_target_df.isnull().sum().values.max() == len(each_target_df):
                self.target_logger.info(f"[{config['sys_id']}_{config['inst_type']}_{config['target_id']}] "
                                         f"{inst_type}_{target_id} - invalid dataset(특정 지표 전체 null임)")
            else:
                if inst_type not in df_dict:
                    df_dict[inst_type] = {}
                df_dict[inst_type][target_id] = each_target_df
                self.target_logger.debug(f"[{config['sys_id']}_{config['inst_type']}_{config['target_id']}] "
                                         f"{inst_type}_{target_id} - valid dataset \n"
                                         f"{has_all_nans_count_c2[has_all_nans_count_c2].index}")
        return df_dict

    # dict to df
    def dict_to_df(self, df_dict):
        df_stack = pd.DataFrame()
        for inst_type, data in df_dict.items():
            for target, df in data.items():
                df["time"] = df.index
                df = pd.melt(df, id_vars='time', var_name='name', value_name='real_value')
                df["inst_type"] = inst_type
                df["target_id"] = target
                df_stack = pd.concat([df_stack, df])
        if len(df_stack) == 0:
            self.target_logger.error("not enough serving data")
            raise ModuleException("E704")
        return df_stack

    def create_logger(self, config):

        logger_dir = str(Path(self.log_dir) / sc.EXEM_AIOPS_EVENT_FCST_MULTI / config['inst_type'])
        logger_name = f"{config['module']}_{config['inst_type']}_{config['sys_id']}"
        error_log_dict = {
            "log_dir": str(Path(self.log_dir) / sc.ERROR_LOG_DEFAULT_PATH),
            "file_name": sc.EXEM_AIOPS_EVENT_FCST_MULTI,
        }
        logger = Logger().get_default_logger(logdir=logger_dir, service_name=logger_name, error_log_dict=error_log_dict)

        return logger