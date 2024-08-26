import os, shutil
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import DataLoader
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from algorithms import aimodel
from algorithms.ocdigcn.graph_util import generate_graph_data, ParseDataset
from common import constants
from common.error_code import Errors
from common.redisai import REDISAI

class OCDiGCN(aimodel.AIModel):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        #torch graph model
        self.algo_name = constants.MODEL_S_DIGCN
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.center = None
        self.num_features = None
        self.h_dims = 128
        self.h_layers = 2
        self.batch_size = 64
        self.lr = 0.01
        self.weight_decay = 1e-4
        self.epochs = 100
        self.graph_con_num = 30

        drain_config = TemplateMinerConfig()
        self.template_miner = TemplateMiner(config=drain_config)
        self.template_df = pd.DataFrame(columns=['EventId', 'EventTemplate', 'Occurrences'])
        self.normal_keyword_list = ['info', 'debug']
        self.error_keyword_list = ['error', 'fatal', 'fail', 'exception', 'critical']

        self.score_threshold = None  # distance from center (float value)
        self.cnt_threshold = self.config['parameter'][self.algo_name]['threshold']['anomaly_threshold']

    def init_config(self, config):
        self.config = config
        self.model_dir = self.config['model_dir']
        self.score_threshold = self.config[self.algo_name]["score_threshold"]['thre_5']

    def set_template_df(self):
        for cluster in self.template_miner.drain.clusters:
            self.template_df = self.template_df.append(pd.Series([cluster.cluster_id, ' '.join(cluster.log_template_tokens), cluster.size],
                                                       index=self.template_df.columns), ignore_index=True)
        self.num_features = len(self.template_df) + 1

    def matching_template_idxs(self, log_df):
        self.logger.info(f"[DiGCN] drain3 Matching start")
        time_matching_s = time.time()
        msg_lines = list(log_df['msg_transform'].values)
        tidx_list = [self.template_miner.match(msg).cluster_id if self.template_miner.match(msg) else self.num_features for msg in msg_lines]
        self.logger.info(f"[DiGCN] Matching end (elapsed = {time.time() - time_matching_s:.2f}s)")
        log_df['tidx'] = tidx_list

        return log_df

    def del_graph_data(self, train=False):
        file_path = os.path.join(self.config['model_dir'], f"{self.algo_name}", "serving")
        if train:
            file_path = os.path.join(self.config['model_dir'], f"{self.algo_name}", "train")
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)

    async def predict(self, standard_datetime, log_df):
        self.logger.info(f"digcnlog's serving starts.")
        start_digcn = time.time()

        self.del_graph_data()
        redis_model_key = REDISAI.make_redis_model_key(str(Path(self.config['model_dir'])/self.algo_name/f"{self.algo_name}_model"), "")
        redis_template_miner_key = REDISAI.make_redis_model_key(str(Path(self.config['model_dir'])/self.algo_name/"template_miner"), "")
        redis_center_key = REDISAI.make_redis_model_key(str(Path(self.config['model_dir'])/self.algo_name/"center"), "")
        try:
            self.template_miner = REDISAI.inference_pickle(redis_template_miner_key)
            center = REDISAI.inference_pickle(redis_center_key)
        except Exception as ex:
            self.logger.error(f"error occurred during inference redisai pickle. {ex}.")
            return {'error_code': Errors.E714.value, 'error_msg':  Errors.E714.desc}, None

        self.set_template_df()
        log_df = self.matching_template_idxs(log_df)
        g_idxs = generate_graph_data(log_df, self.template_df, self.graph_con_num, self.config['model_dir'] + f'/{self.algo_name}/serving')

        tidx_list = list(log_df['tidx'])
        decom_df = pd.DataFrame(columns=['eventID', 'score_list'])
        decom_df['eventID'] = tidx_list

        try:
            graph_dataset = ParseDataset(root=self.config['model_dir'], name=f'/{self.algo_name}/serving')
        except Exception as e:
            self.logger.info(f"Unable to serve digcn model due to no graph data. (input log is all same template.)")
            return {'error_code': Errors.E703.value, 'error_msg': 'no graph data'}, None

        serving_loader = DataLoader(graph_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)
        self.logger.info(f"num dataset ===== graph: {len(graph_dataset)}, loader:{len(serving_loader)}")

        with torch.no_grad():
            idx, predict_idxs = 0, []
            for batch in serving_loader:
                try:
                    pred = torch.Tensor(REDISAI.inference_digcn(redis_model_key, batch))

                    '''graph decomposition'''
                    input_template_idxs = tidx_list[g_idxs[idx] - 29: g_idxs[idx] + 1]
                    node_list = list(dict.fromkeys(input_template_idxs))

                    score_dic = {}  # score_dic = {tidx_6: score_6, tidx_23: score_23, ...}
                    for i in range(len(node_list)):
                        score = torch.sum((pred[i] - center) ** 2)
                        score_dic[node_list[i]] = round(float(score), 5)

                    # stack score to input data
                    input_idxs = np.arange(g_idxs[idx] - 29, g_idxs[idx] + 1)
                    for i in range(len(input_idxs)):
                        if np.isnan(decom_df['score_list'].iloc[input_idxs[i]]).all():
                            decom_df['score_list'].iloc[input_idxs[i]] = [score_dic[input_template_idxs[i]]]
                        else:
                            decom_df['score_list'].iloc[input_idxs[i]].append(score_dic[input_template_idxs[i]])

                    predict_idxs.append(idx)
                    idx += 1
                except Exception as e:
                    self.logger.debug(e)
                    idx += 1
                    continue
        decom_df['decom_avg_score'] = decom_df['score_list'].apply(np.mean)
        result, header = self.post_processing_result(log_df, decom_df, standard_datetime)
        self.logger.info(f'[Asynchronous-digcn] time taken: {round(time.time() - start_digcn, 5)}')
        return result, header

    def post_processing_result(self, log_df, decom_df, standard_datetime):
        res = {"keys": ["time", "sys_id", "inst_type", "target_id", "metric", "line_no", "anomaly", "real_log", "anomaly_score"],
               "values": []}

        if decom_df is None:
            return res, None

        anomaly_count, total_count = 0, 0
        self.logger.debug(f"anomaly score threshold: {self.score_threshold}")
        for i in range(len(decom_df)):
            if pd.to_datetime(log_df['time'].iloc[i]) >= pd.to_datetime(standard_datetime):
                total_count += 1

                if decom_df['decom_avg_score'].iloc[i] > self.score_threshold:
                    self.logger.debug(f"anomaly detected. score: {decom_df['decom_avg_score'].iloc[i]}")
                    line_no = log_df['line_no'].iloc[i]
                    anomaly_count += 1

                    res["values"].append(
                        [
                            log_df['time'].iloc[i],
                            self.config['sys_id'],
                            self.config['inst_type'],
                            self.config['target_id'],
                            constants.MSG,  # matric_name
                            int(float(line_no)) if type(float(line_no)) is float else int(line_no),
                            True,
                            log_df['msg'].iloc[i],
                            round(decom_df['decom_avg_score'].iloc[i], 5)
                        ]
                    )

        # make digcn serving header
        header = {}
        serving_time = standard_datetime
        header["time"] = datetime.strptime(serving_time[:16], '%Y-%m-%d %H:%M') if len(serving_time) >= 16 else serving_time
        header["sys_id"] = self.config['sys_id']
        header["target_id"] = self.config['target_id']
        header["total_count"] = total_count
        header["anomaly_count"] = anomaly_count
        header["anomaly"] = bool(anomaly_count > self.cnt_threshold)
        header["anomaly_threshold"] = self.cnt_threshold

        if header["anomaly_count"] == 0:
            header = None

        return res, header