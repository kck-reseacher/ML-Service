import os
from pathlib import Path

import numpy as np
from tensorflow.keras.utils import to_categorical, Sequence

from algorithms import aimodel
from algorithms.logseq.log2template import Log2Template
from common import constants, aicommon
from common.error_code import Errors
from common.memory_analyzer import MemoryUtil
from common.redisai import REDISAI


################################################################
# template_index range (1 ~ n) => softmax_index range (0 ~ n-1)
################################################################

class LogDataGenerator(Sequence):
    def __init__(self, x_data, y_data, batch_size=1024):
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.x_data))
        np.random.shuffle(self.indexes)

    def __len__(self):
        return round(len(self.x_data) / self.batch_size)

    def __getitem__(self, idx):
        indexes = self.indexes[idx*self.batch_size: (idx+1)*self.batch_size]
        x_batch = self.x_data[indexes]
        y_batch = self.y_data[indexes]

        return x_batch, y_batch


class LogSeq(aimodel.AIModel):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.logseq_params = None
        self.window_size = 30
        self.batch_size = 1024
        self.epochs = 50
        self.hidden_size = 32
        self.test_perc = 30

        self.top_k = constants.DEFAULT_N_TOP_K
        self.anomaly_threshold = None

        # template model
        self.log2template = Log2Template(config, logger)
        self.most_freq_idxs = []

        # tf model
        self.model = None
        self.model_dir = None

        self.progress_name = constants.MODEL_B_LOGSEQ
        self.under_number_of_digits = 6#소수점 버림 자릿수

        self.service_id = constants.MODEL_S_LOGSEQ

        self.mu = MemoryUtil(logger)

    def init_serve(self):
        self.top_k = self.config['parameter']['service'][constants.MODEL_S_LOGSEQ]["range"][constants.MSG]['top_k']
        self.anomaly_threshold = self.config['parameter']['service'][constants.MODEL_S_LOGSEQ]["range"][constants.MSG]['anomaly_threshold']

    def set_top_k(self, parameter):
        update_val = parameter['parameter']['service'][constants.MODEL_S_LOGSEQ]["range"][constants.MSG]['top_k']
        self.logger.info(f"[LogSeq] top_k changed : {self.top_k} => {update_val}")
        self.top_k = update_val

    def set_anomaly_threshold(self, parameter):
        new_thres = parameter['parameter']['service'][constants.MODEL_S_LOGSEQ]["range"][constants.MSG]['anomaly_threshold']
        self.logger.info(f"[LogSeq] anomaly_threshold changed : {self.anomaly_threshold} => {new_thres}")
        self.anomaly_threshold = new_thres

    def get_sequence_data(self, data):
        x_data, y_data = [], []
        for i in range(len(data) - self.window_size):
            x_data.append(data[i:i + self.window_size])
            y_data.append(data[i + self.window_size])

        x_data, y_data = np.array(x_data), to_categorical(y_data, num_classes=self.log2template.n_templates)
        return x_data, y_data

    def predict(self, serving_logs_df):
        serving_logs_df = self.log2template.log2tidx(serving_logs_df)
        if serving_logs_df is None:
            return {'error_code': Errors.E711.value, 'error_msg': Errors.E711.desc}

        x_data, y_data = self.get_sequence_data(serving_logs_df["tidx_inp"])
        y_cmp = serving_logs_df['tidx_cmp'].values[self.window_size:]
        # model load and inference
        fpath = os.path.join(self.config['model_dir'])
        onnx_model_path = str(Path(fpath) / f"{constants.MODEL_S_LOGSEQ}" / "onnx_model.onnx")
        onnx_model_key = REDISAI.make_redis_model_key(onnx_model_path, ".onnx")
        preds = list(map(lambda x: list(enumerate(x)), REDISAI.inference(onnx_model_key, x_data, data_type='int64')[0]))
        # h5_preds = list(map(lambda x: list(enumerate(x)), self.model.predict(x=x_data, batch_size=512)))
        preds = np.array(list(map(lambda p: sorted(p, key=lambda x: x[1], reverse=True), preds)))
        preds = preds[:, :self.top_k, :]
        '''
        # format
        (tidx_1, proba_1) => top1
        (tidx_2, proba_2) => top2
               ...
        (tidx_k-1, proba_k-1)
        (tidx_k, proba_k) => topk
        '''

        result = {}
        total_anomaly_cnt = 0
        for i in range(len(preds)):
            is_anomaly = np.argmax(y_data[i]) not in set(preds[i][:, 0]).union(self.most_freq_idxs) or y_cmp[i] == -1
            # expected_templates = self.log2template.tidx2template(preds[i][:, 0])
            expected_tidxs = preds[i][:, 0] + 1
            probas = preds[i][:, 1]
            result[str(i)] = {'anomaly': is_anomaly,
                              'real': serving_logs_df['msg'].iloc[self.window_size+i],
                              'tidx': np.argmax(y_data[i]) + 1,
                              'expected_tidxs': expected_tidxs,
                              # 'expected_templates': expected_templates,
                              'probabilities': probas,
                              'line_no': serving_logs_df['line_no'].iloc[self.window_size+i],
                              'time': serving_logs_df['_time'].iloc[self.window_size+i]
                              }
            if is_anomaly:
                total_anomaly_cnt += 1
            # self.logger.debug(f"result_{i} = {result[str(i)]}")

        self.logger.info(f"total_anomaly_cnt : {total_anomaly_cnt}")
        return result

    def load(self, model_dir):
        try:
            self.model_dir = model_dir
            self.log2template.load(model_dir)
            self.most_freq_idxs = self.log2template.get_normal_idxs()

            etc_path = os.path.join(model_dir, f"{constants.MODEL_S_LOGSEQ}/etc_info.pkl")
            etc_key = REDISAI.make_redis_model_key(etc_path, ".pkl")
            etc_info = REDISAI.inference_joblib(etc_key)
            self.logger.info(f"[LogSeq] etc_info loaded (4/4)")
            # self.top_k = etc_info['top_k']
            self.log2template.mined_period = etc_info['mined_period']

            return True
        except Exception as e:
            self.logger.info(f"[LogSeq] Error log while Load() : {e}")
            return False
