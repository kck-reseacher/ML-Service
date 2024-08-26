import os
import time
import glob
import shutil
import joblib
import argparse
import numpy as np
import tensorflow as tf

from abc import *
from pathlib import Path

from algorithms import aimodel
from algorithms import tsmixer
import algorithms.tsmixer.tsmixer_norm
import algorithms.tsmixer.tsmixer_rev_in
from algorithms.tsmixer.data_loader import TSFDataLoader

from common import constants
from common.onnx import ONNX
from common.aicommon import sMAPE
from common.aicommon import Query
from common.redisai import REDISAI
from common.timelogger import TimeLogger
from common.module_exception import ModuleException


class TSMixer(aimodel.AIModel, metaclass=ABCMeta):
    def __init__(self, id, config, logger):
        self.logger = logger
        self.config = config
        self.model_dir = self.config['model_dir']
        self.fcst_tsmixer = None
        self.scaler = {}
        self.parse_args()
        self.progress_name = constants.MODEL_S_TSMIXER

        self.except_failure_date_list = config.get('except_failure_date_list', [])
        self.except_business_list = config.get('except_business_list', [])

    def parse_args(self):
        parser = argparse.ArgumentParser(description='TSMixer for Time Series Forecasting')
        # basic config
        parser.add_argument('--seed', type=int, default=0, help='random seed')
        parser.add_argument('--model', type=str, default='tsmixer_rev_in', help='model name, options: [tsmixer_norm, tsmixer_rev_in]')

        parser.add_argument('--checkpoint_dir', type=str, default=self.model_dir, help='location of model checkpoints')
        parser.add_argument('--delete_checkpoint', action='store_true', help='delete checkpoints after the experiment')

        # forecasting task
        parser.add_argument('--window_size', type=int, default=60, help='input sequence length')
        parser.add_argument('--pred_len', type=int, default=30, help='prediction sequence length')

        # model hyperparameter
        parser.add_argument('--n_block', type=int, default=2, help='number of block for deep architecture')
        parser.add_argument('--ff_dim', type=int, default=128, help='fully-connected feature dimension',)
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
        parser.add_argument('--norm_type', type=str, default='B', choices=['L', 'B'],help='LayerNorm or BatchNorm')
        parser.add_argument('--activation', type=str, default='gelu', choices=['relu', 'gelu'], help='Activation function')
        parser.add_argument('--kernel_size', type=int, default=8, help='kernel size for CNN')
        parser.add_argument('--temporal_dim', type=int, default=128, help='temporal feature dimension')
        parser.add_argument('--hidden_dim', type=int, default=64, help='hidden feature dimension')

        # optimization
        parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
        parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
        parser.add_argument('--batch_size', type=int, default=2048, help='batch size of input data')
        parser.add_argument('--learning_rate', type=float, default=0.05, help='optimizer learning rate')
        parser.add_argument('--patience', type=int, default=5, help='number of epochs to early stop')

        self.args = parser.parse_args(args=[])

        tf.keras.utils.set_random_seed(self.args.seed)

    def fit(self, df, train_progress = None):

        feats = list(df.columns)

        tsm_parser = TSFDataLoader(df, self.args.batch_size, self.args.window_size, self.args.pred_len)

        self.exp_id = f'{self.args.model}_sl{self.args.window_size}_pl{self.args.pred_len}_lr{self.args.learning_rate}_nt{self.args.norm_type}_{self.args.activation}_nb{self.args.n_block}_dp{self.args.dropout}_fd{self.args.ff_dim}'

        train_data = tsm_parser.get_train()
        val_data = tsm_parser.get_val()
        test_data = tsm_parser.get_test()

        self.logger.info(f"model training start")
        with TimeLogger(f"[tsmixer] model training time :", self.logger):
            build_model = getattr(tsmixer, self.args.model).build_model

            model = build_model(
                input_shape=(self.args.window_size, tsm_parser.n_feature),
                pred_len=self.args.pred_len,
                norm_type=self.args.norm_type,
                activation=self.args.activation,
                dropout=self.args.dropout,
                n_block=self.args.n_block,
                ff_dim=self.args.ff_dim,
                target_slice=tsm_parser.target_slice,
            )

            optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.learning_rate)
            model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanSquaredError()])
            checkpoint_path = os.path.join(self.args.checkpoint_dir, f'{self.exp_id}_best')
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
            )
            early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.args.patience)
            start_training_time = time.time()
            model.fit(
                train_data,
                epochs=self.args.train_epochs,
                validation_data=val_data,
                callbacks=[checkpoint_callback, early_stop_callback],
            )
            end_training_time = time.time()
            elasped_training_time = end_training_time - start_training_time

            self.logger.info(f'Training finished in {elasped_training_time} secconds')
            model.load_weights(checkpoint_path)

            Query.update_module_status_by_training_id(self.config['db_conn_str'], self.config['train_history_id'],
                                                      train_progress,
                                                      self.progress_name,
                                                      100, round(elasped_training_time))

            if self.args.delete_checkpoint:
                for f in glob.glob(checkpoint_path + '*'):
                    os.remove(f)

            X_test = list()
            y_test = list()

            for x, y in test_data:
                X_test.append(y)
                y_test.append(model.predict(x))

            X_test = np.concatenate(X_test)
            y_test = np.concatenate(y_test)

            tsmixer_res_dict = dict()

            for i in range(len(feats)):  # 지표별
                tmp_smape = list()

                for j in range(X_test.shape[1]):  # 시점별
                    tmp_s = sMAPE(tsm_parser.inverse_transform(X_test[:, j, :])[:, i], np.round(tsm_parser.inverse_transform(y_test[:, j, :])[:, i], 2))
                    tmp_smape.append(tmp_s)

                tsmixer_res_dict[feats[i]] = {'smape': np.round(np.mean(np.array(tmp_smape)), 2)}

        self.logger.info("start to save")
        self.save(tsm_parser, model)
        self.logger.info("success save model")

        res = {self.progress_name:
                    {
                        'duration_time': elasped_training_time,
                        'features': feats,
                        'train_metrics': tsmixer_res_dict
                     }
                }

        return res, None, 0, None

    def predict(self, df):
        path = str(Path(self.model_dir))

        with TimeLogger(f"[{self.progress_name} Serving] elapsed time :", self.logger):
            if any(columns not in df.columns for columns in self.pred_feats):
                self.logger.error(f"The learned feature does not exist in the serving data.")
                raise ModuleException('E706')

            scaler_key = REDISAI.make_redis_model_key(f"{path}/{self.progress_name}_scaler", "")
            scaler = REDISAI.inference_joblib(scaler_key)

            onnx_feat_model_key = REDISAI.make_redis_model_key(f"{path}/{self.progress_name}", '')

            df = df[self.pred_feats]

            if df.isnull().sum().sum() > 0:
                feat_data = df.astype('float32').interpolate(limit_direction='both').fillna(method='ffill').fillna(method='bfill')
            else:
                feat_data = df.astype('float32')

            feat_data = feat_data[-self.args.window_size:]

            try:
                feat_data_scaled_rep = scaler.transform(feat_data)[np.newaxis, :, :]
                preds = REDISAI.inference(onnx_feat_model_key, feat_data_scaled_rep)[0]

            except Exception as e:
                raise ModuleException('E705')

        return np.round(np.maximum(scaler.inverse_transform(np.squeeze(preds)), 0), 2)

    def init_config(self, config):
        self.config = config
        self.model_dir = self.config['model_dir']
        self.params = config['results'][self.progress_name]
        self.pred_feats = self.params['features']

    def save_files(self, path, tsm_parser, model):

        feat_scaler_path = os.path.join(path, f"{self.progress_name}_scaler.pkl")
        onnx_feat_model_path = os.path.join(path, f"{self.progress_name}.onnx")

        # Convert Tensorflow to ONNX
        ONNX.onnx_save(model, onnx_feat_model_path)

        # scaler
        joblib.dump(tsm_parser.scaler, feat_scaler_path)

        self.logger.info(f"model/scaler have been saved to {self.model_dir}")

    def save(self, tsm_parser, model):

        path = f'{str(Path(self.model_dir))}'

        if os.path.exists(path):
            shutil.rmtree(path)

        Path(path).mkdir(exist_ok=True, parents=True)

        self.save_files(path, tsm_parser, model)