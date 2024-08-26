import datetime
import time
from common import constants

from logpresso import LogpressoClient

class Logpresso():

    def __init__(self, config):
        self.logger = None
        self.config = config

        self.logpresso_url = config['logpresso_info']['host']
        self.logpresso_port = int(config['logpresso_info']['port'])
        self.logpresso_id = config['logpresso_info']['id']
        self.logpresso_pw = config['logpresso_info']['password']

    def init_logpresso(self, logger):
        self.logger = logger

    def get_log_message(self, query, algorithm=None):
        log_messages = list()

        start_time = time.time()

        try:

            with LogpressoClient(
                self.logpresso_url,
                self.logpresso_port,
                self.logpresso_id,
                self.logpresso_pw
            ) as client:
                with client.query(query) as cursor:
                    for row in cursor:
                        line = dict(row.data())
                        if line["line"] == "": continue

                        if algorithm == constants.MODEL_S_LOGSEQ:
                            line = self.make_logseq_train_format(line)

                        log_messages.append(line)

                        self.logger.debug(f"line : {line}")
        except Exception as e:
            self.logger.exception(f"error occurred in Logpresso")

        finally:
            if client:
                client.client.close()

        self.logger.info(f"** query : {query} | query time : {time.time() - start_time}s | fetched data len : {len(log_messages)}")

        return log_messages

    def make_proc_sparse_query(self, lp_request_query_time, lp_table_name, lp_host_tag):
        query = f'proc sparse_log_drain("{lp_request_query_time}", "{lp_table_name}", "{lp_host_tag}")'
        self.logger.debug(f"LP maked query : {query}")
        return query

    @staticmethod
    def make_logseq_train_format(line):
        remake_dict = dict()
        remake_dict['msg'] = line["line"]
        remake_dict['_time'] = line['time']

        return remake_dict