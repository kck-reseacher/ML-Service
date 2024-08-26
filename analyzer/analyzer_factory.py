from common import aicommon


class AnalyzerFactory:
    def __init__(self, logger):
        self.logger = logger

    def get_analyzer(self, path, param, multi_module_name=None):
        instance = None
        try:

            module_name = param["module"] if multi_module_name is None else multi_module_name
            self.logger.info(f"module_name : {module_name}")

            class_name = aicommon.Utils.to_camel_case(module_name)
            self.logger.info(f"class_name : {class_name}")
            self.logger.info(f"path : {path}")

            target_class = aicommon.Utils.get_module_class(
                module_name, class_name, path
            )
            instance = target_class(param, self.logger)
        except Exception as e:
            self.logger.exception(
                f"[Error] Unexpected exception during serving : {e}"
            )
            raise e
        return instance
