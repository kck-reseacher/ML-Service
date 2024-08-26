import base64


class Base64Util:

    def base64encoding(input: str) -> str:
        return str(base64.b64encode(input.encode()), encoding='utf-8')

    def base64decoding(input: str) -> str:
        return base64.b64decode(input).decode(encoding='utf-8')

    @staticmethod
    def get_config_decode_value(config):
        if type(config) is dict:
            for key in config.keys():
                config[key] = Base64Util.base64decoding(config[key])

        return config
