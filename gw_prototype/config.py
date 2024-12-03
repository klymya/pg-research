import configparser

def get_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config
