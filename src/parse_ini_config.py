import configparser


def get_config(config_filepath="../sample.ini"):
    config_parser = configparser.ConfigParser()
    config_parser.read(config_filepath)
    conf_dict = {section: dict(config_parser.items(section)) for section in config_parser.sections()}
    return conf_dict


