import yaml


def parse_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    print(f"Config {config_path} parsed successfully")
    return cfg
