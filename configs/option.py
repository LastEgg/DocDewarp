import argparse
import yaml
from pathlib import Path
import os


def get_option(config_name, verbose=True):
    # 首先读取YAML配置文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建 config.yaml 的绝对路径
    yaml_path = os.path.join(current_dir, config_name)
    if Path(yaml_path).exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
    else:
        yaml_config = {}

    if verbose:
        print("-" * 30)
        print("Current Configuration:")
        for key, value in yaml_config.items():
            print(f"{key}: {value}")
        print("-" * 30)

    return argparse.Namespace(**yaml_config)


if __name__ == "__main__":
    config = get_option()
    print(config)
