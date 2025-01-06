from tools.datasets_invd3d.datasets import *
from configs.option import get_option
from tqdm import tqdm
import torch
from configs.option import get_option
from tools.datasets_invd3d.datasets_cache import *



opt = get_option("config_invd3d.yaml")


train_dataloader, valid_dataloader = get_dataloader(opt)

for i, batch in enumerate(tqdm(train_dataloader)):
    pass