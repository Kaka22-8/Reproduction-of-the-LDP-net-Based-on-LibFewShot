# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

import os
import torch
from core.config import Config
from core import Test


PATH = "./results/LDPNet-miniImageNet--ravi-resnet10-5-5-Apr-10-2026-16-29-52"
VAR_DICT = {
    "data_root": "D:\datasets\CUB_200_2011_FewShot",
    "image_size": 224,
    "augment": False,
    "augment_times": 1,
    "augment_times_query": 1,
    "test_way": 5,
    "test_shot": 1,
    "test_query": 15,
    "test_episode": 600,
    "episode_size": 1,
    "test_epoch": 5,
    "device_ids": "0",
    "n_gpu": 1,
}



def main(rank, config):
    test = Test(rank, config, PATH)
    test.test_loop()


if __name__ == "__main__":
    config = Config(os.path.join(PATH, "config.yaml"), VAR_DICT).get_config_dict()

    if config["n_gpu"] > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
        torch.multiprocessing.spawn(main, nprocs=config["n_gpu"], args=(config,))
    else:
        main(0, config)
