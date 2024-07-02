#!/usr/bin/python3
"""
model_path=./model/mseg-3m.pth

--model_name=mseg-3m --model_path=./model --input_file=./data
python -u mseg_semantic/tool/universal_demo.py \
  --config=${config} model_name ${model_name} model_path ${model_path} input_file ${input_file}


Run inference over all images in a directory, over all frames of a video,
or over all images specified in a .txt file.
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import mseg.utils.names_utils as names_utils

import mseg_semantic.utils.logger_utils as logger_utils
from mseg_semantic.utils import config
from mseg_semantic.utils.config import CfgNode
from mseg_semantic.tool.inference_task import InferenceTask
import os



_ROOT = Path(__file__).resolve().parent.parent


logger = logger_utils.get_logger()

cv2.ocl.setUseOpenCL(False)


def run_universal_demo(args, use_gpu: bool = True) -> None:
    """Run model inference on image(s)/video, with predictions provided in the universal taxonomy.

    Args:
        args:
        use_gpu
    """
    if "scannet" in args.dataset:
        args.img_name_unique = False
    else:
        args.img_name_unique = True

    args.u_classes = names_utils.get_universal_class_names()
    args.print_freq = 10

    args.split = "test"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    logger.info(args)
    logger.info("=> creating model ...")
    args.num_model_classes = len(args.u_classes)

    itask = InferenceTask(
        args,
        base_size=args.base_size,
        crop_h=args.test_h,
        crop_w=args.test_w,
        input_file=args.input_file,
        model_taxonomy="universal",
        eval_taxonomy="universal",
        scales=args.scales,
    )
    itask.execute()



def get_parser() -> CfgNode:


    """ """
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation")
    parser.add_argument(
        # "--config", type=str, default=f"{_ROOT}/config/final_test/1080/default_config_360.yaml", help="config file"
        "--config",type=str, default=f"{_ROOT}/config/default_config_360_ss.yaml", help="config file"

    )
    parser.add_argument('--input_file', default='/data/xc/datasets/semantic_kitti/sequences/image/00/image_2', help='Data directory')
    parser.add_argument('--save_folder', default='/data/xc/datasets/semantic_image/00', help='Data directory')
    parser.add_argument(
        "--file_save", type=str, default="default", help="eval result to save, when lightweight option is on"
    )
    parser.add_argument('--model_path', default='./model/mseg-3m.pth', help='Pretrained weights directory.')
    parser.add_argument('--model_name', default="mseg-3m", help='Model name')
    parser.add_argument('--include-nms',action="store_true", default="'torchscript'-3m", help='export name')
    parser.add_argument(
        "opts", help="see config/ade20k/ade20k_pspnet50.yaml for all options", default=None, nargs=argparse.REMAINDER
    )  # model path is passed in

    args = parser.parse_args()
    # print(args)
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


if __name__ == "__main__":
    """ """
    use_gpu = True
    #use_gpu = False

    args = get_parser()

    if args.dataset == "default":
        args.dataset = Path(args.input_file).stem

    assert isinstance(args.model_name, str)
    assert isinstance(args.model_path, str)


    print(args)
    run_universal_demo(args, use_gpu)
