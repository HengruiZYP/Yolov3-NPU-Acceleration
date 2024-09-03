import json
import os
import argparse
import sys
import yaml
import numpy as np
import warnings
import cv2
import time
warnings.filterwarnings("ignore")

parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 2)))
sys.path.insert(0, parent_path)
from yolo import YoloV3, vis


def argsparser():
    """
    解析命令行参数，配置模型参数和优化器等参数
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="./model/config.json",
        help=("path of deploy config.json"),
    )
    parser.add_argument(
        "--infer_yml",
        type=str,
        default="./model/infer_cfg.yml",
        help=("path of infer_cfg.yml"),
    )
    parser.add_argument(
        "--test_image",
        type=str,
        default="./test_images/000000025560.jpg",
        help="Path of test image file.",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="whether to visualize."
    )
    parser.add_argument(
        "--with_profile", action="store_true", help="whether to predict with profile."
    )
    return parser


def main(args):
    """main"""
    # init YoloV3
    config = args.config
    assert os.path.exists(config), "deploy_config does not exist."
    
    infer_yml = args.infer_yml
    assert os.path.exists(infer_yml), "infer_yml does not exist."
    with open(infer_yml, "r") as f:
        infer_yml = yaml.safe_load(f)
        
    detector = YoloV3(config, infer_yml)
    
    cap = cv2.VideoCapture(0)
    # 获取摄像头的输入大小
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("w: ", width)
    print("h: ", height)
    
    if cap.isOpened():
        print("video opened")
        
    else:
        print("video not opened")
    
    # 帧率参数
    prev_time = time.time()
    frame_count = 0
    fps = 0
    
    while True:
        
        ret, frame = cap.read()
    	
        if not ret:
            break
            
        # 计算帧率        
        current_time = time.time()
        frame_count += 1        
        elapsed_time = current_time - prev_time
        
        if elapsed_time >= 1:            
            fps = frame_count / elapsed_time            
            prev_time = current_time            
            frame_count = 0
        
        # 在帧上绘制帧率        
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)
    	
    	# with_profile
        with_profile = args.with_profile
        if with_profile:
            results = detector.predict_profile(frame)
            print("total time: ", detector.total_time)
        else:
            results = detector.predict_image(frame)
            
        # visualize
        visualize = args.visualize
        if visualize:
            render_img = vis(frame, results)
            cv2.imshow('camera', render_img)
            
        if cv2.waitKey(5) & 0xff == 27:
            break
            
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argsparser()
    args = parser.parse_args()
    main(args)
