# 一、模型说明
该模型(yolov3)用于目标检测任务，基于PaddleDetection2.6进行模型训练，paddle版本使用2.4.0，输入数据尺寸可以为320x320/416x416/608x608，数据格式为coco格式。 

# 二、效果预览

![image](Yolov3-NPU-Acceleration/vedio/yolov3.gif)

# 三、使用方式
## 3.1 模型训练
__模型生产基于aistudio平台进行__，确保已有aistudio账号。

aistudio地址：https://aistudio.baidu.com/aistudio/index


### 3.1.1 环境准备

aistudio创建项目, 选择paddle2.4.0版本。

### 3.1.2 模型训练、评估、导出
请参考项目：[aistudio项目链接](https://aistudio.baidu.com/projectdetail/6595113?contributionType=1&sUid=1318783&shared=1&ts=1701078320517)

__请参考如下版本__：![](res/aistudio_version.jpg)

## 3.2 模型转换

### 3.2.1 已转换模型
本项目已转换好使用coco数据集训练的yolov3-darknet53模型，置于model文件夹内供使用。

## 3.3 模型部署
__模型部署基于板卡进行__

首先安装依赖库，确保当前位于/home/edgeboard/yolov3-python目录下：
```bash
sudo pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
同时yolov3的部署额外需要onnxruntime软件，安装：
```bash
sudo apt update
sudo apt install onnxruntime
```
## 3.3.1 ppnc推理  
- 将模型生产阶段产生的model.nb、model.json、model.po、model.onnx模型文件传输至板卡，置于/home/edgeboard/yolov3-python/model文件夹
- model目录下修改config.json配置
```json
{
    "mode": "professional",
    "model_dir": "./model", 
    "model_file": "model"
}
```

    - mode: 固定为"professional"
    - model_dir：传输至板卡的模型文件(model.json、model.nb、model.onnx、model.po)的目录
    - model_file: 传输至板卡的四个模型文件的文件名，固定为model

- 运行推理脚本
    
    确保当前位于/home/edgeboard/yolov3-python目录下：

    ```shell
    sudo python3 tools/infer_demo.py \
    --config ./model/config.json \
    --infer_yml ./model/infer_cfg.yml \
    --test_image ./test_images/000000025560.jpg \
    --visualize \
    --with_profile
    ```

    命令行选项参数如下：

        - config: 上文建立的config.json的路径
        --infer_yml: 模型导出时生成的infer_cfg.yml文件
        - test_image: 测试图片路径
        - visualize: 是否可视化，若设置则会在该路径下生成vis.jpg渲染结果，默认不生成
        - with_profile: 是否推理耗时，若设置会输出包含前处理、模型推理和后处理的总耗时，默认不输出

## 3.3.2 paddle、ppnc对比测试
- 模型文件  
    确保aistudio上导出的paddle静态图模型文件(xx.pdmodel)和(xx.pdiparams)已经传输至板卡，置于/home/edgeboard/yolov3-python/model目录下。
- 执行测试  
    确保当前位于/home/edgeboard/yolov3目录下：

    ```shell
    sudo python3 tools/test.py \
    --config  ./model/config.json  \
    --infer_yml ./model/infer_cfg.yml \
    --model_dir ./model \
    --test_dir ./test_images \
    --output_dir ./output_dir \
    ```

    各命令行选项参数如下：   
        - config: 同上  
        - infer_yml: 同上  
        - model_dir: paddle静态图模型文件(model.pdmodel)和(model.pdiparams)所在目录  
        - test_dir: 测试图片文件夹路径  
        - output_dir: 存放结果文件，默认为"./output_dir"，该路径下会生成paddle_result_pickle、paddle_result_images、ppnc_result_pickle、ppnc_result_images目录，分别存放paddle和ppnc的pickle格式结果和可视化的结果数据。


