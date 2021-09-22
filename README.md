![VERSION](https://img.shields.io/pypi/pyversions/torch)
# SentencesJudger
SentencesJudger 是一个基于GRU神经网络的句子判断程序，基本的功能是判断文章中的某一句话是否为一个优美的句子。  
- - -

[English](./README-EN.md)|[Demo](https://sentencesjudger.xyz)

## 如何使用SentencesJudger
1. 确认Python运行环境
2. 安装pyTorch与LTP
    ```bash
    python3 -m pip install -U torch ltp
    ```
1. 运行`main.py`
    ```bash
    python3 main.py
    ```

## 效果
| 数据集 | 准确度 |
| -- | -- |
| train | 96.7% |
| test | 83.5% |

## 注意事项
* 确保模型的加载路径
* 在首次运行时连接网络（以便加载LTP模型）
* 使用UTF-8编码
* 内置模型仅支持中文

## 训练模型
在`train.py`中选择训练样本、测试样本与输出模型路径，运行`train.py`即可。训练样本、测试样本为csv文件，格式可参考`sample.csv`。  

## 其他
目前模型因样本数量过少，判断效果并不理想，持续更新中。  

## 使用方案
* [pyTorch](https://github.com/pytorch/pytorch)
* [LTP](https://github.com/HIT-SCIR/ltp)
