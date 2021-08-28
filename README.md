![VERSION](https://img.shields.io/pypi/pyversions/torch)
# SentencesJudger
SentencesJudger 是一个基于GRU神经网络的句子判断程序，基本的功能是判断文章中的某一句话是否为一个优美的句子。  
SentencesJudger is a program based on GRU network. Its basic function is to judge whether a sentence in the article is a beautiful sentence.
- - -
## 使用/Using
* [pyTorch](https://github.com/pytorch/pytorch)
* [LTP](https://github.com/HIT-SCIR/ltp)

## 如何使用SentencesJudger?/How to use SentencesJudger?
1. 确认Python运行环境 **/**  Make sure the Python runtime environment
3. 安装pyTorch与LTP **/** Install pyTorch and ltp
```
pip install torch
pip install ltp
```
3. 运行·main.p·y **/** Run ·main.py·

## 效果/effect
* 在训练数据中准确率为96.7% **/** Accuracy 96.7% in train-data
* 在测试数据中准确率为83.5% **/** Accuracy 83.5% in test-data

## 注意事项/Points for attention
* 确保模型的加载路径 **/** Make sure the model loading path
* 在首次运行时连接网络 **/** Connect to the Internet in first running
* 使用UTF-8编码 **/** Use UTF-8 encoding
* 内置模型仅支持中文 **/** The built-in model only supports Chinese

## 训练模型/Training the model
在`train.py`中选择训练样本、测试样本与输出模型路径，运行`train.py`即可。训练样本、测试样本为csv文件，格式可参考`sample.csv`。  
Choice the path of train-data,test-data and model-output in `train.py`, then run the `train.py`. train_data and test_data are csv file, the format of them can reference `sample.csv`.

## 其他/Others
目前模型因样本数量过少，判断效果并不理想，持续更新中。  
Because of the train-data is too few, the judgment effect is not ideal.Continuous updating.
