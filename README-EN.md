![VERSION](https://img.shields.io/pypi/pyversions/torch)

# SentencesJudger
SentencesJudger is a program based on GRU network. Its basic function is to judge whether a sentence in the article is a beautiful sentence.
- - -

[中文](./README.md)|[Website](https://sentencesjudger.xyz)

## How to use SentencesJudger?
1. Check the Python runtime environment
2. Install pyTorch, jieba and paddlepaddle(for the part of speech tagging module of jieba)
    ```bash
    python3 -m pip install -U torch jieba paddlepaddle
    ```
3. Run `main.py`

## Peformance
| dataset | Accuracy |
| -- | -- |
| train | 96.7% |
| test | 83.5% |

## Notice
* Make sure the model loading path are correct
* Use UTF-8 encoding
* The built-in model only supports Chinese

## Train the model
Choice the path of train-dataset, test-dataset and model-output in `train.py`, then run the `train.py`. train_data and test_data are csv file, the format of them can reference `sample.csv`.

## Others 
Because of the train-data is too few, the judgment effect is not ideal.Continuous updating.

## Powered by
* [pyTorch](https://github.com/pytorch/pytorch)
* [jieba](https://github.com/fxsjy/jieba)
