# Machine_Translation

基于[iwslt2017](https://huggingface.co/datasets/iwslt2017/tree/main/data/2017-01-trnted/texts/zh/en)数据集，实现了从英文到中文的机器翻译

支持的模型：隐马尔科夫模型、最大熵模型、神经网络

支持的评估指标：BLEU

## 环境配置

```
pip install -r requirements.txt
```

## 仓库结构

```
root
- data（存放数据集）
  - zh-en
- evaluation（指标评测代码）
  -
- models（实现的模型代码）
  -
- preprocess（数据预处理代码）
  - dataset.py
  - demo.py

.gitignore
README.md
requirements.txt
...
```