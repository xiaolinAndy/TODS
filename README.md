# TODS
The repository of our work "Topic-Oriented Dialogue Summarization". The paper is under reviewing.

## Instructions

### 1. Introduction

We propose a new dialogue summarization task named **Topic-Oriented Dialogue Summarization (TODS)**. Given a dialogue and a specific topic, the task aims to generate a summary that covers the main content related to the given topic. To achieve this task, we propose three topic-related auxiliary tasks for TODS, including **Topic Identification Task**, **Attention Restriction Task**, and **Topic Summary Distinguishing Task**.

We experiment on two datasets ([CSDS](https://github.com/xiaolinAndy/CSDS) and [DialogSum](https://github.com/cylnlp/dialogsum)) by modifying them to adapt for TODS, and two baseline methods (BART-base and BART-large).

### 2. Necessary Resources

- CSDS dataset: We put the processed CSDS-topic in [data/CSDS/topic/](https://github.com/xiaolinAndy/RODS/blob/main/data/CSDS/topic/).
- DialogSum dataset: We put the processed DialogSum-topic in [data/DialogSum/topic/](https://github.com/xiaolinAndy/RODS/blob/main/data/CSDS/topic/).
- Pretrained BART model: 
  - bart-base Chinese for CSDS, available at [here](https://huggingface.co/fnlp/bart-base-chinese).
  - bart-large Chinese for CSDS, available at [here](https://huggingface.co/fnlp/bart-large-chinese).
  - bart-base English for DialogSum , available at [here](https://huggingface.co/facebook/bart-base).
  - bart-large English for DialogSum , available at [here](https://huggingface.co/facebook/bart-large).

### 3. Usage

#### Requirements

- python == 3.8
- pytorch == 1.8.1
- files2rouge == 2.1.0
- jieba == 0.42.1
- numpy == 1.21.2
- cytoolz == 0.11.2
- nltk == 3.6.5
- bert-score == 0.3.10
- moverscore == 1.0.3
- transformers == 4.4.1

#### Instructions

1. Go to the *models/* directory.
2. Run the bash file *run_CSDS.sh* to train and test on CSDS-topic
3. Run the bash file *run_DialogSum.sh* to train and test on DIALOGSUM-topic

### 4. Acknowledgement

The reference codes of the provided methods come from:

- [CPT](https://github.com/fastnlp/CPT)

We thanks for all these researchers who have made their codes publicly available.

### 5. Citation

We will update the citation format after the paper is accepted.

If you have any issues, please contact with [haitao.lin@nlpr.ia.ac.cn](mailto:haitao.lin@nlpr.ia.ac.cn)
