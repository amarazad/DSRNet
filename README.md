# DSRNet 

This repository contains the dataset and source code for the following paper published in PAKDD 2021:

#### ["Meta-Context Transformers for Domain-Specific Response Generation"](https://www.springerprofessional.de/en/meta-context-transformers-for-domain-specific-response-generatio/19145648)

<p align="center">
<img src="https://github.com/suranjanas/internship-2020/blob/master/final_clean_up/img/model_arch.png" alt="model image" width="500" height="350">
</p>


This repository is based on hugginface transformer package and OpenAI GPT-2. Baseline scripts are adapted from [MultiTurnDialogueZoo](https://github.com/gmftbyGMFTBY/MultiTurnDialogZoo) for HRED & VHRED. The results indicate that DSRNet is able to generate natural language response given dialogue history, questions & topics naturally and adequately, even in a multi-party interlocutor space. It can be used to train an NLG model with very limited examples.

ArXiv paper: [https://arxiv.org/abs/2010.05572](https://arxiv.org/abs/2010.05572)

## Contents

+ [Setup](https://github.com/suranjanas/internship-2020/blob/master/final_clean_up/README.md#setup)
+ [Data Preparation](#data-preprocessing-and-format-generation)
+ [Pipeline](#pipeline)
  - [Domain-pretraining](#domain-pre-training--optional)
  - [Training](#training)
  - [Decoding](#decoding)
+ [Evaluation](#evaluation)
+ [Citation](#citation)

## Setup

Please use the below command to clone and install the requirements.

```bash
git clone <repo>
conda env create -f environment.yml
conda activate transformers
```

Download the following nltk packages in the virtual environment

```
python
import nltk
nltk.download('words')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
exit()
python -m spacy download en_core_web_sm
```

Fetch and unzip the mallet package

```bash
wget http://mallet.cs.umass.edu/dist/mallet-2.0.8.tar.gz
tar -xvf mallet-2.0.8.tar.gz
```

## Data Preprocessing and Format Generation

The [Ubuntu IRC](https://github.com/jkkummerfeld/irc-disentanglement) dataset contains 77,563 annotated messages of IRC. Almost all are from the [Ubuntu IRC Logs](https://irclogs.ubuntu.com/) for the `#ubuntu` channel.
A small set is a re-annotation of the `#linux` channel data from [Elsner and Charniak (2008)](https://www.asc.ohio-state.edu/elsner.14/resources/chat-manual.html).
The dataset is present in our repo in the `kummerfeld/data` folder.

**Data Download**

Download the dataset here ([zip](https://github.com/jkkummerfeld/irc-disentanglement/zipball/master) or [tar](https://github.com/jkkummerfeld/irc-disentanglement/tarball/master)) and unzip.
Create the following directories:

```
mkdir kummerfeld
cd kummerfeld\
mkdir data
cd ..
cd ..
```

Copy only the train, dev and test folders from the downloaded folder to `kummerfeld/data`:

`cp -r <path-of-downloaded-folder>/data/<train, dev, or test>/ kummerfeld/data/`

**Conversation Mining**

+ To extract conversations and use them as context : <code>python extract_conversation.py --turn_len=3 --add_qstn=False --sliding_win=True</code>
+ To have consecutive utterances as context :  modify `--sliding_win=False` in above command.

**Argument Description**

```
--turn_len : To adjust the no. of turns in the dialogue history; default=3
--add_qstn : Adds query from the dialogue history to the context; default=False
--sliding_win : Allows processing the dialogue history in a sliding window fashion; default=True
```

**Adding Query:**

+ To generate a question detector : <code>python question_detect.py</code>
+ To add query to the context along with conversation : <code>python extract_conversation.py --turn_len=3 --add_qstn=True --sliding_win=True</code>


**Adding Topic:**

+ To add topic words to the context : 

```
python extract_conversation.py
python preprocessing.py  #need only run once
python topic_modeling.py --add_qstn=False
```

+ To add query and topic words to the context, run `topic_modeling.py` with  `add_qstn=True`.


**Adding Entity:**

We are unable to release the original code for domain-specific entity extraction for copyright issues. However, we have provided the code used to inject the entity words into the context by replacing the entity extraction module with that of [Spacy](https://spacy.io/). Please note, that this will not reproduce the scores mentioned with entities in the paper. For datasets/use-cases with domain-specific vocabulary, it is highly recommended to use a domain specific entity extraction mechanism.

To inject your domain-specific entity extraction code to our entity data formatting code, refer to lines 28 and 144 in [entity_extraction.py](https://github.com/suranjanas/internship-2020/blob/master/final_clean_up/entity_extraction.py).

For natural language datasets, this code can be used directly.

To recreate the domain-specific entity extraction code we used for the technical domain, the steps in [Mohapatra et. al.](https://link.springer.com/chapter/10.1007/978-3-030-03596-9_35) should be followed.

+ To add entity words to the context : 

```
python extract_conversation.py
python entity_extraction.py --add_qstn=False
```

+ To add query and entity words to the context, run `entity_extraction.py` with  `add_qstn=True`.


**Data files generated include:** 

<code>kummerfeld/ctxt-train-{task}.txt</code>: training set in txt format separated by special tokens.
<code>kummerfeld/ctxt-dev-{task}.txt</code>: development set in txt format separated by special tokens.
<code>kummerfeld/ctxt-test-{task}.txt</code>: testing set in txt format separated by special tokens.

- task can be topic/qstn/qstn-topic/None

<code>kummerfeld/data/{mode}</code>: contains the raw irc dataset files

- mode can be train/dev/test

**Data format**

```
Line 1 : I went to propietary drivers i have both selected [eos] drm driver for Intel GMA500 [eos] drm driver for Intel GMA500 [eoc] driver, nvidia, card, explain, boot, adjust, drive, domain, window, machin [eot] [sep] and Intel Cedarview graphics driver [eos]
Line 2 : drm driver for Intel GMA500 drm driver for Intel GMA500 [eos] and Intel Cedarview graphics driver [eos] Only one should be activated [eoc] driver, nvidia, card, explain, boot, adjust, drive, domain, window, machin [eot] [sep] Only one should be activated [eos]

[eos] : indicates end-of-turn
[eoc] : indicates end-of-context
[eot] : indicates end-of-topic
[eoq] : indicates end-of-query
[sep] : separates context from ground truth response 
```

## Pipeline

#### Domain pre-training  **(optional)**

While the model can run without this step, to enhance the accuracy, this step can be adopted.
We fine-tuned the GPT-2 language model using this [code](https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py) on a subset of the [Ubuntu 2.0 dataset](https://github.com/rkadlec/ubuntu-ranking-dataset-creator). The fine-tuned model is then used in the following training step.


#### Training

```bash
export CUDA_VISIBLE_DEVICES=0
python train.py --output_dir=MODEL_SAVE_PATH --model_type=gpt2 --model_name_or_path=PRE_TRINED_MODEL_PATH --do_train --do_eval --eval_data_file=kummerfeld/ctxt-dev.txt --per_gpu_train_batch_size 16 --num_train_epochs EPOCH --learning_rate LR --overwrite_cache --use_tokenize --train_data_file=kummerfeld/ctxt-train.txt --overwrite_output_dir
```

<code>MODEL_SAVE_PATH </code>: Path for the saving model .

<code>PRE_TRAINED_MODEL_PATH </code>: Initial checkpoint; Could start from gpt2, gpt2-medium or domain-pretrained model.

<code>EPOCH </code>: Number of training epochs;  5 is enough for a reasonable performance

<code>LR </code>: Learning rate; 5e-5, 1e-5, or 1e-4

To visualize the train and evaluation loss curves, 

- you can manually check in the train_results.txt and eval_results.txt in MODEL_SAVE_PATH
- or, run <code>python plot.py --dir MODEL_SAVE_PATH</code> and look at the `loss.png` generated in MODEL_SAVE_PATH.

#### Decoding

```bash
mkdir output
export CUDA_VISIBLE_DEVICES=0
python generate.py --model_path=CHECKPOINT --test_file 'kummerfeld/ctxt-test.txt' --generate_path 'output/gen.txt' --true_path 'output/true.txt' --json_path 'output/output.json'
```

Add path to your model checkpoint in `CHECKPOINT`

Refer to the `output.json` file for a neat representation of context, ground truth and generated utterances.


## Evaluation

install nlg-eval following the instructions in [NLG-EVAL](https://github.com/Maluuba/nlg-eval) (It is better to follow their custom setup.)

```
nlg-eval --hypothesis=output/gen.txt --references=output/true.txt > output/nlg_eval.txt
```

For [Bert Score](https://github.com/Tiiiger/bert_score):

```
pip install bert-score
bert-score -r output/true.txt -c output/gen.txt --lang en
```

## Citation

If you use this code and data in your research, please cite our paper:

```
@misc{kar2020metacontext,
      title={Meta-Context Transformers for Domain-Specific Response Generation}, 
      author={Debanjana Kar and Suranjana Samanta and Amar Prakash Azad},
      year={2020},
      eprint={2010.05572},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

