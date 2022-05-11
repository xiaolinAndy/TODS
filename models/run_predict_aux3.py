import argparse
import json
import logging
import os
import random
import sys
import nltk

import numpy as np
import torch
import transformers
from transformers import (AutoConfig, AutoModel, BertTokenizer, BertForTokenClassification,
                          DataCollatorForTokenClassification, HfArgumentParser, DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, Trainer, TrainerCallback, AutoModelForSeq2SeqLM, BartTokenizer)
from transformers.trainer_utils import is_main_process
from datasets import load_metric, Dataset
from utils_att import DataTrainingArguments, ModelArguments, load_json

import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from modeling_cpt import CPTModel, CPTForConditionalGeneration
from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline
from evaluate import get_rouge, get_topic_acc
from MultiAttentionModClass import DataCollatorForMultiTaskSeq2Seq, MultiTaskSeq2SeqTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default='/path/to/model', type=str)
parser.add_argument("--dataset", default="lcsts", type=str)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--batch_size", default='4', type=str)
parser.add_argument("--epoch", default='10', type=str)
parser.add_argument("--data_dir", default="/path/to/dataset/", type=str)
parser.add_argument("--gradient_accumulation_steps", default="2", type=str)
parser.add_argument("--output_dir",default="CSDS",type=str)
parser.add_argument("--sum_mode",default="final",type=str)
parser.add_argument("--max_source_length",default="512",type=str)
args = parser.parse_args()
arg_dict = args.__dict__
print(args)

logger = logging.getLogger(__name__)

dataset_name = arg_dict['dataset']
outdir_1 = 'output'
if not os.path.exists(outdir_1):
    os.mkdir(outdir_1)

outdir = outdir_1 + '/' + args.output_dir
if not os.path.exists(outdir):
    os.mkdir(outdir)

seed = 2021
# outdir=outdir+'/'+str(seed)
length_map={'CSDS':'200',
            'DialogSum': '200'}

args = [
    '--model_name_or_path', arg_dict['model_path'],
    '--do_predict',
    '--test_file', os.path.join(arg_dict['data_dir'], 'test.json'),
    '--output_dir', outdir,
    '--per_device_train_batch_size', arg_dict['batch_size'],
    '--per_device_eval_batch_size', arg_dict['batch_size'],
    '--overwrite_output_dir',
    '--max_source_length', arg_dict['max_source_length'],
    '--val_max_target_length=' + length_map[arg_dict['dataset']],
    '--predict_with_generate=1',
    '--seed', str(1000 * seed),
    '--num_train_epochs', arg_dict['epoch'],
    '--save_strategy', 'epoch',
    '--evaluation_strategy', 'epoch',
    '--learning_rate', str(arg_dict['lr']),
    '--gradient_accumulation_steps', arg_dict['gradient_accumulation_steps'],
    '--sum_mode', arg_dict['sum_mode']
]
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses(args)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(training_args.seed)

datasets = {}
data_files = {}
if data_args.train_file is not None:
    data_files["train"] = data_args.train_file
if data_args.validation_file is not None:
    data_files["validation"] = data_args.validation_file
if data_args.test_file is not None:
    data_files["test"] = data_args.test_file
for key in data_files:
    print(key)
    datasets[key] = load_json(data_files[key])

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

# Log on each process the small summary:
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
# Set the verbosity to info of the Transformers logger (on main process only):
if is_main_process(training_args.local_rank):
    transformers.utils.logging.set_verbosity_info()
logger.info("Training/evaluation parameters %s", training_args)

if arg_dict['dataset'] == 'CSDS':
    tokenizer=BertTokenizer.from_pretrained(model_args.model_name_or_path)
else:
    tokenizer = BartTokenizer.from_pretrained(model_args.model_name_or_path)
if 'bart' in model_args.model_name_or_path:
    model = BartForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
else:
    model = CPTForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
model.config.max_length = data_args.val_max_target_length
model.config.no_repeat_ngram_size = -1

column_names = datasets["test"].column_names
max_target_length = data_args.val_max_target_length
padding = False


def preprocess_function(examples):
    inputs = examples['dialogue']
    topics = examples['topic']
    targets = examples['sum']
    att_min = examples['att_min']
    att_max = examples['att_max']
    model_inputs = {'input_ids_t2s': [], 'attention_mask_t2s': [], 'att_strict': [],
                    'input_ids_s2t': [], 'attention_mask_s2t': []}
    inputs_tokenized = []
    # topic2sum
    for i in range(len(inputs)):
        input_tmp = {'input_ids': [], 'attention_mask': [], 'att_strict': []}
        input_res = tokenizer(inputs[i], max_length=data_args.max_source_length, padding=padding, truncation=True)
        input_tmp['input_ids'].append(tokenizer._convert_token_to_id('[CLS]'))
        input_tmp['attention_mask'].append(1)
        input_tmp['att_strict'].append(0)
        for j in range(len(input_res['input_ids'])):
            input_tmp['input_ids'] += input_res['input_ids'][j][1:]
            input_tmp['attention_mask'] += input_res['attention_mask'][j][1:]
            if j >= att_min[i] and j <= att_max[i]:
                input_tmp['att_strict'] += [1] * len(input_res['input_ids'][j][1:])
            else:
                input_tmp['att_strict'] += [0] * len(input_res['input_ids'][j][1:])
        # input_tmp['input_ids'] = input_tmp['input_ids'][:data_args.max_source_length]
        # input_tmp['attention_mask'] = input_tmp['attention_mask'][:data_args.max_source_length]
        # input_tmp['att_strict'] = input_tmp['att_strict'][:data_args.max_source_length]
        inputs_tokenized.append(input_tmp)


    topics = tokenizer(topics, max_length=max_target_length, padding=padding, truncation=True)

    for i in range(len(inputs)):
        remain_length = data_args.max_source_length - len(topics['input_ids'][i])
        input_ids = inputs_tokenized[i]['input_ids'][:remain_length]
        input_ids[-1] = tokenizer._convert_token_to_id('[SEP]')
        model_inputs['input_ids_t2s'].append(input_ids
                                         + topics['input_ids'][i])
        model_inputs['attention_mask_t2s'].append([1] * len(model_inputs['input_ids_t2s'][-1]))
        model_inputs['att_strict'].append(inputs_tokenized[i]['att_strict'][:remain_length] + [1] * len(topics['input_ids'][i]))
        assert len(model_inputs['attention_mask_t2s'][-1]) == len(model_inputs['att_strict'][-1])

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

    model_inputs["labels_t2s"] = labels["input_ids"]

    # dial2topic
    topics = examples['all_topic']
    targets = examples['sum']
    targets = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)
    for i in range(len(inputs)):
        remain_length = data_args.max_source_length
        input_ids = inputs_tokenized[i]['input_ids'][:remain_length]
        input_ids[-1] = tokenizer._convert_token_to_id('[SEP]')
        model_inputs['input_ids_s2t'].append(input_ids)
        model_inputs['attention_mask_s2t'].append([1] * len(model_inputs['input_ids_s2t'][-1]))

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(topics, max_length=max_target_length, padding=padding, truncation=True)

    model_inputs["labels_s2t"] = labels["input_ids"]
    return model_inputs


if training_args.do_train:
    train_dataset = datasets["train"]
    if "train" not in datasets:
        raise ValueError("--do_train requires a train dataset")
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

if training_args.do_eval:
    max_target_length = data_args.val_max_target_length
    if "validation" not in datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = datasets["validation"]
    if data_args.max_val_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

if training_args.do_predict:
    max_target_length = data_args.val_max_target_length
    if "test" not in datasets:
        raise ValueError("--do_predict requires a test dataset")
    test_dataset = datasets["test"]
    if data_args.max_test_samples is not None:
        test_dataset = test_dataset.select(range(data_args.max_test_samples))
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

# max_eval_num = 30000
# if len(eval_dataset) > max_eval_num:
#     eval_dataset = Dataset.from_dict(eval_dataset[:max_eval_num])
# print(len(eval_dataset))
# test_dataset = Dataset.from_dict(test_dataset[:20])

# Data collator
label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
data_collator = DataCollatorForMultiTaskSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8 if training_args.fp16 else None,
    input_pad_token_id=tokenizer.pad_token_id,
)

# Metric
from rouge import Rouge

rouge = Rouge()


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # # rougeLSum expects newline after each sentence
    # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    # labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    while '' in preds:
        idx = preds.index('')
        preds[idx] = '。'

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if data_args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    scores = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
    for key in scores:
        scores[key] = scores[key]['f'] * 100

    result = scores

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


class TestCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
        metrics['epoch'] = state.epoch
        state.log_history.append(metrics)

training_args.remove_unused_columns = False
training_args.label_names = ['labels_t2s', 'labels_s2t']
# Initialize our Trainer
trainer = MultiTaskSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    callbacks=[TestCallback],
)

# Predicting
predictions_sum, predictions_topic, labels, metrics, _ = trainer.predict(test_dataset, metric_key_prefix="predict", max_length=200, num_beams=5)
test_preds = tokenizer.batch_decode(
    predictions_sum, skip_special_tokens=True,
)
test_preds = [pred.strip() for pred in test_preds]
output_test_preds_file = os.path.join(training_args.output_dir, "test_generations.txt")
with open(output_test_preds_file, "w", encoding='UTF-8') as writer:
    writer.write("\n".join(test_preds))

test_preds = tokenizer.batch_decode(
    predictions_topic, skip_special_tokens=True,
)
test_preds = [pred.strip() for pred in test_preds]
output_topic_preds_file = os.path.join(training_args.output_dir, "test_generations_topic.txt")
with open(output_topic_preds_file, "w", encoding='UTF-8') as writer:
    writer.write("\n".join(test_preds))

def get_ref_file(json_file, ref_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    sums = []
    for sample in data:
        sums.append(sample['QASumm'])
    with open(ref_file, 'w') as f:
        for sum in sums:
            f.write(sum+'\n')

def get_topic_file(json_file, ref_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    sums = []
    for sample in data:
        sums.append(sample['Topic'])
    with open(ref_file, 'w') as f:
        for sum in sums:
            f.write(sum+'\n')


if arg_dict['dataset'] == 'CSDS':
    ref_file = 'CSDS_data/' + data_args.sum_mode + '_aspect_refs_test.txt'
    #get_ref_file('CSDS_data/processed_aspect/test.json', ref_file)
    get_rouge(output_test_preds_file, ref_file)

else:
    ref_file = 'DialogSum/test_refs.txt'
    #get_ref_file('DialogSum/test.json', ref_file)
    os.system('files2rouge DialogSum/test_refs.txt %s -s rouge.txt -e .' % output_test_preds_file)

