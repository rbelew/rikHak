''' captum-numelBug (nee overrule)
Created 23 Sept 24

@author: rik
'''

import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Union

import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import datasets

import string
import re
import sys
import time

import socket
HOST = socket.gethostname()

import torch
from torch.nn import functional as F

import transformers
from transformers import (
	AutoConfig,
	AutoModelForSequenceClassification,
	AutoTokenizer,
	EvalPrediction,
	HfArgumentParser,
	PretrainedConfig,
	Trainer,
	TrainingArguments,
	default_data_collator,
	set_seed,
)
from transformers.trainer_utils import is_main_process

from transformers_interpret import SequenceClassificationExplainer

import captum
from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients, TokenReferenceBase

@dataclass
class ModelArguments:
	"""
	Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
	"""

	model_name_or_path: str = field(
		metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
	)
	config_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
	)
	tokenizer_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
	)
	cache_dir: Optional[str] = field(
		default=None,
		metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
	)
	use_fast_tokenizer: bool = field(
		default=True,
		metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
	)

task_to_keys = {
	"cola": ("sentence", None),
	"mnli": ("premise", "hypothesis"),
	"mrpc": ("sentence1", "sentence2"),
	"qnli": ("question", "sentence"),
	"qqp": ("question1", "question2"),
	"rte": ("sentence1", "sentence2"),
	"sst2": ("sentence", None),
	"stsb": ("sentence1", "sentence2"),
	"wnli": ("sentence1", "sentence2"),
}
@dataclass
class DataTrainingArguments:
	"""
	Arguments pertaining to what data we are going to input our model for training and eval.
	"""

	task_name: Optional[str] = field(
		default=None,
		metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
	)
	max_seq_length: int = field(
		default=128,
		metadata={
			"help": "The maximum total input sequence length after tokenization. Sequences longer "
			"than this will be truncated, sequences shorter will be padded."
		},
	)
	overwrite_cache: bool = field(
		default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
	)
	pad_to_max_length: bool = field(
		default=True,
		metadata={
			"help": "Whether to pad all samples to `max_seq_length`. "
			"If False, will pad the samples dynamically when batching to the maximum length in the batch."
		},
	)
	train_file: Optional[str] = field(
		default=None, metadata={"help": "A csv or a json file containing the training data."}
	)
	validation_file: Optional[str] = field(
		default=None, metadata={"help": "A csv or a json file containing the validation data."}
	)

	def __post_init__(self):
		if self.task_name is not None:
			self.task_name = self.task_name.lower()
			if self.task_name not in task_to_keys.keys():
				raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
		elif self.train_file is None or self.validation_file is None:
			raise ValueError("Need either a GLUE task or a training/validation file.")
		else:
			extension = self.train_file.split(".")[-1]
			assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
			extension = self.validation_file.split(".")[-1]
			assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

Label_name = 'label'

# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
def count_TrainParam(model: torch.nn.Module) -> int:
    """ Returns the number of learnable parameters for a PyTorch model """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_AllParam(model: torch.nn.Module) -> int:
    """ Returns the number of learnable parameters for a PyTorch model """
    return sum(p.numel() for p in model.parameters())

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

def pred_classif(model,input,ttype,attn):

	out = model(input) # ,ttype,attn)

	logitsCPU = out.logits.cpu().detach()
	logits = logitsCPU.numpy()[0]
	
	lbl1wins = logits[0] < logits[1]
	corrLbl = 1 if lbl1wins else 0
	
	prob = torch.softmax(logitsCPU,1)
	
	return (corrLbl,logits,prob)
	
def captumExample(model,tokenizer,eval_dataset,sampleIdx):
	
	ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
	sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
	cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence
								
	evalEG0 = eval_dataset[sampleIdx]

	# accumalate couple samples in this array for visualization purposes
	vis_data_records_ig = []

				
	input_ids =evalEG0['input_ids']
	token_type_ids = evalEG0['token_type_ids']
	attention_mask = evalEG0['attention_mask']		
	corrLbl = evalEG0[Label_name]

	sepIdx = input_ids.index(sep_token_id)
	prompt = tokenizer.decode(input_ids[:sepIdx])
	print(f'captumExample: {sampleIdx=}  {sepIdx=} {prompt=}')
	
	token_reference = TokenReferenceBase(reference_token_idx=PAD_IDX)
	
	lig = LayerIntegratedGradients(model, model.bert.embeddings)

	in_tensor = torch.tensor( [ input_ids ] ,device=DEVICE)
	ttype_tensor = torch.tensor( [ token_type_ids ] ,device=DEVICE)
	attn_tensor = torch.tensor( [ attention_mask ] ,device=DEVICE)
	 
	indices = in_tensor[0].detach().tolist()

	all_tokens = tokenizer.convert_ids_to_tokens(indices)
	
	seq_length = len(indices)

	model.zero_grad()
	
	pred,logits,prob = pred_classif(model,in_tensor,ttype_tensor,attn_tensor)
	pred_int = round(pred)
	
	print(f'captumExample: {sampleIdx=} {pred=} {logits=} {prob=} ')
	
	reference_indices = token_reference.generate_reference(seq_length, device=device).unsqueeze(0)
	
	attributions_ig, delta = lig.attribute(in_tensor, reference_indices, \
										additional_forward_args = (ttype_tensor,attn_tensor), \
                                        n_steps=500, return_convergence_delta=True)

	print(f'captumExample: {sampleIdx=} {corrLbl=} delta: {abs(delta)}')

	# add_attributions_to_visualizer(attributions_ig, prompt, pred, pred_int, corrLbl, delta, vis_data_records_ig)	

def main():
	global DEVICE
	global device 
	global CaptumPlotDir

	if HOST.startswith('wayne'):
		device= 'mps'
			  
	print(f'HOST={HOST} DEVICE={device}')
	DEVICE = device
	
	for package in (torch, transformers,captum):
		print(package.__name__, package.__version__)

	OverruleDataDir = '/Users/rik/.cache/overrule/'
	
	# ModelPath = 'casehold/legalbert'
	# CacheModelPath = '/Users/rik/.cache/huggingface/hub/models--casehold--legalbert/'
	
 # ModelPath = 'nlpaueb/legal-bert-small-uncased'
 # CacheModelPath = '/Users/rik/.cache/huggingface/models/nlpaueb/legal-bert-small-uncased/'

	ModelPath = 'prajjwal1/bert-tiny'
	CacheModelPath = '/Users/rik/.cache/huggingface/models/prajjwal1/bert-tiny/'
	print(f'{ModelPath=}')
	
	OverruleOutDir = '/Users/rik/data/ai4law/overrule/'
	
	CaptumPlotDir = '/Users/rik/data/ai4law/captum/overrule/'
		
	## FT=FineTune base model
	paramListFT = []
	paramListFT.append('--model_name_or_path');  paramListFT.append(CacheModelPath)
	# paramListFT.append('--validation_file');     paramListFT.append(f'{OverruleDataDir}overruling.csv')
	paramListFT.append('--validation_file');     paramListFT.append(f'{OverruleDataDir}test.csv')
	paramListFT.append('--train_file');          paramListFT.append(f'{OverruleDataDir}train.csv')
	paramListFT.append('--max_seq_length');      paramListFT.append('128')
	paramListFT.append('--output_dir');          paramListFT.append(OverruleOutDir)
	# paramListFT.append('--do_train')
	paramListFT.append('--do_eval')
	paramListFT.append('--evaluation_strategy'); paramListFT.append('steps')
	paramListFT.append('--max_seq_length');      paramListFT.append('128')
	paramListFT.append('--per_device_train_batch_size=16')
	paramListFT.append('--learning_rate=1e-5')
	paramListFT.append('--num_train_epochs=2.0')
	paramListFT.append('--overwrite_output_dir=True')
	paramListFT.append('--logging_steps');       paramListFT.append('50')
	
	paramList = paramListFT
	
	parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
	# NB: ptl argument must be added after parser created
	parser.add_argument("--ptl", type=bool, default=False)

	model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses(args=paramList)

	# Setup logging
	logger = logging.getLogger(__name__)
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
	)

	# Log on each process the small summary:
	logger.warning(
		f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
		+ f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
	)
	# Set the verbosity to info of the Transformers logger (on main process only):
	if is_main_process(training_args.local_rank):
		transformers.utils.logging.set_verbosity_info()
		transformers.utils.logging.enable_default_handler()
		transformers.utils.logging.enable_explicit_format()
	logger.info(f"Training/evaluation parameters {training_args}")

	# Set seed before initializing model.
	set_seed(training_args.seed)

	ds = datasets.load_dataset("csv", data_files={"train": data_args.train_file, "validation": data_args.validation_file})
	
	# Overruling is a binary classification task
	is_regression = ds["train"].features["label"].dtype in ["float32", "float64"]
	
	label_list = ds["train"].unique("label")
	label_list.sort()  # Sort for deterministic ordering
	num_labels = len(label_list)

	# Load pretrained model and tokenizer
	config = AutoConfig.from_pretrained(
		model_args.config_name if model_args.config_name else model_args.model_name_or_path,
		num_labels=num_labels,
		finetuning_task=data_args.task_name,
		cache_dir=model_args.cache_dir,
	)
	tokenizer = AutoTokenizer.from_pretrained(
		model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
		cache_dir=model_args.cache_dir,
		# Defaults to using fast tokenizer
		use_fast=model_args.use_fast_tokenizer,
		# 240921
		# cf. https://github.com/pytorch/pytorch/issues/121113
		# padding='max_length'
	)
	model = AutoModelForSequenceClassification.from_pretrained(
		model_args.model_name_or_path,
		from_tf=bool(".ckpt" in model_args.model_name_or_path),
		config=config,
		cache_dir=model_args.cache_dir,
		local_files_only=True,
	)
	model.to(device)
	global PAD_IDX
	global CLS_IDX
	global SEP_IDX
	PAD_IDX = tokenizer.pad_token_id
	CLS_IDX = tokenizer.cls_token_id
	SEP_IDX = tokenizer.sep_token_id
	
	print(f'model NParam={float(count_AllParam(model)):.3e} NTrainableParam={float(count_TrainParam(model)):.3e} ')

	# Preprocess dataset
	non_label_column_names = [name for name in ds["train"].column_names if name != "label"]
	if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
		sentence1_key, sentence2_key = "sentence1", "sentence2"
	else:
		if len(non_label_column_names) >= 2:
			sentence1_key, sentence2_key = non_label_column_names[:2]
		else:
			sentence1_key, sentence2_key = non_label_column_names[0], None

	# Padding strategy
	if data_args.pad_to_max_length:
		padding = "max_length"
		max_length = data_args.max_seq_length
	else:
		# Pad dynamically at batch creation, to the max sequence length in each batch
		padding = False
		max_length = None

	# Some models have set the order of the labels to use, so set the specified order here
	label_to_id = None
	if (
		model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
		and data_args.task_name is not None
		and is_regression
	):
		# Some have all caps in their config, some don't.
		label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
		if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
			label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
		else:
			logger.warn(
				"Your model seems to have been trained with labels, but they don't match the dataset: ",
				f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
				"\nIgnoring the model labels as a result.",
			)
	elif data_args.task_name is None:
		label_to_id = {v: i for i, v in enumerate(label_list)}

	def preprocess_function(examples):
		# Tokenize the texts
		args = (
			(examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
		)
		result = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)

		# Map labels to IDs (not necessary for GLUE tasks)
		if label_to_id is not None and "label" in examples:
			result["label"] = [label_to_id[l] for l in examples["label"]]
		return result

	ds = ds.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)
	train_dataset = ds["train"]
		
	eval_dataset = ds["validation_matched" if data_args.task_name == "mnli" else "validation"]
	# Get the corresponding test set for GLUE task
	if data_args.task_name is not None:
		test_dataset = ds["test_matched" if data_args.task_name == "mnli" else "test"]

	# Log a few random samples from the training set:
	for index in random.sample(range(len(train_dataset)), 3):
		logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

	# Get the corresponding metric function for GLUE task
	if data_args.task_name is not None:
		metric = datasets.load_metric("glue", data_args.task_name)

	# Define custom compute_metrics function, returns F1 metric for Overruling and ToS binary classification tasks
	def compute_metrics(p: EvalPrediction):
		preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
		preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
		metric = datasets.load_metric("f1")
		# Compute F1 for binary classification task
		f1 = metric.compute(predictions=preds, references=p.label_ids)
		return f1

	# Initialize our Trainer
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset if training_args.do_eval else None,
		compute_metrics=compute_metrics,
		tokenizer=tokenizer,
		# Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
		data_collator=default_data_collator if data_args.pad_to_max_length else None,
	)

	eval_results = {}

	# Evaluation on eval_dataset
	logger.info("*** Evaluate ***")

	captumExample(model,tokenizer,eval_dataset,0)
			
	print('here')

	eval_result = trainer.evaluate(eval_dataset=eval_dataset)

	output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
	if trainer.is_world_process_zero():
		with open(output_eval_file, "w") as writer:
			logger.info(f"***** Eval results *****")
			for key, value in eval_result.items():
				logger.info(f"  {key} = {value}")
				writer.write(f"{key} = {value}\n")

	eval_results.update(eval_result)
	
	return eval_results

if __name__ == '__main__':
	main()

