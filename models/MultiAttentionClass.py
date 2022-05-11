import pdb

import numpy as np
import torch
import transformers
import torch.nn as nn
import time
import collections
import json
import random
import math

from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from torch.nn.utils.rnn import pad_sequence
from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from dataclasses import dataclass
from torch.nn import CrossEntropyLoss
from transformers import (AutoConfig, AutoModel, BertTokenizer,BertForTokenClassification,
                          DataCollatorForTokenClassification, HfArgumentParser,DataCollatorForSeq2Seq,Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, Trainer, TrainerCallback,AutoModelForSeq2SeqLM, BartConfig)
from transformers.tokenization_utils_base import BatchEncoding
from torch.utils.data.dataloader import DataLoader
from datasets import load_metric,Dataset
from transformers.utils import logging
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer
from transformers.generation_utils import LogitsProcessorList, StoppingCriteriaList
from transformers.generation_stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
import torch.nn.functional as F
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.file_utils import (
    WEIGHTS_NAME,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_distributed_available,
    is_torch_tpu_available,
    is_training_run_on_sagemaker,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    get_last_checkpoint,
    set_seed,
    speed_metrics,
)
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers import BartForConditionalGeneration
from transformers.models.bart.modeling_bart import (BartEncoder, BartDecoder, BartModel, shift_tokens_right, _expand_mask)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from transformers.generation_utils import (BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

logger = logging.get_logger(__name__)
BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]


@dataclass
class DataCollatorForMultiTaskSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    input_pad_token_id: int = 0

    def __call__(self, features):
        labels_t2s = [feature["labels_t2s"] for feature in features] if "labels_t2s" in features[0].keys() else None
        labels_s2t = [feature["labels_s2t"] for feature in features] if "labels_s2t" in features[0].keys() else None
        input_ids_t2s = [feature["input_ids_t2s"] for feature in features] if "input_ids_t2s" in features[0].keys() else None
        input_ids_s2t = [feature["input_ids_s2t"] for feature in features] if "input_ids_s2t" in features[0].keys() else None

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels_t2s is not None:
            max_label_length = max(max(len(l) for l in labels_t2s), max(len(l) for l in labels_s2t))
            max_input_length = max(max(len(l) for l in input_ids_t2s), max(len(l) for l in input_ids_s2t))
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder_t2s = [self.label_pad_token_id] * (max_label_length - len(feature["labels_t2s"]))
                feature["labels_t2s"] = (
                    feature["labels_t2s"] + remainder_t2s if padding_side == "right" else remainder + feature["labels_t2s"]
                )
                remainder_s2t = [self.label_pad_token_id] * (max_label_length - len(feature["labels_s2t"]))
                feature["labels_s2t"] = (
                    feature["labels_s2t"] + remainder_s2t if padding_side == "right" else remainder + feature[
                        "labels_s2t"]
                )
                # pad inputs
                remainder_t2s = [self.input_pad_token_id] * (max_input_length - len(feature["input_ids_t2s"]))
                feature["input_ids_t2s"] = (
                    feature["input_ids_t2s"] + remainder_t2s if padding_side == "right" else remainder + feature[
                        "input_ids_t2s"]
                )
                remainder_s2t = [self.input_pad_token_id] * (
                            max_input_length - len(feature["input_ids_s2t"]))
                feature["input_ids_s2t"] = (
                    feature["input_ids_s2t"] + remainder_s2t if padding_side == "right" else remainder + feature[
                        "input_ids_s2t"]
                )
                # pad attention mask
                feature["attention_mask_t2s"] = feature["attention_mask_t2s"] + remainder_t2s
                feature["attention_mask_s2t"] = feature["attention_mask_s2t"] + remainder_s2t
                # pad attention strict
                feature["att_strict"] = feature["att_strict"] + remainder_t2s

        padded_input = {}
        for feature in features:
            for key, value in feature.items():
                if key not in padded_input:
                    padded_input[key] = []
                padded_input[key].append(value)
        features = BatchEncoding(padded_input, tensor_type='pt')

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids_t2s = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels_t2s"])
            decoder_input_ids_s2t = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels_s2t"])
            features["decoder_input_ids_t2s"] = decoder_input_ids_t2s
            features["decoder_input_ids_s2t"] = decoder_input_ids_s2t

        return features

class MultiTaskSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels_t2s" in inputs:
            labels_t2s = inputs.pop("labels_t2s")
            labels_s2t = inputs.pop("labels_s2t")
        else:
            labels_t2s = None
            labels_s2t = None
        # print(inputs['input_ids_t2s'][0], inputs['input_ids_t2s'][1], inputs['input_ids_t2s'][2], inputs['input_ids_t2s'][3])
        # print(inputs['input_ids_t2s'].shape)
        # print(inputs['attention_mask_t2s'][0], inputs['attention_mask_t2s'][1], inputs['attention_mask_t2s'][2],
        #       inputs['attention_mask_t2s'][3])
        # print(inputs['attention_mask_t2s'].shape)
        # exit()
        outputs_t2s = model(input_ids=inputs['input_ids_t2s'],
                            decoder_input_ids=inputs['decoder_input_ids_t2s'],
                            attention_mask=inputs['attention_mask_t2s'],
                            labels=inputs['labels_t2s'],
                            output_attentions=True)
        outputs_s2t = model(input_ids=inputs['input_ids_s2t'],
                            decoder_input_ids=inputs['decoder_input_ids_s2t'],
                            attention_mask=inputs['attention_mask_s2t'],
                            labels=inputs['labels_s2t'])
        #print(outputs_t2s.cross_attentions[-1])  # (batch_size, num_heads, sequence_length, sequence_length)
        cross_att = torch.mean(torch.mean(outputs_t2s.cross_attentions[-1], dim=2), dim=1)  # (batch_size, sequence_length)
        cross_att = cross_att * (1 - inputs['att_strict'])
        # print(cross_att[0])
        # print(inputs['att_strict'][0])
        # print(torch.sum(cross_att, dim=-1))
        # exit()
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels_t2s is not None:
            loss_t2s = self.label_smoother(outputs_t2s, labels_t2s)
            loss_s2t = self.label_smoother(outputs_s2t, labels_s2t)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss_fct = CrossEntropyLoss(reduction='none')
            lm_logits_t2s = outputs_t2s['logits']  # (bs, seq_len, vocab_size)
            masked_lm_loss_t2s = loss_fct(lm_logits_t2s.view(-1, self.model.config.vocab_size), inputs['labels_t2s'].view(-1))
            loss_mask_t2s = torch.ne(inputs['labels_t2s'], -100)
            masked_lm_loss_t2s = torch.sum(masked_lm_loss_t2s.view(inputs['labels_t2s'].shape[0], -1),
                                           dim=-1) / torch.sum(loss_mask_t2s, dim=-1)  # (bs)

            lm_logits_s2t = outputs_s2t['logits']  # (bs, seq_len, vocab_size)
            masked_lm_loss_s2t = loss_fct(lm_logits_s2t.view(-1, self.model.config.vocab_size), inputs['labels_s2t'].view(-1))
            #print(masked_lm_loss_s2t)
            #loss_mask_s2t = torch.ne(inputs['labels_s2t'], -100)
            masked_lm_loss_s2t, _ = torch.max(masked_lm_loss_s2t.view(inputs['labels_s2t'].shape[0], -1), dim=-1)# / torch.sum(loss_mask_s2t, dim=-1)  # (bs)
            #weight_s2t = (F.softmax(masked_lm_loss_s2t) * inputs['labels_s2t'].shape[0]).detach()
            if inputs['labels_s2t'].shape[0] > 1:
                weight_s2t = ((1 - F.softmax(masked_lm_loss_s2t)) * inputs['labels_s2t'].shape[0] / (inputs['labels_s2t'].shape[0] - 1)).detach()
            else:
                weight_s2t = torch.ones(1).to(masked_lm_loss_t2s.device)
            #print(weight_s2t)
            #exit()
            if self.args.weight_mode == 's2t':
                loss_t2s = torch.mean(weight_s2t * masked_lm_loss_t2s)
            else:
                loss_t2s = outputs_t2s["loss"] if isinstance(outputs_t2s, dict) else outputs_t2s[0]
            loss_s2t = outputs_s2t["loss"] if isinstance(outputs_s2t, dict) else outputs_s2t[0]
            loss_att = torch.mean(torch.sum(cross_att, dim=-1))
        if self.args.weight_mode == 'bias' or self.args.weight_mode == 's2t':
            loss = loss_t2s + 0.2 * loss_s2t + loss_att
        else:
            loss = loss_t2s + loss_s2t + loss_att

        return (loss, (loss_t2s, loss_s2t, loss_att), (outputs_t2s, outputs_s2t)) if return_outputs else (loss, (loss_t2s, loss_s2t, loss_att))

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        if not self.args.predict_with_generate or prediction_loss_only:
            has_labels = all(inputs.get(k) is not None for k in self.label_names)
            inputs = self._prepare_inputs(inputs)
            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
                else:
                    ignore_keys = []

            # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
            if has_labels:
                labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
                if len(labels) == 1:
                    labels = labels[0]
            else:
                labels = None


            with torch.no_grad():
                if has_labels:
                    loss, losses, (outputs_t2s, outputs_s2t) = self.compute_loss(model, inputs, return_outputs=True)
                    losses = [s.mean().detach() for s in losses]
                    if isinstance(outputs_t2s, dict):
                        logits_t2s = tuple(v for k, v in outputs_t2s.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits_t2s = outputs_t2s[1:]
                    if isinstance(outputs_s2t, dict):
                        logits_s2t = tuple(v for k, v in outputs_t2s.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits_s2t = outputs_s2t[1:]
                else:
                    loss = None
                    if self.use_amp:
                        with autocast():
                            outputs = model(**inputs)
                    else:
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

            if prediction_loss_only:
                return (losses, None, None)

            logits = nested_detach(logits)
            if len(logits) == 1:
                logits = logits[0]

            return (losses, logits, labels)

        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
        }

        generated_tokens_t2s = self.model.generate(
            inputs["input_ids_t2s"],
            attention_mask=inputs["attention_mask_t2s"],
            **gen_kwargs,
        )
        generated_tokens_s2t = self.model.generate(
            inputs["input_ids_s2t"],
            attention_mask=inputs["attention_mask_s2t"],
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens_t2s.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens_t2s = self._pad_tensors_to_max_len(generated_tokens_t2s, gen_kwargs["max_length"])
        if generated_tokens_s2t.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens_s2t = self._pad_tensors_to_max_len(generated_tokens_s2t, gen_kwargs["max_length"])

        # with torch.no_grad():
        #     if self.use_amp:
        #         with autocast():
        #             outputs = model(**inputs)
        #     else:
        #         outputs = model(**inputs)
        #     if has_labels:
        #         if self.label_smoother is not None:
        #             loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
        #         else:
        #             loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
        #     else:
        #         loss = None

        if self.args.prediction_loss_only:
            return (None, None, None)

        # labels = inputs["labels"]
        # if labels.shape[-1] < gen_kwargs["max_length"]:
        #     labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (None, (generated_tokens_t2s, generated_tokens_s2t), None)

    def prediction_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ):
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        if self.args.deepspeed and not self.args.do_train:
            # no harm, but flagging to the user that deepspeed config is ignored for eval
            # flagging only for when --do_train wasn't passed as only then it's redundant
            logger.info("Detected the deepspeed argument but it will not be used for evaluation")

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, half it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Batch size = %d", batch_size)
        losses_host: List[torch.Tensor] = [None, None, None]
        preds_host_user: Union[torch.Tensor, List[torch.Tensor]] = None
        preds_host_agent: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host_user: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host_agent: Union[torch.Tensor, List[torch.Tensor]] = None

        world_size = max(1, self.args.world_size)

        eval_losses_gatherer_sum = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)
        eval_losses_gatherer_topic = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)
        eval_losses_gatherer_att = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)
        if not prediction_loss_only:
            # The actual number of eval_sample can be greater than num_examples in distributed settings (when we pass
            # a batch size to the sampler)
            make_multiple_of = None
            if hasattr(dataloader, "sampler") and isinstance(dataloader.sampler, SequentialDistributedSampler):
                make_multiple_of = dataloader.sampler.batch_size
            preds_gatherer_user = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)
            preds_gatherer_agent = DistributedTensorGatherer(world_size, num_examples,
                                                             make_multiple_of=make_multiple_of)
            labels_gatherer_user = DistributedTensorGatherer(world_size, num_examples,
                                                             make_multiple_of=make_multiple_of)
            labels_gatherer_agent = DistributedTensorGatherer(world_size, num_examples,
                                                              make_multiple_of=make_multiple_of)

        model.eval()

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        for step, inputs in enumerate(dataloader):
            loss_list, logits, labels = self.prediction_step(model, inputs, prediction_loss_only,
                                                             ignore_keys=ignore_keys)
            if loss_list is not None:
                loss_list = [loss.repeat(batch_size) for loss in loss_list]
                losses_host = loss_list if losses_host[0] is None else [torch.cat((losses_host[i], loss_list[i]), dim=0)
                                                                        for i in range(len(loss_list))]
            if logits is not None:
                preds_host_user = logits[0] if preds_host_user is None else nested_concat(preds_host_user, logits[0],
                                                                                          padding_index=-100)
                preds_host_agent = logits[1] if preds_host_agent is None else nested_concat(preds_host_agent, logits[1],
                                                                                            padding_index=-100)
            if labels is not None:
                labels_host_user = labels[0] if labels_host_user is None else nested_concat(labels_host_user, labels[0],
                                                                                            padding_index=-100)
                labels_host_agent = labels[1] if labels_host_agent is None else nested_concat(labels_host_agent,
                                                                                              labels[1],
                                                                                              padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                eval_losses_gatherer_sum.add_arrays(self._gather_and_numpify(losses_host[0], "eval_sum_losses"))
                eval_losses_gatherer_topic.add_arrays(self._gather_and_numpify(losses_host[1], "eval_topic_losses"))
                eval_losses_gatherer_att.add_arrays(self._gather_and_numpify(losses_host[2], "eval_att_losses"))
                if not prediction_loss_only:
                    preds_gatherer_user.add_arrays(self._gather_and_numpify(preds_host_user, "eval_preds_user"))
                    preds_gatherer_agent.add_arrays(self._gather_and_numpify(preds_host_agent, "eval_preds_agent"))
                    labels_gatherer_user.add_arrays(self._gather_and_numpify(labels_host_user, "eval_label_ids_user"))
                    labels_gatherer_agent.add_arrays(
                        self._gather_and_numpify(labels_host_agent, "eval_label_ids_agent"))

                # Set back to None to begin a new accumulation
                losses_host, preds_host_user, preds_host_agent, labels_host_user, labels_host_agent = [None, None, None], None, None, None, None


        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer_sum.add_arrays(self._gather_and_numpify(losses_host[0], "eval_sum_losses"))
        eval_losses_gatherer_topic.add_arrays(self._gather_and_numpify(losses_host[1], "eval_topic_losses"))
        eval_losses_gatherer_att.add_arrays(self._gather_and_numpify(losses_host[2], "eval_att_losses"))
        if not prediction_loss_only:
            preds_gatherer_user.add_arrays(self._gather_and_numpify(preds_host_user, "eval_preds_user"))
            preds_gatherer_agent.add_arrays(self._gather_and_numpify(preds_host_agent, "eval_preds_agent"))
            labels_gatherer_user.add_arrays(self._gather_and_numpify(labels_host_user, "eval_label_ids_user"))
            labels_gatherer_agent.add_arrays(self._gather_and_numpify(labels_host_agent, "eval_label_ids_agent"))

        eval_loss_sum = eval_losses_gatherer_sum.finalize()
        eval_loss_topic = eval_losses_gatherer_topic.finalize()
        eval_loss_att = eval_losses_gatherer_att.finalize()
        preds_user = preds_gatherer_user.finalize() if not prediction_loss_only else None
        preds_agent = preds_gatherer_agent.finalize() if not prediction_loss_only else None
        label_ids_user = labels_gatherer_user.finalize() if not prediction_loss_only else None
        label_ids_agent = labels_gatherer_agent.finalize() if not prediction_loss_only else None

        if self.compute_metrics is not None and preds_user is not None and label_ids_user is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds_user, label_ids=label_ids_user))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if eval_loss_sum is not None:
            metrics[f"{metric_key_prefix}_loss_sum"] = eval_loss_sum.mean().item()
        if eval_loss_topic is not None:
            metrics[f"{metric_key_prefix}_loss_topic"] = eval_loss_topic.mean().item()
        if eval_loss_att is not None:
            metrics[f"{metric_key_prefix}_loss_att"] = eval_loss_att.mean().item()
            metrics[
                f"{metric_key_prefix}_loss_total"] = eval_loss_sum.mean().item() + 0.2 * eval_loss_topic.mean().item() + eval_loss_att.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return (preds_user, preds_agent, label_ids_user, label_ids_agent, metrics)

    def predict(
            self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "eval",
            max_length: Optional[int] = None,
            num_beams: Optional[int] = None,
    ):
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:obj:`Dataset`):
                Dataset to run the predictions on. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. Has to implement the method :obj:`__len__`
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        .. note::

            If your predictions or labels have different sequence length (for instance because you're doing dynamic
            padding in a token classification task) the predictions will be padded (on the right) to allow for
            concatenation into one array. The padding index is -100.

        Returns: `NamedTuple` A namedtuple with the following keys:

            - predictions (:obj:`np.ndarray`): The predictions on :obj:`test_dataset`.
            - label_ids (:obj:`np.ndarray`, `optional`): The labels (if the dataset contained some).
            - metrics (:obj:`Dict[str, float]`, `optional`): The potential dictionary of metrics (if the dataset
              contained labels).
        """
        # memory metrics - must set up as early as possible
        self._max_length = max_length
        self._num_beams = num_beams
        self._memory_tracker.start()

        if test_dataset is not None and not isinstance(test_dataset, collections.abc.Sized):
            raise ValueError("test_dataset must implement __len__")

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        output = self.prediction_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        output[-1].update(speed_metrics(metric_key_prefix, start_time, len(test_dataset)))

        self._memory_tracker.stop_and_update_metrics(output[-1])

        return output

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            max_length: Optional[int] = None,
            num_beams: Optional[int] = None,
    ) -> Dict[str, float]:

        self._max_length = max_length
        self._num_beams = num_beams
        self._memory_tracker.start()

        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        output = self.prediction_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        n_samples = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
        output[-1].update(speed_metrics(metric_key_prefix, start_time, n_samples))
        self.log(output[-1])

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output[-1])

        self._memory_tracker.stop_and_update_metrics(output[-1])

        return output[-1]

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss, losses = self.compute_loss(model, inputs)
        else:
            loss, losses = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            losses = [s.mean() for s in losses]

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps
            losses = [s / self.args.gradient_accumulation_steps for s in losses]

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach(), [s.detach() for s in losses]

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        **kwargs,
    ):

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        self.is_in_train = True

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({self.args.output_dir})")

        if resume_from_checkpoint is not None and os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
            logger.info(f"Loading model from {resume_from_checkpoint}).")
            if isinstance(self.model, PreTrainedModel):
                self.model = self.model.from_pretrained(resume_from_checkpoint)
                model_reloaded = True
            else:
                state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self.model = self.model.to(self.args.device)
            self.model_wrapped = self.model

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                    self.args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(self.args.num_train_epochs)
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = self.args.max_steps
            num_train_epochs = 1
            num_update_steps_per_epoch = max_steps

        delay_optimizer_creation = self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE
        if self.args.deepspeed:
            model, optimizer, lr_scheduler = init_deepspeed(self, num_training_steps=max_steps)
            self.model = model.module
            self.model_wrapped = model  # will get further wrapped in DDP
            self.deepspeed = model  # DeepSpeedEngine object
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        if is_torch_tpu_available():
            world_size = xm.xrt_world_size()
        elif self.args.local_rank != -1:
            world_size = dist.get_world_size()
        else:
            world_size = 1

        total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps * world_size
        num_examples = (
            self.num_examples(train_dataloader)
            if train_dataset_is_sized
            else total_train_batch_size * self.args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, "trainer_state.json")
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, "trainer_state.json"))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not self.args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not self.args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        self.state.trial_params = hp_params(trial) if trial is not None else None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(self.args.device)
        tr_losses = [torch.tensor(0.0).to(self.args.device), torch.tensor(0.0).to(self.args.device)]
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        self._total_flos = self.state.total_flos
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not self.args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if train_dataset_is_sized
                else self.args.max_steps * self.args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                if (
                    ((step + 1) % self.args.gradient_accumulation_steps != 0)
                    and self.args.local_rank != -1
                    and self.args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        loss, losses = self.training_step(model, inputs)
                        tr_loss += loss
                        tr_losses = [tr_losses[i] + losses[i] for i in range(len(tr_losses))]
                else:
                    loss, losses = self.training_step(model, inputs)
                    tr_loss += loss
                    tr_losses = [tr_losses[i] + losses[i] for i in range(len(tr_losses))]
                self._total_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= self.args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.use_amp:
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(self.args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(self.args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                self.args.max_grad_norm,
                            )

                    # Optimizer step
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    if not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_losses, model, trial, epoch)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_losses, model, trial, epoch)

            if self.args.tpu_metrics_debug or self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif self.args.local_rank != -1:
                dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(self.model, PreTrainedModel):
                self.model = self.model.from_pretrained(self.state.best_model_checkpoint)
                if self.place_model_on_device:
                    self.model = self.model.to(self.args.device)
            else:
                state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
                )

        metrics = speed_metrics("train", start_time, self.state.max_steps)
        if self._total_flos is not None:
            self.store_flos()
            metrics["total_flos"] = self.state.total_flos
        self.log(metrics)

        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()

        if self.deepspeed:
            # free up any memory that might be useful for eval
            self.deepspeed = None
            self.optimizer = None
            self.lr_scheduler = None
            self.model_wrapped = self.model
            gc.collect()  # force memory release
            # to restore normal behavior outside of train replay the place_model_on_device logic w/o deepspeed
            self.place_model_on_device = self.args.place_model_on_device
            if self.is_model_parallel:
                self.place_model_on_device = False

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        return TrainOutput(self.state.global_step, self._total_loss_scalar / self.state.global_step, metrics)

    def _maybe_log_save_evaluate(self, tr_losses, model, trial, epoch):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_losses_scalar = [s.item() for s in tr_losses]
            # reset tr_loss to zero
            tr_losses[0] -= tr_losses[0]
            tr_losses[1] -= tr_losses[1]

            logs["loss_sum"] = round(tr_losses_scalar[0] / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["loss_topic"] = round(tr_losses_scalar[1] / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_losses_scalar[0]
            self._globalstep_last_logged = self.state.global_step

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate()
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


# class MultiBart(BartForConditionalGeneration):
#     def forward(self,
#         input_ids=None,
#         attention_mask=None,
#         decoder_input_ids=None,
#         decoder_attention_mask=None,
#         head_mask=None,
#         decoder_head_mask=None,
#         encoder_outputs=None,
#         past_key_values=None,
#         inputs_embeds=None,
#         decoder_inputs_embeds=None,
#         labels=None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):


