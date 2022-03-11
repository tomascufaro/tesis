#!/usr/bin/env python
# coding: utf-8

import inspect
import os
import sys
import torchaudio
import glob
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
from evaluator import Evaluator
from sklearn.model_selection import train_test_split
from database import Database
from dataset_preprocessor import Preprocessor, Standard_Scaler
from transformers.models.wav2vec2.feature_extraction_wav2vec2 import (
    Wav2Vec2FeatureExtractor,
)
import argparse
from parser import args

locals().update(args)
model_name = base_model.split('/')[-1]
db = Database(collection)
print(args)

# ## Prepare Data for Training
if OUTLIERS:
    csv_train = 'train.csv'
    csv_test = 'test.csv'
else:
    csv_train = f'train_no_outliers_{collection}.csv'
    csv_test = f'test_no_outliers_{collection}.csv'
    
from datasets import load_dataset, load_metric

save_path = ".csv_files"

data_files = {
    "train": f"{save_path}/{csv_train}",
    "validation": f"{save_path}/{csv_test}",
}

dataset = load_dataset(
    "csv",
    data_files=data_files,
    delimiter="\t",
)
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

print(train_dataset)
print(eval_dataset)


# We need to specify the input and output column
input_column = "path"
output_column = "emotion"


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    import multiprocessing

    ncpus = multiprocessing.cpu_count()
    torch.set_num_threads((ncpus-1)*2)

negative_cases = train_dataset[output_column].count("negative")
positive_cases = train_dataset[output_column].count("positive")
print("Casos negativos: ", negative_cases)
print("Casos positivos: ", positive_cases)

w0 = (negative_cases + positive_cases) / (2 * negative_cases)
w1 = (positive_cases + negative_cases) / (2 * positive_cases)
class_weights = torch.tensor([w0, w1])
if no_class_weights:
    class_weights = torch.tensor([float(1), float(1)])

# Analizar este metodo: https://arxiv.org/abs/1901.05555

print("Class_weights: ", class_weights)

# we need to distinguish the unique labels in our SER dataset
label_list = train_dataset.unique(output_column)
label_list.sort()  # Let's sort it for determinism
num_labels = len(label_list)
print(f"A classification problem with {num_labels} classes: {label_list}")


# There are three merge strategies `mean`, `sum`, and `max`

from transformers import AutoConfig, Wav2Vec2Processor


model_name_or_path = base_model
pooling_mode = "mean"

# config
config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    label2id={label: i for i, label in enumerate(label_list)},
    id2label={i: label for i, label in enumerate(label_list)},
    finetuning_task="wav2vec2_clf",
)
setattr(config, "pooling_mode", pooling_mode)


processor = Wav2Vec2FeatureExtractor.from_pretrained(
    model_name_or_path,
)
target_sampling_rate = processor.sampling_rate
print(f"The target sampling rate: {target_sampling_rate}")


from database import Database
from sklearn.preprocessing import MinMaxScaler
from audio_process import Filter
import torch.nn.functional as F


def speech_file_to_array_fn(path):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    bandpass_filter = Filter()
    speech = bandpass_filter.process(speech, sampling_rate)
    return speech


def label_to_id(label, label_list):

    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label


def get_features_from_db(path):
    id = "iemocap_" + os.path.basename(path) + "_"
    collection = (
        db.dataset_no_aug.loc[db.dataset_no_aug["_id"] == id]
        .drop(columns=["_id", "augmented", "label"])
        .to_numpy()
    )
    return collection


def preprocess_function(examples):
    speech_list = [speech_file_to_array_fn(path) for path in examples[input_column]]
    feature_list = [get_features_from_db(path) for path in examples[input_column]]
    target_list = [label_to_id(label, label_list) for label in examples[output_column]]
    result = processor(speech_list, sampling_rate=target_sampling_rate)
    result["labels"] = list(target_list)
    result["features"] = list(feature_list)

    return result


train_dataset = train_dataset.map(
    preprocess_function, batch_size=100, batched=True, num_proc=4
)
eval_dataset = eval_dataset.map(
    preprocess_function, batch_size=100, batched=True, num_proc=4
)

if dataset_size:
    train_dataset = train_dataset.select(range(dataset_size))
    eval_dataset = train_dataset.select(range(int(dataset_size * 0.2)))

# ## Model
#
# Before diving into the training part, we need to build our classification model based on the merge strategy.


from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.file_utils import ModelOutput


@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model,
)


class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        # self.input_layer = nn.Linear(config.hidden_size + 57, 128)
        # self.dropout = nn.Dropout(p=0.2)
        # self.dense = nn.Linear(128, 128)
        # # self.dense2 = nn.Linear(128, 128)
        # # self.dropout2 = nn.Dropout(config.final_dropout)
        # self.out_proj = nn.Linear(128, config.num_labels)
        self.dense = nn.Linear(config.hidden_size + 57, config.hidden_size + 57)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size + 57, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.sigmoid(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config, class_weights=[1, 1]):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = "mean"
        self.config = config
        self.class_weights = class_weights
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(self, hidden_states, mode="mean"):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']"
            )

        return outputs

    def forward(
        self,
        input_values,
        features,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)

        features = torch.tensor(features)
        size = features.size()
        features = torch.reshape(features, (size[0], size[-1]))
        features = torch.cat([hidden_states, features], 1)
        logits = self.classifier(features)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss(weight=self.class_weights)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch

import transformers
from transformers import Wav2Vec2Processor


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [feature["labels"] for feature in features]
        acoustic_features = [feature["features"] for feature in features]
        d_type = torch.long if isinstance(label_features[0], int) else torch.float

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(label_features, dtype=d_type)
        batch["features"] = torch.tensor(acoustic_features)
        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


# Next, the evaluation metric is defined. There are many pre-defined metrics for classification/regression problems, but in this case, we would continue with just **Accuracy** for classification and **MSE** for regression. You can define other metrics on your own.


is_regression = False


import numpy as np
from transformers import EvalPrediction
from evaluator import Evaluator


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    if is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        # evaluator = Evaluator()
        # metrics = Evaluator.evaluate_from_pred(p.label_ids, preds)
        # r = {'recall':metrics.recall}
        # return r
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


# Now, we can load the pretrained XLSR-Wav2Vec2 checkpoint into our classification model with a pooling strategy.


model = Wav2Vec2ForSpeechClassification.from_pretrained(
    model_name_or_path,
    config=config,
    class_weights=class_weights,
)


# The first component of XLSR-Wav2Vec2 consists of a stack of CNN layers that are used to extract acoustically meaningful - but contextually independent - features from the raw speech signal. This part of the model has already been sufficiently trained during pretraining and as stated in the [paper](https://arxiv.org/pdf/2006.13979.pdf) does not need to be fine-tuned anymore.
# Thus, we can set the `requires_grad` to `False` for all parameters of the *feature extraction* part.


model.freeze_feature_extractor()


# In a final step, we define all parameters related to training.


from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir=model_name,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=epochs,
    # fp16=True,
    save_steps=save_steps,
    eval_steps=eval_steps,
    logging_steps=10,
    learning_rate=1e-4,
    save_total_limit=2,
)


from typing import Any, Dict, Union
from transformers.trainer_callback import TrainerCallback
import torch
from packaging import version
from torch import nn
from transformers.data.data_collator import DataCollator
from typing import Callable
from transformers import (
    Trainer,
    is_apex_available,
)
from transformers.modeling_utils import PreTrainedModel
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


class CTCTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        class_weights: list = None,
    ):

        self.class_weights = class_weights
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        loss_fct = CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()


# Now, all instances can be passed to Trainer and we are ready to start training!

trainer = CTCTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor,
    class_weights=class_weights,
)



if TRAIN:
    trainer.train()


## Evaluation

if TEST:
    import librosa
    from sklearn.metrics import classification_report

    test_dataset = load_dataset(
        "csv", data_files={"test": f".csv_files/{csv_test}"}, delimiter="\t"
    )["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_checkpoint = 0
    checkpoints = glob.glob(f"{model_name}/checkpoint-*")
    for checkpoint in checkpoints:
        n = int(checkpoint.split("-")[-1])
        if n > n_checkpoint:
            n_checkpoint = n

    model_name_or_path = f"{model_name}/checkpoint-{n_checkpoint}"

    config = AutoConfig.from_pretrained(model_name_or_path)
    setattr(config, "pooling_mode", pooling_mode)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name_or_path,
    )
    model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(
        device
    )

    def speech_file_to_array_fn(batch):
        speech_array, sampling_rate = torchaudio.load(batch["path"])
        speech_array = speech_array.squeeze().numpy()
        speech_array = librosa.resample(
            np.asarray(speech_array), sampling_rate, processor.sampling_rate
        )

        batch["speech"] = speech_array
        id = "iemocap_" + os.path.basename(batch["path"]) + "_"
        collection = (
            db.dataset_no_aug.loc[db.dataset_no_aug["_id"] == id]
            .drop(columns=["_id", "augmented", "label"])
            .to_numpy()
        )
        batch["features"] = collection
        return batch

    def predict(batch):
        features = processor(
            batch["speech"],
            sampling_rate=processor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        acoustic_features = batch["features"]
        input_values = features.input_values.to(device)
        # attention_mask = features.attention_mask.to(device)

        with torch.no_grad():
            logits = model(input_values, features=acoustic_features).logits

        pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        batch["predicted"] = pred_ids
        return batch

    test_dataset = test_dataset.map(speech_file_to_array_fn)

    result = test_dataset.map(predict, batched=True, batch_size=8)

    label_names = [config.id2label[i] for i in range(config.num_labels)]

    y_true = [0 if name == "negative" else 1 for name in result["emotion"]]
    y_pred = result["predicted"]
    evaluator = Evaluator()
    metrics = evaluator.evaluate_from_pred(y_true, y_pred, beta=1)
    print(metrics)
    report = classification_report(y_true, y_pred, target_names=label_names)
    print(report)
     
    lines = inspect.getsource(Wav2Vec2ClassificationHead.__init__)
    classification_string =  "\n".join(lines.split("\n")[1:])
    lines = inspect.getsource(Wav2Vec2ClassificationHead.forward)
    classification_string2 =  "\n".join(lines.split("\n")[1:])
    with open(model_name_or_path + '/results.txt', 'a') as results_file:
        results_file.write(str(args))

        return (loss, outputs) if return_outputs else loss

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()


# Now, all instances can be passed to Trainer and we are ready to start training!

trainer = CTCTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor,
    class_weights=class_weights,
)



if TRAIN:
    trainer.train()


## Evaluation

if TEST:
    import librosa
    from sklearn.metrics import classification_report

    test_dataset = load_dataset(
        "csv", data_files={"test": f".csv_files/{csv_test}"}, delimiter="\t"
    )["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_checkpoint = 0
    checkpoints = glob.glob(f"{model_name}/checkpoint-*")
    for checkpoint in checkpoints:
        n = int(checkpoint.split("-")[-1])
        if n > n_checkpoint:
            n_checkpoint = n

    model_name_or_path = f"{model_name}/checkpoint-{n_checkpoint}"

    config = AutoConfig.from_pretrained(model_name_or_path)
    setattr(config, "pooling_mode", pooling_mode)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name_or_path,
    )
    model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(
        device
    )

    def speech_file_to_array_fn(batch):
        speech_array, sampling_rate = torchaudio.load(batch["path"])
        speech_array = speech_array.squeeze().numpy()
        speech_array = librosa.resample(
            np.asarray(speech_array), sampling_rate, processor.sampling_rate
        )

        batch["speech"] = speech_array
        id = "iemocap_" + os.path.basename(batch["path"]) + "_"
        collection = (

        return (loss, outputs) if return_outputs else loss

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()


# Now, all instances can be passed to Trainer and we are ready to start training!

trainer = CTCTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor,
    class_weights=class_weights,
)



if TRAIN:
    trainer.train()


## Evaluation

if TEST:
    import librosa
    from sklearn.metrics import classification_report

    test_dataset = load_dataset(
        "csv", data_files={"test": f".csv_files/{csv_test}"}, delimiter="\t"
    )["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_checkpoint = 0
    checkpoints = glob.glob(f"{model_name}/checkpoint-*")
    for checkpoint in checkpoints:
        n = int(checkpoint.split("-")[-1])
        if n > n_checkpoint:
            n_checkpoint = n

    model_name_or_path = f"{model_name}/checkpoint-{n_checkpoint}"

    config = AutoConfig.from_pretrained(model_name_or_path)
    setattr(config, "pooling_mode", pooling_mode)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name_or_path,
    )
    model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(
        device
    )

    def speech_file_to_array_fn(batch):
        speech_array, sampling_rate = torchaudio.load(batch["path"])
        speech_array = speech_array.squeeze().numpy()
        speech_array = librosa.resample(
            np.asarray(speech_array), sampling_rate, processor.sampling_rate
        )


        return (loss, outputs) if return_outputs else loss

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()


# Now, all instances can be passed to Trainer and we are ready to start training!

trainer = CTCTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor,
    class_weights=class_weights,
)



if TRAIN:
    trainer.train()


## Evaluation

if TEST:
    import librosa
    from sklearn.metrics import classification_report

    test_dataset = load_dataset(
        "csv", data_files={"test": f".csv_files/{csv_test}"}, delimiter="\t"
    )["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_checkpoint = 0
    checkpoints = glob.glob(f"{model_name}/checkpoint-*")
    for checkpoint in checkpoints:
        n = int(checkpoint.split("-")[-1])
        if n > n_checkpoint:
            n_checkpoint = n

    model_name_or_path = f"{model_name}/checkpoint-{n_checkpoint}"

    config = AutoConfig.from_pretrained(model_name_or_path)
    setattr(config, "pooling_mode", pooling_mode)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name_or_path,
    )
    model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(
        device
    )

    def speech_file_to_array_fn(batch):
        speech_array, sampling_rate = torchaudio.load(batch["path"])
        speech_array = speech_array.squeeze().numpy()
        speech_array = librosa.resample(
            np.asarray(speech_array), sampling_rate, processor.sampling_rate
        )

        batch["speech"] = speech_array
        id = "iemocap_" + os.path.basename(batch["path"]) + "_"
        collection = (
            db.dataset_no_aug.loc[db.dataset_no_aug["_id"] == id]
            .drop(columns=["_id", "augmented", "label"])
            .to_numpy()
        )
        batch["features"] = collection
        return batch

    def predict(batch):
        features = processor(
            batch["speech"],
            sampling_rate=processor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        acoustic_features = batch["features"]
        input_values = features.input_values.to(device)
        # attention_mask = features.attention_mask.to(device)

        with torch.no_grad():
            logits = model(input_values, features=acoustic_features).logits

        pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        batch["predicted"] = pred_ids
        return batch

    test_dataset = test_dataset.map(speech_file_to_array_fn)

    result = test_dataset.map(predict, batched=True, batch_size=8)

    label_names = [config.id2label[i] for i in range(config.num_labels)]

    y_true = [0 if name == "negative" else 1 for name in result["emotion"]]
    y_pred = result["predicted"]

    metrics = Evaluator().evaluate_from_pred(y_true, y_pred)
    print(metrics)
    report = classification_report(y_true, y_pred, target_names=label_names)
    print(report)

    lines = inspect.getsource(Wav2Vec2ClassificationHead.__init__)
    classification_string = "\n".join(lines.split("\n")[1:])
    lines = inspect.getsource(Wav2Vec2ClassificationHead.forward)
    classification_string2 = "\n".join(lines.split("\n")[1:])
    with open(model_name_or_path + "/results.txt", "a") as results_file:
        results_file.write(str(args))
        results_file.write("\n")
        results_file.write(str(report))
        results_file.write("\n\n\n")
        results_file.write(classification_string)
        results_file.write("\n")
        results_file.write(classification_string2)
        results_file.write("\n")
if shutdown:
    import os

    os.system("shutdown")

# In[9]:


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchaudio
# from transformers import AutoConfig, Wav2Vec2Processor

# import librosa
# import IPython.display as ipd
# import numpy as np
# import pandas as pd


# # In[ ]:


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_name_or_path = "facebook/wav2vec2-base-960h"
# config = AutoConfig.from_pretrained(model_name_or_path)
# processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
# sampling_rate = processor.feature_extractor.sampling_rate
# model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)


# # In[ ]:


# def speech_file_to_array_fn(path, sampling_rate):
#     speech_array, _sampling_rate = torchaudio.load(path)
#     resampler = torchaudio.transforms.Resample(_sampling_rate)
#     speech = resampler(speech_array).squeeze().numpy()
#     return speech


# def predict(path, sampling_rate):
#     speech = speech_file_to_arrtheay_fn(path, sampling_rate)
#     features = processor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

#     input_values = features.input_values.to(device)
#     attention_mask = features.attention_mask.to(device)

#     with torch.no_grad():
#         logits = model(input_values, attention_mask=attention_mask).logits

#     scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
#     outputs = [{"Emotion": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)]
#     return outputs


# STYLES = """
# <style>
# div.display_data {
#     margin: 0 auto;
#     max-width: 500px;
# }
# table.xxx {
#     margin: 50px !important;
#     float: right !important;
#     clear: both !important;
# }
# table.xxx td {
#     min-width: 300px !important;
#     text-align: center !important;
# }
# </style>
# """.strip()

# def prediction(df_row):
#     path, emotion = df_row["path"], df_row["emotion"]
#     df = pd.DataFrame([{"Emotion": emotion, "Sentence": "    "}])
#     setup = {
#         'border': 2,
#         'show_dimensions': True,
#         'justify': 'center',
#         'classes': 'xxx',
#         'escape': False,
#     }
#     ipd.display(ipd.HTML(STYLES + df.to_html(**setup) + "<br />"))
#     speech, sr = torchaudio.load(path)
#     speech = speech[0].numpy().squeeze()
#     speech = librosa.resample(np.asarray(speech), sr, sampling_rate)
#     ipd.display(ipd.Audio(data=np.asarray(speech), autoplay=True, rate=sampling_rate))

#     outputs = predict(path, sampling_rate)
#     r = pd.DataFrame(outputs)
#     ipd.display(ipd.HTML(STYLES + r.to_html(**setup) + "<br />"))


# # In[ ]:


# test = pd.read_csv("/content/data/test.csv", sep="\t")
# test.head()


# # In[ ]:


# prediction(test.iloc[0])


# # In[ ]:


# prediction(test.iloc[1])


# # In[ ]:


# prediction(test.iloc[2])
