"""
Training script for a question‑answering model.

This module defines a command‑line interface for training a HuggingFace
transformer model on a QA dataset.  It handles argument parsing,
dataset loading, preprocessing, and integration with Weights & Biases
for experiment tracking.
"""

import json
import datetime
import os.path

from datasets import load_dataset
from argparse import ArgumentParser
from datasets.arrow_dataset import Dataset

from transformers import (
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    DefaultDataCollator,
)

from rdl_ml_utils.handlers.wandb_handler import WanDBHandler

from result_loggers import RadLabQAModelWandbConfig

tokenizer = None
max_context_size = 512


def make_parser(desc: str = "") -> ArgumentParser:
    """
    Create an argument parser for the training script.

    Parameters
    ----------
    desc : str, optional
        Description shown in the help output.  Defaults to an empty string.

    Returns
    -------
    ArgumentParser
        Configured parser with options for model path, dataset, output
        directory, and other training parameters.
    """
    p = ArgumentParser(description=desc)
    p.add_argument(
        "-m",
        "--model-path",
        dest="model_path",
        default="sdadas/polish-roberta-large-v2",
    )
    p.add_argument(
        "-d", "--dataset-path", dest="dataset_path", default="clarin-pl/poquad"
    )
    p.add_argument(
        "-o", "--output-dir", dest="output_dir", default="/mnt/local/models/qa"
    )
    p.add_argument(
        "-O",
        "--out-model-name",
        dest="out_model_name",
        default="polish-roberta-large-v2-qa",
    )
    p.add_argument(
        "--max-context-size", dest="max_context_size", default=512, type=int
    )
    p.add_argument(
        "-s",
        "--store-artifacts-to-wb",
        dest="store_artifacts_to_wb",
        action="store_true",
    )
    return p


def load_qa_dataset(dataset_path: str) -> Dataset:
    """
    Load a QA dataset from the HuggingFace hub.

    Parameters
    ----------
    dataset_path : str
        Identifier of the dataset to load (e.g., ``clarin-pl/poquad``).

    Returns
    -------
    Dataset
        The loaded dataset object.
    """
    # squad = load_dataset(dataset_path, split="train[:5000]")
    poquad = load_dataset(dataset_path)
    return poquad


def preprocess_function(examples):
    """
    Preprocess raw examples for model training.

    The function tokenizes questions and contexts, computes offset
    mappings, and determines start/end token positions for the answer
    spans.  It returns a dictionary compatible with the HuggingFace
    Trainer API.

    Parameters
    ----------
    examples : dict
        A batch of raw examples containing ``question``, ``context``,
        and ``answers`` fields.

    Returns
    -------
    dict
        Tokenized inputs with ``start_positions`` and ``end_positions``
        added.
    """
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_context_size,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if (
            offset[context_start][0] > end_char
            or offset[context_end][1] < start_char
        ):
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def main(argv=None):
    """
    Entry point for training the QA model.

    Parses command‑line arguments, loads and preprocesses the dataset,
    configures WandB logging, and runs the training loop using the
    HuggingFace ``Trainer`` class.  Model artifacts and training
    arguments are saved to the specified output directory.

    Parameters
    ----------
    argv : list of str, optional
        Custom argument list; if ``None`` the arguments are taken from
        ``sys.argv``.
    """

    global tokenizer
    global max_context_size

    args = make_parser().parse_args(argv)

    max_context_size = args.max_context_size

    dataset_odir = None
    dataset = load_qa_dataset(args.dataset_path)
    if args.store_artifacts_to_wb:
        dataset_odir = args.dataset_path.replace("/", "_")
        dataset.save_to_disk(dataset_odir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    model_workdir = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    out_dir_model = os.path.join(args.output_dir, model_workdir, args.out_model_name)
    print("out_dir_model:", out_dir_model)

    training_args = TrainingArguments(
        output_dir=out_dir_model,
        evaluation_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=2000,
        learning_rate=2e-6,  # 2e-5,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        num_train_epochs=5,
        weight_decay=0.1,
        push_to_hub=False,
        report_to=["wandb"],
        load_best_model_at_end=True,
    )

    if not os.path.exists(out_dir_model):
        os.makedirs(out_dir_model)
    with open(os.path.join(out_dir_model, "args_main.json"), "wt") as fout:
        json.dump(args.__dict__, fout, indent=2)

    with open(os.path.join(out_dir_model, "args_trainig.json"), "wt") as fout:
        fout.write(json.dumps(training_args.to_json_string(), indent=2))

    run_conf_dict = {
        "dataset_path": args.dataset_path,
        "base_model": args.model_path,
        "output_dir": args.output_dir,
        "out_model_name": args.out_model_name,
        "max_context_size": args.max_context_size,
    }

    wnb_config = RadLabQAModelWandbConfig()
    WanDBHandler.init_wandb(
        wandb_config=wnb_config,
        run_config=run_conf_dict,
        training_args=training_args,
    )

    if args.store_artifacts_to_wb:
        odata_name = dataset_odir
        WanDBHandler.add_dataset(local_path=dataset_odir, name=odata_name)

    model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DefaultDataCollator(),
    )

    trainer.train()

    out_model_path = os.path.join(out_dir_model, "best_model")
    trainer.save_model(out_model_path)

    if args.store_artifacts_to_wb:
        WanDBHandler.add_model(
            name=wnb_config.PROJECT_NAME + "_" + wnb_config.BASE_RUN_NAME,
            local_path=out_model_path,
        )

    WanDBHandler.finish_wand()


if __name__ == "__main__":
    main()
