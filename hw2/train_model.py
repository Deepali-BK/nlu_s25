"""
Code for Problem 1 of HW 2.
"""
import pickle
from typing import Any, Dict
import os
import evaluate
import numpy as np
import optuna
#import optuna.samplers import GridSampler

from datasets import Dataset, load_dataset
# from transformers import BertTokenizerFast, BertForSequenceClassification, \
# Trainer, TrainingArguments, EvalPrediction
from transformers import BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments, EvalPrediction, EarlyStoppingCallback


def preprocess_dataset(dataset: Dataset, tokenizer: BertTokenizerFast) \
        -> Dataset:
    """
    Problem 1d: Implement this function.

    Preprocesses a dataset using a Hugging Face Tokenizer and prepares
    it for use in a Hugging Face Trainer.

    :param dataset: A dataset
    :param tokenizer: A tokenizer
    :return: The dataset, prepreprocessed using the tokenizer
    """

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",  
            truncation=True,
            max_length=512,  
        )
    #return dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return dataset.map(tokenize_function, batched=True)
    
    #raise NotImplementedError("Problem 1d has not been completed yet!")


def init_model(trial: Any, model_name: str, use_bitfit: bool = False) -> \
        BertForSequenceClassification:
    """
    Problem 2a: Implement this function.

    This function should be passed to your Trainer's model_init keyword
    argument. It will be used by the Trainer to initialize a new model
    for each hyperparameter tuning trial. Your implementation of this
    function should support training with BitFit by freezing all non-
    bias parameters of the initialized model.

    :param trial: This parameter is required by the Trainer, but it will
        not be used for this problem. Please ignore it
    :param model_name: The identifier listed in the Hugging Face Model
        Hub for the pre-trained model that will be loaded
    :param use_bitfit: If True, then all parameters will be frozen other
        than bias terms
    :return: A newly initialized pre-trained Transformer classifier
    """
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    if use_bitfit:
        for name, param in model.named_parameters():
            if "bias" not in name:
                param.requires_grad = False

    return model
    #raise NotImplementedError("Problem 2a has not been completed yet!")

#new
def compute_metrics(eval_pred: EvalPrediction):
    """Computes accuracy for evaluation."""
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

#new
def get_next_run_id(base_dir="checkpoints"):
    """Finds the next available run number by checking existing directories."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return 1  # Start numbering from 1

    existing_runs = [
        int(d.split("-")[-1]) for d in os.listdir(base_dir) 
        if d.startswith("run-") and d.split("-")[-1].isdigit()
    ]
    
    return max(existing_runs, default=0) + 1
    


    
def init_trainer(model_name: str, train_data: Dataset, val_data: Dataset,
                 use_bitfit: bool = False) -> Trainer:
    """
    Prolem 2b: Implement this function.

    Creates a Trainer object that will be used to fine-tune a BERT-tiny
    model on the IMDb dataset. The Trainer should fulfill the criteria
    listed in the problem set.

    :param model_name: The identifier listed in the Hugging Face Model
        Hub for the pre-trained model that will be fine-tuned
    :param train_data: The training data used to fine-tune the model
    :param val_data: The validation data used for hyperparameter tuning
    :param use_bitfit: If True, then all parameters will be frozen other
        than bias terms
    :return: A Trainer used for training
    """
    #new
    run_id = get_next_run_id() 
    output_dir = f"checkpoints/{run_id}"
    
    training_args = TrainingArguments(
        #output_dir="./checkpoints",
        output_dir = output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        #save_total_limit=3,  
        metric_for_best_model="accuracy", greater_is_better=True,
        logging_dir="./logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model_init=lambda trial: init_model(trial, model_name, use_bitfit),
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,  # Added metric computation
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    return trainer

    #raise NotImplementedError("Problem 2b has not been completed yet!")


def hyperparameter_search_settings() -> Dict[str, Any]:
    """
    Problem 2c: Implement this function.

    Returns keyword arguments passed to Trainer.hyperparameter_search.
    Your hyperparameter search must satisfy the criteria listed in the
    problem set.

    :return: Keyword arguments for Trainer.hyperparameter_search
    """
    
    search_space = {
    "learning_rate": [3e-5, 5e-5, 1e-4, 3e-4],
    "per_device_train_batch_size": [8, 16, 32, 64, 128]
    }
    
    return {
        "direction": "maximize",  # Optimize for accuracy
        "backend": "optuna",  # Use Optuna for hyperparameter tuning
         #"n_trials": None,  # Number of trials
        "hp_space": lambda trial: {
            "learning_rate": trial.suggest_categorical("learning_rate", search_space["learning_rate"]),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", search_space["per_device_train_batch_size"]),
            },
        "sampler": optuna.samplers.GridSampler(search_space),
        }

    #raise NotImplementedError("Problem 2c has not been completed yet!")

#new
def apply_bitfit(model):
    for name, param in model.named_parameters():
        if "bias" not in name:  # Freeze all except bias terms
            param.requires_grad = False


if __name__ == "__main__":  # Use this script to train your model
    model_name = "prajjwal1/bert-tiny"

    # Load IMDb dataset and create validation split
    imdb = load_dataset("imdb")
    split = imdb["train"].train_test_split(.2, seed=3463)
    imdb["train"] = split["train"]
    imdb["val"] = split["test"]
    del imdb["unsupervised"]
    del imdb["test"]

    # Preprocess the dataset for the trainer
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    imdb["train"] = preprocess_dataset(imdb["train"], tokenizer)
    imdb["val"] = preprocess_dataset(imdb["val"], tokenizer)

    training_args = TrainingArguments(
        output_dir="./checkpoints",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir="./logs",
        logging_steps=10,
    )

    # Fine-tune WITH BitFit
    model_with_bitfit = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    apply_bitfit(model_with_bitfit)  # Apply BitFit

    trainer_with_bitfit = Trainer(
        model_init=lambda trial: init_model(trial, "bert-base-uncased", use_bitfit=True),
        args=training_args,
        train_dataset=imdb["train"],
        eval_dataset=imdb["val"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    best_trial_with_bitfit = trainer_with_bitfit.hyperparameter_search(**hyperparameter_search_settings())

    # Save best hyperparameters
    with open("train_results_with_bitfit.p", "wb") as f:
        pickle.dump(best_trial_with_bitfit, f)

    # Fine-tune WITHOUT BitFit
    model_without_bitfit = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    trainer_without_bitfit = Trainer(
        model_init=lambda trial: init_model(trial, "bert-base-uncased", use_bitfit=False),
        args=training_args,
        train_dataset=imdb["train"],
        eval_dataset=imdb["val"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    best_trial_without_bitfit = trainer_without_bitfit.hyperparameter_search(**hyperparameter_search_settings())

    # Save best hyperparameters
    with open("train_results_without_bitfit.p", "wb") as f:
        pickle.dump(best_trial_without_bitfit, f)

    # Load results and print them
    with open("train_results_with_bitfit.p", "rb") as f:
        best_bitfit = pickle.load(f)

    with open("train_results_without_bitfit.p", "rb") as f:
        best_no_bitfit = pickle.load(f)

    # Extract best hyperparameters and accuracy
    results_table = f"""
    | Validation Accuracy | Learning Rate | Batch Size |
    |---------------------|--------------|------------|
    | {best_no_bitfit.objective:.4f}  | {best_no_bitfit.hyperparameters['learning_rate']} | {best_no_bitfit.hyperparameters['per_device_train_batch_size']} |
    | {best_bitfit.objective:.4f}  | {best_bitfit.hyperparameters['learning_rate']} | {best_bitfit.hyperparameters['per_device_train_batch_size']} |
    """

    print(results_table)


"""
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir="./logs",
        logging_steps=10,
    )

    
    # Set up trainer
    trainer = init_trainer(model_name, imdb["train"], imdb["val"],
                           use_bitfit=True)

    # Train and save the best hyperparameters
    
    best = trainer.hyperparameter_search(**hyperparameter_search_settings())
    print(best)
    with open("train_results.p", "wb") as f:
        pickle.dump(best, f)
  

    # Load BERT model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

  # Fine-tune WITH BitFit
    model_with_bitfit =apply_bitfit(model)  # Apply BitFit before training

    trainer_with_bitfit = Trainer(
        model=model_with_bitfit,
        args=training_args,
        train_dataset=imdb["train"],
        eval_dataset=imdb["val"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer = init_trainer(model_with_bitfit, imdb["train"], imdb["val"],
                           use_bitfit=True)


    best_trial_with_bitfit = trainer_with_bitfit.hyperparameter_search(**hyperparameter_search_settings())
    print(best_trial_with_bitfit)

  # Save best hyperparameters
    with open("train_results_with_bitfit.p", "wb") as f:
        pickle.dump(best_trial_with_bitfit, f)

  # Fine-tune WITHOUT BitFit (Train all parameters)
    model_without_bitfit = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    trainer_without_bitfit = Trainer(
        model=model_without_bitfit,
        args=training_args,
        train_dataset=imdb["train"],
        eval_dataset=imdb["val"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer = init_trainer(model_without_bitfit, imdb["train"], imdb["val"],
                           use_bitfit=False)


    best_trial_without_bitfit = trainer_without_bitfit.hyperparameter_search(**hyperparameter_search_settings())

  # Save best hyperparameters
    with open("train_results_without_bitfit.p", "wb") as f:
        pickle.dump(best_trial_without_bitfit, f)
    print(train_results_without_bitfit)

    with open("train_results_with_bitfit.p", "rb") as f:
        best_bitfit = pickle.load(f)

    with open("train_results_without_bitfit.p", "rb") as f:
        best_no_bitfit = pickle.load(f)

# Extract best hyperparameters and accuracy
    results_table = f
| Validation Accuracy | Learning Rate | Batch Size |
|---------------------|--------------|------------|
| {best_no_bitfit.objective:.4f}  | {best_no_bitfit.hyperparameters['learning_rate']} | {best_no_bitfit.hyperparameters['per_device_train_batch_size']} |
| {best_bitfit.objective:.4f}  | {best_bitfit.hyperparameters['learning_rate']} | {best_bitfit.hyperparameters['per_device_train_batch_size']} |

    print(results_table)

"""
    
