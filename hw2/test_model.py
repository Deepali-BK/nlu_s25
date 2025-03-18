
"""
Code for Problem 1 of HW 2.
"""
import pickle
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments

from train_model import preprocess_dataset

def compute_metrics(eval_pred):
    """Computes accuracy for evaluation."""
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def init_tester(directory: str) -> Trainer:
    """
    Prolem 2b: Implement this function.

    Creates a Trainer object that will be used to test a fine-tuned
    model on the IMDb test set. The Trainer should fulfill the criteria
    listed in the problem set.

    :param directory: The directory where the model being tested is
        saved
    :return: A Trainer used for testing
    """
    
    model = BertForSequenceClassification.from_pretrained(directory, num_labels=2)
    training_args = TrainingArguments(
        output_dir="./test_results",
        per_device_eval_batch_size=8,
        do_train=False,  # Disable training
        do_eval=True,    # Enable evaluation
        evaluation_strategy="no", #new
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        #eval_dataset=test_data,
        #eval_dataset=imdb, #new
        compute_metrics=compute_metrics,
    )

    return trainer


    #raise NotImplementedError("Problem 2b has not been completed yet!")


if __name__ == "__main__":  # Use this script to test your model
    model_name = "prajjwal1/bert-tiny"

    # Load IMDb dataset
    imdb = load_dataset("imdb")
    del imdb["train"]
    del imdb["unsupervised"]

    # Preprocess the dataset for the tester
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    imdb["test"] = preprocess_dataset(imdb["test"], tokenizer)

    # Set up tester
    tester = init_tester("path_to_your_best_model")

    # Test
    results = tester.predict(imdb["test"])
    with open("test_results.p", "wb") as f:
        pickle.dump(results, f)
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
