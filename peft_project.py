from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding, AutoModelForSequenceClassification
import torch
import numpy as np
from peft import LoraConfig, get_peft_model, AutoPeftModelForSequenceClassification

"""Loading and Evaluating a Foundation Model"""

#set device to gpu
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Load the train and test splits of the imdb dataset
splits = ['train', 'test']

dataset={split: dataset for split, dataset in zip(splits, load_dataset('imdb', split=splits))}

# Thin out the dataset to make it run faster 
for split in splits:
    dataset[split] = dataset[split].shuffle(seed=42).select(range(700))


#Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    """Preprocess the imdb dataset by returning tokenized examples."""
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = {}
for split in splits:
    tokenized_dataset[split] = dataset[split].map(preprocess_function, batched=True)

# print(tokenized_dataset)

#Load gpt Model
model = AutoModelForSequenceClassification.from_pretrained('gpt2')
model.config.pad_token_id = model.config.eos_token_id


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions=torch.from_numpy(predictions)
    labels = torch.from_numpy(labels).long()  # Convert labels to tensor
    loss = torch.nn.CrossEntropyLoss()(predictions, labels)  # Calculate the evaluation loss
    accuracy = (np.argmax(predictions, axis=1) == labels).float().mean()
     # Print the metrics dictionary for debugging
    metrics = {"eval_loss": loss.item(), "accuracy": accuracy.item()}
    print("Metrics:", metrics)

    return metrics
 


#Evaluate model

model_trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./data/initial_eval",
        learning_rate=2e-1,
        # Reduce the batch size if you don't have enough memory
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        label_names=["labels"],
    ),
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

model_trainer.evaluate()


#instatiate LoRa config
config = LoraConfig(fan_in_fan_out=True)

#get a trainable PEFT model
lora_model = get_peft_model(model, config)

#Checking Trainable Parameters of a PEFT Model
lora_model.print_trainable_parameters()

#Train and evaluate lora model
lora_trainer = Trainer(
    model=lora_model,
    args=TrainingArguments(
        output_dir="./data/lora_eval",
        learning_rate=2e-1,
        # Reduce the batch size if you don't have enough memory
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        gradient_accumulation_steps=4,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        label_names=["labels"],
    ),
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

lora_trainer.train()

#Save trained PEFT Model
lora_model.save_pretrained("gpt-lora")

#Load saved PEFT Model
lora_model = AutoPeftModelForSequenceClassification.from_pretrained("gpt-lora")

lora_model.config.pad_token_id = lora_model.config.eos_token_id

lora_trainer2 = Trainer(
    model=lora_model,
    args=TrainingArguments(
        output_dir="./data/lora_eval2",
        learning_rate=2e-1,
        # Reduce the batch size if you don't have enough memory
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        gradient_accumulation_steps=4,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        label_names=["labels"],
    ),
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

lora_trainer2.evaluate()
