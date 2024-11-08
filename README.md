# Light-finetuning

Lightweight fine-tuning (PEFT) of a foundational model

## Steps

+ Load a pre-trained model and evaluate its performance
+ Perform parameter-efficient fine tuning using the pre-trained model
+ Perform inference using the fine-tuned model and compare its performance to the original model

+ PEFT technique: LoRA
+ Model: GPT-2
+ Evaluation approach: `evaluate` method with a Hugging Face `Trainer`
+ Fine-tuning dataset: IMDB

## Results

### Pretrained model evaluation

Metrics:
+ 'eval_loss': 1.2469059228897095
+ 'accuracy': 0.47857141494750977

### LORA training and evaluation

trainable params: 294,912 || all params: 124,736,256 || trainable%: 0.2364

Metrics:
Training
+ 'train_runtime': 1015.3933,
+ 'train_samples_per_second': 1.379,
+ 'train_steps_per_second': 0.057,
+ 'train_loss': 7.476080006566541,
+ 'epoch': 1.98

Evaluation
+ 'eval_runtime': 83.0379,
+ 'eval_samples_per_second': 8.43,
+ 'eval_steps_per_second': 1.409,
+ 'epoch': 1.98

  + 'eval_loss': 0.7539224624633789,
  + 'eval_accuracy': 0.6328571200370789
