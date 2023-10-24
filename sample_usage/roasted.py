import sys
sys.path.insert(0,'../')
import FakeRoastUtil_v2
# pip install datasets transformers[sentencepiece]

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

def count_parameters(model):
    s = 0
    for p in model.parameters():
        if p.requires_grad:
            s+=p.numel()
    return s

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(
  tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
  tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

"""## Standard Model Loading (loading a new model to train from scratch)

"""

from transformers import AutoModelForSequenceClassification, AutoConfig
import torch
checkpoint = "bert-base-cased"
config = AutoConfig.from_pretrained(
        checkpoint,
        num_labels=2,
        finetuning_task="mrpc",
        use_auth_token=None,
    )
model = AutoModelForSequenceClassification.from_config(config)
original_parameters = count_parameters(model)
print("total parameters", original_parameters)
#----------------------------------- Roast Stuff -------------------------------
sparsity = 0.1 #10x compression
mapper_args = { "mapper": "pareto", "hasher" : "uhash", "block_k" : 16, "block_n" : 16, "block": 8, "seed" : 123321}
roaster = FakeRoastUtil_v2.ModelRoasterGradScaler(model, True, sparsity, verbose=FakeRoastUtil_v2.NONE,
                                            module_limit_size=None, init_std=0.01,
                                            scaler_mode="v1", mapper_args=mapper_args)
model = roaster.process()
final_parameters = count_parameters(model)
print("parameters reduced from", original_parameters, "to", final_parameters)
#--------------------------------------------------------------------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
"""Standard Optimizer and scheduler"""

from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)
from transformers import get_scheduler

num_epochs = 3
num_training_steps = min(1000, num_epochs * len(train_dataloader))
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

"""#### Standard Training"""

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
steps = 0
done = False
for epoch in range(num_epochs):
    for batch in train_dataloader:
        steps += 1
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # ---------------------------- ROast Stuff --------------------
        FakeRoastUtil_v2.RoastGradScaler().scale_step(model)
        # ---------------------------- Roast Stuff --------------------

        optimizer.step()

        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        if steps >= num_training_steps:
              done = True
              break
    if done:
          break

"""#### Standard evaluation"""

from datasets import load_metric

metric= load_metric("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(metric.compute())

