# LLMs FineTuning
**FineTune LLMs on different Datasets with Different Tasks using different techniques such as `supervised fintuning` `Proximal policy optimization`, `direct preference optimization` and `Group Relative plicy optimazation`.**

to run any of these models all you need to do is cloning the repo locally and just run the files.
to run with out any errors ensure you have the latest versions of `Transformers` and `Unsloth` packages.
```bash
!pip install -U transformers -q
!pip install -U unsloth -q
!pip install -U peft -q
!pip install -U trl -q
!pip install -U bitsandbytes -q
```

i finetuned most of the models with parameter efficent finetuning methods `PEFT`.
**`Peft`**: lets you fine-tune large pre-trained models by adapting only a small portion of parameters, instead of retraining everything. this help on reducing costs and training time.<br>
**`Low Rank Adapters`**: is a method works by injecting a trainable small low rank metrices withen the selected transformers layers. train far fewer parameters yet often matches full-finetune performance.
```bash
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=['q_proj', 'k_proj','v_proj','o_proj','up_proj','down_proj','gate_proj'],
    bias="none",
    task_type="CAUSAL_LM"
)
```
**to load any dataset used in these notebooks or any others from the huggingface hub you only need these four lines of code.**


```bash
!pip install -U datasets 

from datasets import load_dataset

dataset_name = "the name of the dataset on the hub"

dataset = load_dataset(dataset_name, split="the split you want to load")

```
