import warnings

from datasets import load_dataset
from torch import bfloat16

from transformers import (
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    T5ForConditionalGeneration,
    T5TokenizerFast,
)

warnings.filterwarnings("ignore")


class Teli5Model:
    def __init__(self, pre_trained, tokenizer, dataset):
        self.model = T5ForConditionalGeneration.from_pretrained(
            pre_trained, return_dict=True, torch_dtype=bfloat16
        )
        self.tokenizer = T5TokenizerFast.from_pretrained(tokenizer)
        self.train = load_dataset(dataset, split="train")
        self.train.set_format(type="torch")

    def train(self, training_args):
        ignore_pad_token_for_loss = True
        label_pad_token_id = (
            -100 if ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        )
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=None,
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train,
            data_collator=data_collator,
        )

        # Training
        self.trainer.train()
        self.trainer.push_to_hub()


def main():
    PRE_TRAINED = "google/t5-v1_1-base"
    TOKENIZER = "t5-base"
    DATASET = "rusano/ELI5_custom_encoded"
    TRAINING_ARGS = TrainingArguments(
        run_name="Teli5",
        output_dir="../models",
        overwrite_output_dir=True,
        per_device_train_batch_size=8,
        learning_rate=1e-4,
        weight_decay=1e-2,
        num_train_epochs=3,
        log_level="info",
        logging_dir="../logs",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=10,
        save_safetensors=True,
        fp16=True,
        report_to="none",
        push_to_hub=True,
        hub_model_id="rusano/Teli5",
        # gradient_checkpointing=True,
        auto_find_batch_size=True,
    )
    model = Teli5Model(PRE_TRAINED, TOKENIZER, DATASET)
    model.train(TRAINING_ARGS)


if __name__ == "__main__":
    main()
