from datasets import load_dataset
from torch import Tensor
from transformers import T5TokenizerFast


class Teli5Encoder:
    def __init__(self, dataset, tokenizer, input_len, target_len) -> None:
        self.data = load_dataset(dataset)
        self.tokenizer = T5TokenizerFast(tokenizer)
        self.input_len = input_len
        self.target_len = target_len

    def batch_tokenize(self, batch):
        inputs = []
        target = []
        for ids in range(len(batch["question"])):
            inputs.append(f"{batch['question'][ids]}</s>{batch['context'][ids]}</s>")
            target.append(f"{batch['answer'][ids]}</s>")

        input_encoding = self.tokenizer.batch_encode_plus(
            inputs,
            max_length=self.input_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        target_encoding = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.target_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        return {
            "input_ids": Tensor(input_encoding["input_ids"].squeeze()),
            "attention_mask": Tensor(input_encoding["attention_mask"].squeeze()),
            "labels": Tensor(target_encoding["input_ids"].squeeze()),
            "decoder_attention_mask": Tensor(
                target_encoding["attention_mask"].squeeze()
            ),
        }

    def encode(self):
        self.data = self.data.map(
            self.batch_tokenize,
            batched=True,
            remove_columns=self.data["train"].column_names,
        )

        return self.data

    def push(self, dataset, max_shard_size):
        self.data.push_to_hub(dataset, max_shard_size=max_shard_size)


def main():
    DATASET = "rusano/ELI5_custom"
    TOKENIZER = "t5-base"
    INPUT_LEN = 512
    TARGET_LEN = 256
    DATASET_HUB = "rusano/ELI5_custom_encoded"
    dataset = Teli5Encoder(DATASET, TOKENIZER, INPUT_LEN, TARGET_LEN)
    dataset.encode()
    dataset.push(DATASET_HUB, "1GB")


if __name__ == "__main__":
    main()
