import numpy as np
import warnings

from datasets import load_dataset
from evaluate import load
from torch import Tensor, bfloat16

from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
)


class Teli5Model:
    def __init__(self, model, tokenizer, dataset, input_len, generator_args):
        self.model = T5ForConditionalGeneration.from_pretrained(
            model, return_dict=True, torch_dtype=bfloat16
        )
        self.tokenizer = T5TokenizerFast.from_pretrained(tokenizer)
        self.test = load_dataset(dataset, split="test")
        self.test.set_format(type="torch")
        self.input_len = input_len
        self.generator_args = generator_args

    def compute_metrics(eval_pred, metric):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    def predict(self, batch):
        input_ids = batch["input_ids"].to("cuda")
        res = self.generate(input_ids, self.generator_args)
        output = self.tokenizer.batch_decode(res, skip_special_tokens=True)
        output = [item.split(str("</s>")) for item in output]
        return {"predict": output}

    def evaluation(self, metrics_list):
        metrics = {}
        for metric in metrics_list:
            metrics[metric] = load(metric, "mrpc")

        self.test = self.test.map(self.predict, batched=True)
        results = {}
        for metric in metrics_list:
            results[metric] = metrics[metrics].compute(
                self.test["input_ids", self.test["predict"]]
            )
        return results


def main():
    MODEL = "rusano/Teli5"
    TOKENIZER = "t5-base"
    DATASET = "rusano/ELI5_custom_encoded"
    INPUT_LEN = 512
    METRICS = ["glue"]
    GENERATOR_ARGS = {
        "max_length": 512,
        "num_beams": 4,
        "length_penalty": 1.5,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
    }
    model = Teli5Model(MODEL, TOKENIZER, DATASET, INPUT_LEN, GENERATOR_ARGS)
    result = model.evaluation(METRICS)
    print(result)


if __name__ == "__main__":
    main()
