from datasets import Dataset, DatasetDict, load_dataset


class Teli5Dataset:
    def __init__(self, data_dir, train_file, test_file, column_remove):
        self.data_dir = data_dir
        self.train_file = train_file
        self.test_file = test_file
        self.data = DatasetDict(
            {
                "train": load_dataset("json", data_dir=data_dir, data_files=train_file)[
                    "train"
                ].remove_columns(column_remove),
                "test": load_dataset("json", data_dir=data_dir, data_files=test_file)[
                    "train"
                ].remove_columns(column_remove),
            }
        )

    def remove_score(batch):
        ctxs = []
        for ids in range(len(batch["ctxs"])):
            contexts = []
            for context in batch["ctxs"][ids]:
                contexts.append(context[0])
            ctxs.append(contexts)
        return {"ctxs": ctxs}

    def one_pair_qc(batch):
        answers = []
        ctxs = []
        for ids in range(len(batch["ctxs"])):
            answers.append(batch["answers"][ids][0])
            ctxs.append(batch["ctxs"][ids][0])

        return {"answer": answers, "context": ctxs}

    def train_test_split(data, train_size=None):
        size = data["train"].num_rows
        train_size = int(size * train_size)
        train = Dataset.from_dict(data["train"][:train_size])
        val = Dataset.from_dict(data["train"][train_size:])
        test = Dataset.from_dict(data["test"][:])
        return DatasetDict(
            {
                "train": train,
                "val": val,
                "test": test,
            }
        )

    def transform(self):
        self.data["test"] = self.data["test"].map(
            self.remove_score,
            batched=True,
            remove_columns=["ctxs"],
        )
        self.data = self.data.map(
            self.one_pair_qc,
            batched=True,
            remove_columns=["answers", "ctxs"],
        )
        self.data = self.train_test_split(self.data, train_size=0.8)

        return self.data

    def push(self, dataset, max_shard_size):
        self.data.push_to_hub(dataset, max_shard_size=max_shard_size)


def main():
    DATA_DIR = "../data"
    TRAIN_FILE = "ELI5_train.jsonl"
    TEST_FILE = "ELI5_val.jsonl"
    COLUMN_REMOVE = ["question_id"]
    DATASET_HUB = "rusano/ELI5_custom"
    dataset = Teli5Dataset(DATA_DIR, TRAIN_FILE, TEST_FILE, COLUMN_REMOVE)
    dataset.transform()
    dataset.push(DATASET_HUB, "1GB")


if __name__ == "__main__":
    main()
