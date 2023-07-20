import gdown
from os.path import join
from os import mkdir

train_id = "19Mb4ZoUzt_6Aa6is4D4_8Y_25eF8Xsba"
eval_id = "13XsUN3gp5N2FbQ4rCmBFSdoieNyebyYm"
DATA_DIR = "./data/local"
train_file = "ELI5_train.jsonl"
eval_file = "ELI5_val.jsonl"

gdown.download(id=train_id, output=join(DATA_DIR, train_file), use_cookies=False)
gdown.download(id=eval_id, output=join(DATA_DIR, eval_file), use_cookies=False)

try:
    mkdir("./models")
except:
    pass

try:
    mkdir("./logs")
except:
    pass
