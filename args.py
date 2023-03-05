from argparse import Namespace

import torch

args = Namespace(
    batch_size=512,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    embed_size=100,
    hidden_size1=200,
    hidden_size2=200,
    num_layers=2,
    epoch_size=15,
    learning_rate=0.01,

    train_path="./data/snli_1.0_train.txt",
    test_path="./data/snli_1.0_test.txt",
    dev_path="./data/snli_1.0_dev.txt",
    log_path="./logs",
    checkpoint_path="./checkpoints",
)