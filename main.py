import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from args import args
from checkpoint import checkpoint_exist, load_latest_checkpoint, save_checkpoint
from esim import ESIM
from snli_dataset import SNLIDataset, labels

writer = SummaryWriter(args.log_path)

train_set = SNLIDataset(args.train_path)
dev_set = SNLIDataset(args.dev_path, train_set.vocab)
test_set = SNLIDataset(args.test_path, train_set.vocab)

print(f"train set: {len(train_set)}, dev_set: {len(dev_set)}, test_set: {len(test_set)}")


def collate_fn(data):
    s1, s2, label = zip(*data)
    s1 = pad_sequence(s1, batch_first=True)
    s2 = pad_sequence(s2, batch_first=True)
    label = torch.as_tensor(label)
    return s1, s2, label


train_loader = DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_set, args.batch_size, drop_last=True, collate_fn=collate_fn)
test_loader = DataLoader(test_set, args.batch_size, drop_last=True, collate_fn=collate_fn)

print("module:")

net = ESIM(
    len(train_set.vocab),
    args.embed_size,
    args.hidden_size1,
    args.hidden_size2,
    args.num_layers,
    len(labels)
).to(args.device)

print(net)


def get_accuracy(predict, target):
    total_num = target.shape[0]
    accurate_num = (predict.topk(1)[1].squeeze() == target).sum().item()
    return accurate_num / total_num


def train(dataloader, net, criterion, optimizer):
    global global_train_step
    global global_epoch
    with tqdm(dataloader) as tqdm_loader:
        tqdm_loader.set_description(f"Epoch {global_epoch}")

        net.train()
        for s1, s2, target in tqdm_loader:
            global_train_step += 1

            s1 = s1.to(args.device)
            s2 = s2.to(args.device)
            target = target.to(args.device)

            predict = net(s1, s2)

            loss = criterion(predict, target)

            accuracy = get_accuracy(predict, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tqdm_loader.set_postfix(loss=loss.item(), accuracy=accuracy)

            writer.add_scalar("train loss", loss.item(), global_train_step)
            writer.add_scalar("train accuracy", accuracy, global_train_step)


def evaluate(tag, dataloader, net, criterion):
    net.eval()

    global global_epoch
    total_loss = 0
    accuracy = 0
    total_step = 0
    with torch.no_grad():
        for s1, s2, target in dataloader:
            s1 = s1.to(args.device)
            s2 = s2.to(args.device)
            target = target.to(args.device)

            predict = net(s1, s2)

            loss = criterion(predict, target)

            total_loss += loss

            acc = get_accuracy(predict, target)

            accuracy += acc

            total_step += 1

    accuracy /= total_step
    print(f"{tag}: total loss={total_loss}, accuracy={accuracy}")

    writer.add_scalar(tag + " loss", total_loss, global_epoch)
    writer.add_scalar(tag + " accuracy", accuracy, global_epoch)


global_epoch = 0
global_train_step = 0

loss_fn = nn.CrossEntropyLoss().to(args.device)

optim = torch.optim.SGD(net.parameters(), lr=args.learning_rate)

if checkpoint_exist(args.checkpoint_path):
    checkpoint = load_latest_checkpoint(args.checkpoint_path)
    net.load_state_dict(checkpoint["net"])
    optim.load_state_dict(checkpoint["optim"])
    global_epoch = checkpoint["global_epoch"]
    global_train_step = checkpoint["global_train_step"]

for i in range(args.epoch_size):
    global_epoch += 1

    train(train_loader, net, loss_fn, optim)

    save_checkpoint(net, optim, global_epoch, global_train_step)

    evaluate("dev", dev_loader, net, loss_fn)

    evaluate("test", test_loader, net, loss_fn)

writer.close()
