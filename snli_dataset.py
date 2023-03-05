import re

import torch
from torch.utils.data import Dataset

labels = {"entailment": 0, "contradiction": 1, "neutral": 2, "-": 3}


def get_word_index(vocab, word):
    if word in vocab:
        return vocab[word]
    else:
        return vocab["<pad>"]


class SNLIDataset(Dataset):
    def __init__(self, file_path, vocab=None):
        s1 = 5
        s2 = 6
        l = 0

        if vocab is None:
            self.build_vocab = True
            self.vocab = {"<pad>": 0}
        else:
            self.build_vocab = False
            self.vocab = vocab

        self.s1 = []
        self.s2 = []
        self.labels = []
        with open(file_path, encoding="UTF-8") as f:
            with tqdm(f) as tqdm_file:
                tqdm_file.set_description("Load Data")

                for index, line in enumerate(tqdm_file):
                    if index == 0:
                        continue

                    split_lines = line.split("\t")

                    sentence1 = split_lines[s1]
                    sentence2 = split_lines[s2]
                    label = split_lines[l]

                    self.s1.append(self.__sentence2tensor(sentence1))
                    self.s2.append(self.__sentence2tensor(sentence2))
                    self.labels.append(torch.as_tensor(labels[label]))

    def __sentence2tensor(self, s):
        rt = []
        for char in re.findall("[a-zA-Z-]+", s):
            if self.build_vocab and char not in self.vocab:
                self.vocab[char] = len(self.vocab)
            rt.append(get_word_index(self.vocab, char))
        return torch.as_tensor(rt)

    def __getitem__(self, item):
        return self.s1[item], self.s2[item], self.labels[item]

    def __len__(self):
        return len(self.s1)