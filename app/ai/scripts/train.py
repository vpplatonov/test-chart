"""Adapted from https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html"""
import glob
import json
import random
import time
from pathlib import Path

import math
import unicodedata
from config.settings import settings
from torch import nn, LongTensor, optim, save, autograd

from __meta__ import all_letters, n_letters, lineToTensor, RNN

n_hidden = 128
n_epochs = 100000
print_every = 5000
plot_every = 1000
learning_rate = (
    0.005  # If you set this too high, it might explode. If too low, it might not learn
)

category_lines = {}
n_categories = -1
all_categories = []


def findFiles(path):
    return glob.glob(path)


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in all_letters
    )


# Read a file and split into lines
def readLines(filename):
    lines = open(filename).read().strip().split("\n")
    return [unicodeToAscii(line) for line in lines]


def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1)  # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = autograd.Variable(LongTensor([all_categories.index(category)]))
    line_tensor = autograd.Variable(lineToTensor(line))

    return category, line, category_tensor, line_tensor


def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()
    optimizer.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.data.item()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-o",
        type=Path,
        default=Path(settings.model_weights_path),
        help="destination for weighths"
    )
    args = parser.parse_args()

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    all_categories = []

    # Build the category_lines dictionary, a list of lines per category
    # global category_lines, all_categories, n_categories

    for filename in findFiles(settings.train_data_path):
        category = filename.split("/")[-1].split(".")[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    n_categories = len(all_categories)

    # save n_letters, n_hidden, n_categories to json file
    with open(settings.model_metadata_path, 'w') as f:
        json.dump(dict(
            n_letters=n_letters,
            n_hidden=n_hidden,
            n_categories=n_categories,
            all_categories=all_categories
        ), f)

    rnn = RNN(input_size=n_letters, hidden_size=n_hidden, output_size=n_categories)
    optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()


    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return "%dm %ds" % (m, s)


    start = time.time()

    for epoch in range(1, n_epochs + 1):
        category, line, category_tensor, line_tensor = randomTrainingPair()
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss

        # Print epoch number, loss, name and guess
        if epoch % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = "✓" if guess == category else "✗ (%s)" % category
            print(
                "%d %d%% (%s) %.4f %s / %s %s"
                % (
                    epoch,
                    epoch / n_epochs * 100,
                    timeSince(start),
                    loss,
                    line,
                    guess,
                    correct,
                )
            )

        # Add current loss avg to list of losses
        if epoch % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    save(rnn.state_dict(), args.o)
