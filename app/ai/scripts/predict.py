from functools import lru_cache
from pathlib import Path

from ai.scripts.__meta__ import lineToTensor, RNN
from ai.scripts.config.settings import settings
from torch import load, autograd

all_categories = settings.all_categories


@lru_cache
def model_load(weights_file: Path = settings.model_weights_path):
    global all_categories

    print(f"{settings.n_letters=}, {settings.n_hidden=}, {settings.n_categories=}, {all_categories=}")

    rnn = RNN(
        settings.n_letters,
        settings.n_hidden,
        settings.n_categories
    )
    rnn.load_state_dict(load(weights_file))
    rnn.eval()

    return rnn

def model_inference(rnn, line_tensor):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


# Just return an output given a line
def evaluate(line_tensor, weights_file: Path):

    rnn = model_load(weights_file)
    output = model_inference(rnn, line_tensor)

    return output  


def predict(name, n_predictions: int, weights_file: Path):

    output = evaluate(
        autograd.Variable(lineToTensor(name)),
        weights_file=weights_file
    )

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print("(%.2f) %s" % (value, all_categories[category_index]))
        predictions.append(dict(value=value, category=all_categories[category_index]))

    return predictions


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-w", type=Path, default=Path(settings.model_weights_path), help="path to model weights")
    parser.add_argument("-n", type=int, default=3, help="return top n mosty likely classes")
    parser.add_argument("name", type=str, help="name to classify")
    args = parser.parse_args()

    predict(name=args.name, n_predictions=args.n, weights_file=args.w)
