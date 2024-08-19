import string

from torch import nn, zeros, cat, autograd

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


class RNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        # Implicit dimension choice for log_softmax has been deprecated.
        # Change the call to include dim=X as an argument.
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return autograd.Variable(zeros(1, self.hidden_size))
