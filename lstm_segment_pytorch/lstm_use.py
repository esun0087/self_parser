import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
lstm = nn.LSTM(3, 3, batch_first=True)  # Input dim is 3, output dim is 3
inputs = [[torch.randn(1, 3) for _ in range(5)] for i in range(2)]  # make a sequence of length 5
tmp_inputs = copy.deepcopy(inputs)
# initialize the hidden state.
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))

tmp_hidden = copy.deepcopy(hidden)
# for each_inputs in inputs:
#     t = hidden
#     for i in each_inputs:
#         # Step through the sequence one element at a time.
#         # after each step, hidden contains the hidden state.
#         out, t = lstm(i.view(1, 1, -1), t)
#         print (out, out.shape, t)
#     print ("over")


# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension
# for each_inputs in inputs:
#     each_inputs = torch.cat(each_inputs).view(1, len(each_inputs), -1)
#     hidden = tmp_hidden  # clean out hidden state
#     out, hidden = lstm(each_inputs, hidden)
#     print(out)
#     print(hidden)
#     print("over1")

batch_inputs = torch.cat([torch.cat(each_inputs) for each_inputs in inputs]).view(len(inputs), len(inputs[0]), -1)
print ("bach_shape", batch_inputs.shape)
hidden =  (torch.randn(1, 2, 3),
          torch.randn(1, 2, 3))
out, hidden = lstm(batch_inputs, hidden)
print(out, out.shape)
print(hidden[0])
print("over2")