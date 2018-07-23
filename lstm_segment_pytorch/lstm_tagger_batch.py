import torch.nn as nn
import torch.optim as optim
from  lstm_utils import *
from lstm_model_batch import batch_size
import random
def train():
    word_to_ix = get_word_index()
    tag_to_ix = get_tag_index()
    model = LSTMTaggerBatch(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    training_data  = get_train_data()
    for epoch in range(200):  # again, normally you would NOT do 300 epochs, it is toy data
        try:
            batch_data = []
            batch_target = []
            random.shuffle(training_data)
            all_loss = 0
            for i, (sentence, tags) in enumerate(training_data):

                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()

                # Also, we need to clear out the hidden state of the LSTM,
                # detaching it from its history on the last instance.
                model.hidden = model.init_hidden()

                # Step 2. Get our inputs ready for the network, that is, turn them into
                # Tensors of word indices.
                sentence_in = prepare_sequence(sentence, word_to_ix)
                targets = prepare_sequence(tags, tag_to_ix)
                if sentence_in is None or targets is None:
                    continue
                batch_data.append(sentence_in)
                batch_target.append(targets)
                if len(batch_data) == batch_size:
                    data = torch.cat(batch_data).view(len(batch_data), len(sentence_in), -1)
                    # Step 3. Run our forward pass.
                    # print ("data shape", data.shape)
                    # print ("target shape", target.shape)

                    tag_scores = model(data)

                    # Step 4. Compute the loss, gradients, and update the parameters by
                    #  calling optimizer.step()
                    # print ("tag_scores out shape", tag_scores.shape)
                    loss = 0
                    for tag_s, t in zip(tag_scores, batch_target):
                        loss += loss_function(tag_s, t)
                    loss.backward()
                    optimizer.step()
                    # print (i, loss)
                    batch_data = []
                    all_loss += loss.item()
            save_checkpoint("x_batch", model, epoch )
            print("all loss", all_loss)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    train()