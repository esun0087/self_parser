import torch.nn as nn
import torch.optim as optim
import torch
from  lstm_utils import *


def cal_precision(training_data, model, word_to_ix, tag_to_ix):
    s = 0
    aa = 0
    for i, (sentence, tags) in enumerate(training_data):
        model.zero_grad()
        model.hidden = model.init_hidden()
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        if sentence_in is None or targets is None:
            continue
        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)
        predict_label = torch.argmax(tag_scores, dim = 1)
        s += len([True for a, b in zip(predict_label, targets) if a.item() == b.item()])
        aa += targets.shape[0]
    return (s / aa)



def train():
    word_to_ix = get_word_index()
    tag_to_ix = get_tag_index()
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    training_data  = get_train_data()
    test_data = get_test_data1()
    for epoch in range(200):  # again, normally you would NOT do 300 epochs, it is toy data
        try:
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
                # Step 3. Run our forward pass.
                tag_scores = model(sentence_in)

                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                # print ("tag_score shape", tag_scores.shape)
                # print ("targets shape", targets.shape)

                loss = loss_function(tag_scores, targets)
                all_loss += loss.item()
                loss.backward()
                optimizer.step()
            print ("all loss", all_loss)
            print ("train precision", epoch, cal_precision(training_data, model,  word_to_ix, tag_to_ix))
            print ("test precision", epoch, cal_precision(test_data, model,  word_to_ix, tag_to_ix))

            save_checkpoint("x", model, epoch )
        except Exception as e:
            print(e)


if __name__ == '__main__':
    train()