from bilstm_model import *
from bilstm_util import  *
# 制造一些训练数据
training_data = get_train_data()


word_to_ix = get_word_index()
tag_to_ix = get_tag_index()

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# 在训练之前检查预测结果
precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
precheck_tags = torch.LongTensor([tag_to_ix[t] for t in training_data[0][1]])
print(model(precheck_sent))
test_data = get_test_data1()


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
        _, predict_label = model(sentence_in)

        s += len([True for a, b in zip(predict_label, targets) if a == b])
        aa += targets.shape[0]
    return (s / aa)


# 确认从之前的 LSTM 部分的 prepare_sequence 被加载了
for epoch in range(300):  # 又一次, 正常情况下你不会训练300个 epoch, 这只是示例数据
    all_loss = 0
    for sentence, tags in training_data:
        # 第一步: 需要记住的是Pytorch会累积梯度
        # 我们需要在每次实例之前把它们清除
        model.zero_grad()

        # 第二步: 为我们的网络准备好输入, 即
        # 把它们转变成单词索引变量 (Variables)
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.LongTensor([tag_to_ix[t] for t in tags])

        # 第三步: 运行前向传递.
        neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets)

        # 第四步: 计算损失, 梯度以及
        # 使用 optimizer.step() 来更新参数
        neg_log_likelihood.backward()
        all_loss += neg_log_likelihood.item()
        optimizer.step()

    print ("train precision", cal_precision(training_data, model, word_to_ix, tag_to_ix))
    print ("test precision", cal_precision(test_data, model, word_to_ix, tag_to_ix))

    print ("all_loss", all_loss)
    save_checkpoint("bilstm", model, epoch)

# 在训练之后检查预测结果
precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
print(model(precheck_sent))
# 我们完成了!