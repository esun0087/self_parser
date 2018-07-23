from lstm_utils import *

def run_model(model, line):
    model.zero_grad()
    result = model(line)
    return result

def predict():
    model, word_to_idx, tag_to_idx, idx_to_tag = load_model()
    print (tag_to_idx)
    idx_2_tag = {i:j for j,i in tag_to_idx.items()}
    with codecs.open("test.out", "w", "utf-8") as f:
        for line, label in get_test_data():
            x = prepare_sequence(line, word_to_idx)
            if x is not None:
                result = run_model(model, x)
                for i, j in zip(torch.argmax(result, dim = 1), line):
                    f.write("\t".join((j, idx_2_tag[i.item()])) + "\n")
                f.write(("........\n"))

if __name__ == "__main__":
    predict()

