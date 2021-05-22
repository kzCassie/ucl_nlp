from components.dataset import Dataset
import six.moves.cPickle as pickle

train_file="data/conala/train.var_str_sep.bin"
dev_file="data/conala/dev.var_str_sep.bin"
vocab="data/conala/vocab.var_str_sep.new_dev.src_freq3.code_freq3.bin"

# train_set = Dataset.from_bin_file(train_file)
# dev_set   = Dataset.from_bin_file(dev_file)


vocab = pickle.load(open(vocab, 'rb'))
print(type(vocab))
