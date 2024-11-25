# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """Configuration parameters"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/CMDD_train_data.txt'                                # Training set
        self.dev_path = dataset + '/data/dev.txt'                                    # Validation set
        self.test_path = dataset + '/data/CMDD_test_data.txt'                                  # Test set
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class_1.txt').readlines()]                                # Class list
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # Model training results
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # Device

        self.require_improvement = 1000                                 # If no improvement after 1000 batches, stop training
        self.num_classes = 2                       # Number of classes
        self.num_epochs = 300                                             # Number of epochs
        self.batch_size = 128                                           # Mini-batch size
        self.pad_size = 64                                              # Length of each sentence (pad or truncate)
        self.learning_rate = 5e-5                                       # Learning rate
        self.bert_path = '/home/833/xzr/Code/NLP/self_bert_model/bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)                                   # Convolutional kernel sizes
        self.num_filters = 256                                          # Number of convolutional kernels (channels)
        self.dropout = 0.5
        self.n_layers = 2
        self.bidirectional = True
        self.hidden_dim = 128
        self.min_lr = 5e-7


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.dropout)
        self.rnn = nn.LSTM(config.hidden_size,
                           config.hidden_dim // 2,
                           num_layers=config.n_layers,
                           bidirectional=config.bidirectional,
                           dropout=config.dropout if config.n_layers > 1 else 0)
        self.fc = nn.Linear(config.hidden_dim, config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        context = x[0]  # Input sentence
        mask = x[2]  # Mask for padding parts, same size as the sentence, padding parts are marked with 0, e.g., [1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        encoder_out = encoder_out.permute(1, 0, 2)
        lstm_out, (h_n, c_n) = self.rnn(encoder_out)  # h_n shape: (num_layers * num_directions, batch, hidden_size)
        h_n = h_n.view(2, 2, -1, 128 // 2)[-1]  # Only take the last layer's forward and backward hidden states
        h_n = h_n.permute(1, 0, 2).contiguous()
        h_n = F.relu(h_n)
        h_n = torch.flatten(h_n, start_dim=1)
        h_n = self.dropout(h_n)
        out = self.fc(h_n)
        return out
