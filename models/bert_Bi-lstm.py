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
        self.test_path = dataset + '/data/CMDD_test_data.txt'                                  # Test set
        self.vocal_train = dataset + '/data/vocal_train.pth'  # Training set
        self.vocal_test = dataset + '/data/vocal_test.pth'                                     # Test set
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class_1.txt').readlines()]                                        # Class list
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'                  # Model training results
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')             # Device

        self.require_improvement = 1000                                 # If no improvement after 1000 batches, stop training
        self.num_classes = 2                       # Number of classes
        self.num_epochs = 100                                             # Number of epochs
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
        self.min_lr = 5e-7
        self.lstm_hidden = 128
        self.input_size = 1792


class TwoLayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, device, num_layers=5, batch_first=True, dropout=0.05):
        super(TwoLayerLSTM, self).__init__()
        # Define a two-layer LSTM network
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=batch_first,
                            dropout=dropout,
                            bidirectional=True)
        self.device = device
        self.num_classes = num_classes
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.layer_norm = nn.LayerNorm(hidden_size*2)

        # Define a fully connected layer for output classification (assuming num_classes output classes)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            # LSTM forward pass
            h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)  # Initialize hidden state
            c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)  # Initialize cell state

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.layer_norm(out[:, -1, :])
        out = self.fc(out)
        return out


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.dropout)
        self.lstm = TwoLayerLSTM(config.input_size, config.lstm_hidden, config.num_classes, config.device)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        context = x[0]  # Input sentence
        mask = x[2]  # Mask for padding parts, same size as the sentence, padding parts are marked with 0, e.g., [1, 1, 1, 1, 0, 0]
        vocal_feature = x[3]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        pooled_text_features = F.adaptive_avg_pool1d(vocal_feature.permute(0, 2, 1), 64).permute(0, 2, 1)
        feature = torch.cat((pooled_text_features, encoder_out), dim=2)
        out = self.lstm(feature)  # h_n shape: (num_layers * num_directions, batch, hidden_size)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        return out
