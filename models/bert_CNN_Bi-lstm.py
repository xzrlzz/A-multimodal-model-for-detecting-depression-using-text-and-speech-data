# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """Configuration parameters"""
    def __init__(self, dataset):
        self.model_name = 'bert_multi_CNN_LSTM'
        self.train_path = dataset + '/data/CMDD_train_data.txt'                                # Training set
        self.test_path = dataset + '/data/CMDD_test_data.txt'
        self.vocal_train = dataset + '/data/vocal_train.pth'  # Training set
        self.vocal_test = dataset + '/data/vocal_test.pth'
        # Test set
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class_1.txt').readlines()]                                # Class list
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # Model training results
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # Device

        self.require_improvement = 1000                                 # If no improvement after 1000 batches, stop training
        self.num_classes = 2                       # Number of classes
        self.num_epochs = 100                                             # Number of epochs
        self.batch_size = 128                                           # Mini-batch size
        self.pad_size = 64                                              # Length of each sentence (pad or truncate)
        self.learning_rate = 5e-5                                       # Learning rate
        self.bert_path = '/home/833/xzr/Code/NLP/multimodel_new/bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.filter_sizes = (2, 4, 6, 8)                                   # Convolutional kernel sizes
        self.num_filters = 256                                          # Number of convolutional kernels (channels)
        self.dropout = 0.1
        self.vocal_hidden = 1024
        self.lstm_hidden = 128
        self.input_size = 267


class TwoLayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, device, num_layers=2, batch_first=True, dropout=0.02):
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


class ConvBlock_1(nn.Module):
    def __init__(self, channel_in=149, channel=149, kernel_size=3, stride=1, pooling_kernel_size=2, pooling_stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(channel_in, channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activate = nn.ReLU()
        self.norm = nn.BatchNorm1d(channel)
        self.max_pooling = nn.MaxPool1d(kernel_size=pooling_kernel_size, stride=pooling_stride)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.activate(x)
        x = self.norm(x)
        x = self.max_pooling(x)
        x = self.dropout(x)
        return x


class ConvBlock_2(nn.Module):
    def __init__(self, channel_in=149, channel=149, kernel_size=21, stride=1, pooling_kernel_size=2, pooling_stride=2, padding=10):
        super().__init__()
        self.conv = nn.Conv1d(channel_in, channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activate = nn.ReLU()
        self.norm = nn.BatchNorm1d(channel)
        self.max_pooling = nn.MaxPool1d(kernel_size=pooling_kernel_size, stride=pooling_stride)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.activate(x)
        x = self.norm(x)
        x = self.max_pooling(x)
        x = self.dropout(x)
        return x


class ConvBlock_3(nn.Module):
    def __init__(self, channel_in=149, channel=149, kernel_size=51, stride=1, pooling_kernel_size=2, pooling_stride=2, padding=25):
        super().__init__()
        self.conv = nn.Conv1d(channel_in, channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activate = nn.ReLU()
        self.norm = nn.BatchNorm1d(channel)
        self.max_pooling = nn.MaxPool1d(kernel_size=pooling_kernel_size, stride=pooling_stride)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.activate(x)
        x = self.norm(x)
        x = self.max_pooling(x)
        x = self.dropout(x)
        return x


class ConvBlock_4(nn.Module):
    def __init__(self, channel_in=149, channel=149, kernel_size=75, stride=1, pooling_kernel_size=2, pooling_stride=2, padding=37):
        super().__init__()
        self.conv = nn.Conv1d(channel_in, channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activate = nn.ReLU()
        self.norm = nn.BatchNorm1d(channel)
        self.max_pooling = nn.MaxPool1d(kernel_size=pooling_kernel_size, stride=pooling_stride)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.activate(x)
        x = self.norm(x)
        x = self.max_pooling(x)
        x = self.dropout(x)
        return x

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs_txt = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size), stride=(2, 1), padding=(k//2, 0)) for k in config.filter_sizes])
        self.convs_vocal = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.vocal_hidden), stride=(2, 1), padding=(k//2, 0)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.lstm = TwoLayerLSTM(config.input_size, config.lstm_hidden, config.num_classes, config.device)
        self.conv_1 = ConvBlock_1(768, 192)
        self.conv_2 = ConvBlock_2(768, 192)
        self.conv_3 = ConvBlock_3(768, 192)
        self.conv_4 = ConvBlock_4(768, 192)
        self.conv_5 = ConvBlock_1(1024, 256)
        self.conv_6 = ConvBlock_2(1024, 256)
        self.conv_7 = ConvBlock_3(1024, 256)
        self.conv_8 = ConvBlock_4(1024, 256)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = x.squeeze(2)
        return x

    def get_feature(self, x):
        context = x[0]
        mask = x[2]
        vocal_feature = x[3]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        encoder_out = encoder_out.permute(0, 2, 1)
        out_1_1 = self.conv_1(encoder_out)
        out_1_2 = self.conv_2(encoder_out)
        out_1_3 = self.conv_3(encoder_out)
        out_1_4 = self.conv_4(encoder_out)
        out_1 = torch.cat([out_1_1, out_1_2, out_1_3, out_1_4], dim=1)
        out_1 = out_1.permute(0, 2, 1)
        out_1 = out_1.unsqueeze(1)
        vocal_feature = vocal_feature.permute(0, 2, 1)
        out_2_1 = self.conv_5(vocal_feature)
        out_2_2 = self.conv_6(vocal_feature)
        out_2_3 = self.conv_7(vocal_feature)
        out_2_4 = self.conv_8(vocal_feature)
        out_2 = torch.cat([out_2_1, out_2_2, out_2_3, out_2_4], dim=1)
        out_2 = out_2.permute(0, 2, 1)
        out_2 = out_2.unsqueeze(1)
        out_1 = torch.cat([self.conv_and_pool(out_1, conv) for conv in self.convs_txt], 1)
        out_2 = torch.cat([self.conv_and_pool(out_2, conv) for conv in self.convs_vocal], 1)
        out = torch.cat([out_1, out_2], dim=2)

        return out

    def get_txt_feature(self, x):
        context = x[0]
        mask = x[2] 

        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)

        out_1 = encoder_out.unsqueeze(1)
        out_1 = torch.cat([self.conv_and_pool(out_1, conv) for conv in self.convs_txt], 1)

        return out

def get_audio_feature(self, x):
    context = x[0]  # Input sentence
    mask = x[2]  # Attention mask
    out_2 = x[3]  # Audio features
    out_2 = torch.cat([self.conv_and_pool(out_2, conv) for conv in self.convs_vocal], 1)

    # Add the extracted features to the list
    return out_2

    def forward(self, x):
        context = x[0]
        mask = x[2]
        vocal_feature = x[3]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        encoder_out = encoder_out.permute(0, 2, 1)
        out_1_1 = self.conv_1(encoder_out)
        out_1_2 = self.conv_2(encoder_out)
        out_1_3 = self.conv_3(encoder_out)
        out_1_4 = self.conv_4(encoder_out)
        out_1 = torch.cat([out_1_1, out_1_2, out_1_3, out_1_4], dim=1)
        out_1 = out_1.permute(0, 2, 1)
        out_1 = out_1.unsqueeze(1)
        vocal_feature = vocal_feature.permute(0, 2, 1)
        out_2_1 = self.conv_5(vocal_feature)
        out_2_2 = self.conv_6(vocal_feature)
        out_2_3 = self.conv_7(vocal_feature)
        out_2_4 = self.conv_8(vocal_feature)
        out_2 = torch.cat([out_2_1, out_2_2, out_2_3, out_2_4], dim=1)
        out_2 = out_2.permute(0, 2, 1)
        out_2 = out_2.unsqueeze(1)
        out_1 = torch.cat([self.conv_and_pool(out_1, conv) for conv in self.convs_txt], 1)
        out_2 = torch.cat([self.conv_and_pool(out_2, conv) for conv in self.convs_vocal], 1)
        out = torch.cat([out_1, out_2], dim=2)
        out = self.lstm(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        return out