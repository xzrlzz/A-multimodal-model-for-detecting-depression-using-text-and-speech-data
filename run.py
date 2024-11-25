import os
import time
import torch
import torchaudio
import torchaudio.transforms as AT
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from sklearn.model_selection import train_test_split
from importlib import import_module
import argparse
import numpy as np


def load_and_process_data(data_path, processor, model, save_path, train_path, test_path):
    """
    Load and process audio and text data, generate training and testing sets, and save them to the specified paths.

    :param data_path: Path to the raw data
    :param processor: Wav2Vec2Processor instance
    :param model: Wav2Vec2Model instance
    :param save_path: Path to save features
    :param train_path: Path to save the training set text
    :param test_path: Path to save the testing set text
    """
    if os.path.exists(train_path) and os.path.exists(test_path):
        return  # If training and testing files already exist, return immediately

    captions, labels, vocal_features = [], [], []
    resampler = AT.Resample(orig_freq=48000, new_freq=16000)  # Create a resampler

    for category in os.listdir(data_path):
        category_path = os.path.join(data_path, category)
        for subject in os.listdir(category_path):
            subject_path = os.path.join(category_path, subject)
            label = 0 if category == 'healthy' else 1  # Set label based on category

            for file in os.listdir(subject_path):
                if file.endswith('.txt'):
                    with open(os.path.join(subject_path, file), 'r', encoding='utf-8') as f:
                        caption = f.read().replace(' ', '').replace('\n', '').replace('\r',
                                                                                      '')  # Read and clean the text
                        captions.append(f"{caption} {label}")
                        labels.append(label)

                if file.endswith('.wav'):
                    waveform, _ = torchaudio.load(os.path.join(subject_path, file))  # Load audio file
                    waveform = resampler(waveform).mean(dim=0, keepdim=True)  # Resample and average channels

                    if waveform.size(-1) != 320000:
                        waveform = torch.nn.functional.pad(waveform, (0, 320000 - waveform.size(-1)),
                                                           mode='constant').squeeze(0)  # Pad or trim the audio length

                    input_values = processor(waveform.numpy(), return_tensors="pt", padding=True,
                                             sampling_rate=16000).input_values  # Process input
                    with torch.no_grad():
                        features = model(input_values).last_hidden_state.squeeze(
                            0).cpu()  # Extract features and move back to CPU
                        vocal_features.append(features)

    X_train, X_test, vocal_train, vocal_test = train_test_split(captions, vocal_features, test_size=0.2,
                                                                stratify=labels, random_state=42)  # Split the data

    with open(train_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(X_train))  # Save the training set text

    with open(test_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(X_test))  # Save the testing set text

    torch.save(vocal_train, os.path.join(save_path, 'vocal_train.pth'))  # Save the training set features
    torch.save(vocal_test, os.path.join(save_path, 'vocal_test.pth'))  # Save the testing set features


def main():
    """
    Main function responsible for parsing command-line arguments, configuring the model, loading data, and training the model.
    """
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    parser.add_argument('--model', type=str, default='bert_Bi-lstm', help='Choose a model: Bert, ERNIE')
    args = parser.parse_args()

    dataset = './THUCNews'  # Path to the dataset
    model_name = args.model  # Model name
    x = import_module('models.' + model_name)  # Dynamically import the model module
    config = x.Config(dataset)  # Configure the model

    data_path = './raw_data/data/healthy'  # Path to the raw data
    save_path = './THUCNews/data'  # Path to save features
    train_path = os.path.join(save_path, 'CMDD_train_data.txt')  # Path to save the training set text
    test_path = os.path.join(save_path, 'CMDD_test_test.txt')  # Path to save the testing set text

    processor = Wav2Vec2Processor.from_pretrained("./large_model/wav2vec 2.0")  # Load Wav2Vec2 processor
    model = Wav2Vec2Model.from_pretrained(
        "./large_model/wav2vec 2.0").eval()  # Load and set Wav2Vec2 model to evaluation mode

    load_and_process_data(data_path, processor, model, save_path, train_path, test_path)  # Load and process data

    if os.path.exists(train_path) and os.path.exists(test_path):
        train_data, test_data, vocal_train, vocal_test = build_dataset(config)  # Build the dataset
        train_iter = build_iterator(train_data, config, vocal_train)  # Build the training data iterator
        test_iter = build_iterator(test_data, config, vocal_test)  # Build the testing data iterator

        model = x.Model(config).to(config.device)  # Initialize and move the model to the device
        train(config, model, train_iter, test_iter, test_iter)  # Train the model
        print("Training completed.")  # Print training completion message


if __name__ == '__main__':
    start_time = time.time()  # Record the start time
    main()  # Run the main function
    print("Time usage:", time.time() - start_time)  # Print the total time taken
