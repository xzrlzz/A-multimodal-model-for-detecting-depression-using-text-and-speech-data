from pydub.silence import split_on_silence
from pydub import AudioSegment
import os


def slice_and_concat_wav_audio(input_file, output_dir, j, segments_to_concat=5):
    """
    Slice a WAV audio file into segments and concatenate them in specified groups. If there are not enough segments,
    concatenate all valid segments.

    :param input_file: Path to the input WAV audio file (e.g., "his_one/input.wav")
    :param output_dir: Directory path to save the output files (e.g., "./output")
    :param segments_to_concat: Number of segments to concatenate each time, default is 5
    :return: Updated counter for the output file names
    """
    # Load the WAV audio file
    sound = AudioSegment.from_wav(input_file)
    dB_to_reduce = 5.0
    reduced_audio = sound - dB_to_reduce
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Split the audio into segments based on silence
    chunks = split_on_silence(
        reduced_audio,
        min_silence_len=300,  # Minimum silence length in milliseconds
        silence_thresh=-45,  # Silence threshold in dB
        keep_silence=300  # Keep silence of 300 ms before and after each segment
    )

    # Filter out invalid segments (too short or too long)
    chunks = [chunk for chunk in chunks if 100 < len(chunk) < 50000]
    print(f'Valid segments (greater than 2s and less than 10s): {len(chunks)}')

    # Concatenate segments
    concatenated_group = sum(chunks,
                             AudioSegment.empty()) if chunks else None  # If no valid segments, concatenated_group is None

    # Decide whether to concatenate in groups or export a single file
    if concatenated_group:  # Ensure there are valid segments
        if len(chunks) < segments_to_concat:
            # Not enough segments, concatenate all valid segments into one file
            concatenated_group.export(os.path.join(output_dir, f"concat_{j}.wav"), format="wav")
            j += 1
            print("All valid segments have been concatenated into a single file: concat_all.wav")
        else:
            # Enough segments, concatenate in specified groups
            concatenated_groups = []
            for i in range(0, len(chunks), segments_to_concat):
                group = chunks[i:i + segments_to_concat]
                if len(group) < 2 and i + segments_to_concat > len(chunks):
                    continue
                group_concat = sum(group, AudioSegment.empty())
                concatenated_groups.append(group_concat)

            # Export the concatenated audio files
            for i, concat_chunk in enumerate(concatenated_groups):
                concat_chunk.export(os.path.join(output_dir, f"concat_{j}.wav"), format="wav")
                j += 1

            print(f"WAV audio segments have been split and concatenated successfully, results located at: {output_dir}")
    else:
        print("No valid audio segments found to process.")
    return j


# Example usage
path = './raw_data/CMDD'
healthy_save_path = './raw_data/data/healthy'
depression_save_path = './raw_data/data/depression'
# vocal_path = 'E:\\data\\CMDD\\part4\\HC40'
healthy = ['part1', 'part2', 'part3', 'part5']
depression = ['part6', 'part7', 'part8']

# Initialize the counter for output file names
j = 0

# Process healthy audio files
for file in healthy:
    filename = os.path.join(path, file)
    for file_name in os.listdir(filename):
        file_path = os.path.join(filename, file_name)
        for name in os.listdir(file_path):
            if 'Q' in name and name.endswith('.wav'):
                name_path = os.path.join(file_path, name)
                j = slice_and_concat_wav_audio(name_path, os.path.join(healthy_save_path, file_name), j)

# Process depression audio files
for file in depression:
    filename = os.path.join(path, file)
    for file_name in os.listdir(filename):
        file_path = os.path.join(filename, file_name)
        for name in os.listdir(file_path):
            if 'Q' in name and name.endswith('.wav'):
                name_path = os.path.join(file_path, name)
                j = slice_and_concat_wav_audio(name_path, os.path.join(depression_save_path, file_name), j)

