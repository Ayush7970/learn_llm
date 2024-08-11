import os
import lzma
from tqdm import tqdm
from multiprocessing import cpu_count
import concurrent.futures

def extract_text_and_update_vocab(args):
    directory, file_name, output_path, vocab_set = args
    file_path = os.path.join(directory, file_name)
    with lzma.open(file_path, "rt", encoding="utf-8") as infile:
        content = infile.read()
    with open(output_path, "a", encoding="utf-8") as outfile:
        outfile.write(content)
    return set(content)

def get_xz_files(directory):
    return [f for f in os.listdir(directory) if f.endswith(".xz") and os.path.isfile(os.path.join(directory, f))]

def process_files_parallel(file_list, dir_path, output_path):
    vocab_set = set()
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        args = [(dir_path, file_name, output_path, vocab_set) for file_name in file_list]
        for characters in tqdm(executor.map(extract_text_and_update_vocab, args), total=len(file_list)):
            vocab_set.update(characters)
    return vocab_set

data_directory = "enter the name of the directory"
train_output_path = "split_train.txt"
val_output_path = "val_train.txt"
vocab_output_path = "vocab.txt"

xz_files = get_xz_files(data_directory)
total_files = len(xz_files)

# Split files: 90% for training, 10% for validation
split_index = int(total_files * 0.9)
train_files = xz_files[:split_index]
val_files = xz_files[split_index:]

# Clear output files
open(train_output_path, 'w').close()
open(val_output_path, 'w').close()

# Process training files
train_vocab = process_files_parallel(train_files, data_directory, train_output_path)

# Process validation files
val_vocab = process_files_parallel(val_files, data_directory, val_output_path)

# Merge vocabularies and write to vocab file
combined_vocab = train_vocab.union(val_vocab)
with open(vocab_output_path, "w", encoding="utf-8") as vocab_file:
    for char in sorted(combined_vocab):
        vocab_file.write(char + '\n')
