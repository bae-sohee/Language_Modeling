import torch
from torch.utils.data import Dataset

class Shakespeare(Dataset):
    """ Shakespeare dataset
    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
		2) Make list of character indices using the dictionary.
		3) Split the data into chunks of sequence length 30. 
    """

    def __init__(self, input_file):
        with open(input_file, 'r') as f:
            self.data = f.read()

        # Creating a Character Dictionary
        self.chars = sorted(list(set(self.data)))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}
        
        # Creating a Character Index List
        self.data_idx = [self.char_to_idx[char] for char in self.data]

        # Set sequence length(30) and calculate data length
        self.seq_length = 30
        self.data_len = len(self.data_idx) - self.seq_length

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        # Split data into input and target sequences
        input_seq = self.data_idx[idx:idx+self.seq_length]
        target_seq = self.data_idx[idx+1:idx+self.seq_length+1]
        
        input_tensor = torch.zeros((self.seq_length, len(self.chars)), dtype=torch.float32)
        target_tensor = torch.tensor(target_seq, dtype=torch.long)

        for i, char_idx in enumerate(input_seq):
            input_tensor[i][char_idx] = 1.0
        
        return input_tensor, target_tensor

if __name__ == '__main__':
    dataset = Shakespeare("shakespeare.txt")
    print('Dataset name: Shakespeare')
    print(f"Dataset length: {len(dataset)}")
