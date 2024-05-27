import torch
import numpy as np
import argparse

from model import CharRNN, CharLSTM
from dataset import Shakespeare

def generate(model, seed_characters, temperature, idx_to_char, char_to_idx, length=100):
    """ Generate characters
    Args:
        model: trained model
        seed_characters: seed characters
		temperature: T
		args: other arguments if needed
    Returns:
        samples: generated characters
    """
    model.eval()
    samples = seed_characters
    device = next(model.parameters()).device
    input_seq = torch.tensor([char_to_idx[c] for c in seed_characters], dtype=torch.long).unsqueeze(0).to(device)
    
    hidden = model.init_hidden(1)

    if isinstance(hidden, tuple):
        hidden = (hidden[0].to(device), hidden[1].to(device))
    else:
        hidden = hidden.to(device)

    for _ in range(length):
        output, hidden = model(input_seq, hidden)
        output = output[-1] / temperature
        probabilities = torch.softmax(output, dim=0).detach().cpu().numpy()
        next_char_idx = np.random.choice(len(probabilities), p=probabilities)
        next_char = idx_to_char[next_char_idx]
        samples += next_char
        input_seq = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)

    return samples

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate text using a trained RNN or LSTM model.')
    parser.add_argument('--lstm', action='store_true', help='Use the LSTM model for generation')
    parser.add_argument('--rnn', action='store_true', help='Use the RNN model for generation')
    parser.add_argument('--seed', type=str, default='MENENIUS: ', help='Seed string to start text generation')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature parameter for text generation')
    parser.add_argument('--length', type=int, default=100, help='Length of the generated text')
    parser.add_argument('--output', type=str, default='output.txt', help='File to save the generated text')

    args = parser.parse_args()

    input_file = 'shakespeare.txt'
    dataset = Shakespeare(input_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(dataset.chars)
    embedding_size = 128
    hidden_size = 256
    num_layers = 3

    char_to_idx = dataset.char_to_idx
    idx_to_char = dataset.idx_to_char

    if args.lstm:
        model = CharLSTM(vocab_size, embedding_size, hidden_size, vocab_size, num_layers).to(device)
        model.load_state_dict(torch.load('best_char_lstm.pth', map_location=device))
        print("Using LSTM model for generation")
    elif args.rnn:
        model = CharRNN(vocab_size, embedding_size, hidden_size, vocab_size, num_layers).to(device)
        model.load_state_dict(torch.load('best_char_rnn.pth', map_location=device))
        print("Using RNN model for generation")
    else:
        raise ValueError("You must specify either --lstm or --rnn")

    generated_text = generate(model, args.seed, args.temperature, idx_to_char, char_to_idx, length=args.length)
    print(f"Generated text:\n{generated_text}")

    with open(args.output, 'a') as f:
        f.write(f"---------------------------------------------------------------------------\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Temperature: {args.temperature}\n\n")
        f.write(generated_text)
        f.write(f"\n---------------------------------------------------------------------------\n")

    print(f"Generated text saved to {args.output}")