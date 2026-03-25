import torch

# Load names from your generated file
def load_data(file_path="data/TrainingNames.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # Adding start (^) and end ($) tokens to each name 
            names = [f"^{line.strip().lower()}$" for line in f.readlines()]
        
        # Create vocabulary of unique characters
        chars = sorted(list(set(''.join(names))))
        char_to_int = {c: i for i, c in enumerate(chars)}
        int_to_char = {i: c for i, c in enumerate(chars)}
        vocab_size = len(chars)
        
        return names, char_to_int, int_to_char, vocab_size
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Ensure you ran the name generation script.")
        exit()

def name_to_tensor(name, char_to_int):
    """Converts a name string into a LongTensor of character indices."""
    tensor = torch.zeros(len(name)).long()
    for c in range(len(name)):
        tensor[c] = char_to_int[name[c]]
    return tensor