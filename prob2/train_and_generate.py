import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from data_loader import load_data, name_to_tensor
from model_definitions import VanillaRNN, BLSTMModel, AttentionRNN

# Set hyperparameters
hidden_size = 128
lr = 0.001
epochs = 50

names, char_to_int, int_to_char, vocab_size = load_data()

def train_model(model, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training {model_name}...")
    for epoch in range(epochs):
        total_loss = 0
        for name in names:
            tensor = name_to_tensor(name, char_to_int)
            optimizer.zero_grad()
            
            # Input is characters 0 to N-1, Target is 1 to N
            input_tensor = tensor[:-1].unsqueeze(0)
            target_tensor = tensor[1:]
            
            output, _ = model(input_tensor, None)
            loss = criterion(output.squeeze(0), target_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(names):.4f}")

    # Generate 500 names
    generated = []
    model.eval()
with torch.no_grad():
    for _ in range(500):
        name = "^"
        hidden = None
        # Safeguard: Limit length to 20 to prevent infinite terminal scrolling
        while name[-1] != "$" and len(name) < 20:
            try:
                # Ensure name[-1] exists in your char_to_int mapping
                current_char_idx = char_to_int[name[-1]]
                input_char = torch.tensor([[current_char_idx]]).long()
                
                # Forward pass
                output, hidden = model(input_char, hidden)
                
                # Use a temperature scale (e.g., 0.8) to make generation more realistic
                # This helps avoid 'looping' characters (aaaaa)
                prob = F.softmax(output.squeeze() / 0.8, dim=-1).data
                
                # Sample the next character
                char_idx = torch.multinomial(prob, 1).item()
                name += int_to_char[char_idx]
                
            except Exception as e:
                # If a specific name fails, break the inner loop and move to next
                print(f"Generation error: {e}")
                break
                
        generated.append(name.strip("^$"))
    
    with open(f"data/{model_name}_generated.txt", "w") as f:
        f.write("\n".join(generated))

# Execute Training
train_model(VanillaRNN(vocab_size, hidden_size, vocab_size), "VanillaRNN")
train_model(BLSTMModel(vocab_size, hidden_size, vocab_size), "BLSTM")
train_model(AttentionRNN(vocab_size, hidden_size, vocab_size), "AttentionRNN")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def report_model_stats(model):
    # Total parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Size in MB (assuming float32)
    size_mb = (params * 4) / (1024**2)
    
    print(f"Vanilla RNN Parameters: {params}")
    print(f"Model Size: {size_mb:.4f} MB")

# Call this after initializing your VanillaRNN
# report_model_stats(rnn_model)

