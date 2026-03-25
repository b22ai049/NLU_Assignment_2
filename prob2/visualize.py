import matplotlib.pyplot as plt

# Data from your terminal output
epochs = [0, 10, 20, 30, 40]

# Replace these with the exact values from your terminal
vanilla_rnn_loss = [1.8423, 0.7580, 0.7513, 0.7506, 0.7501]
blstm_loss = [0.4042, 0.0000, 0.0000, 0.0000, 0.0000]
attention_loss = [2.2770, 0.8351, 0.6606, 0.5757, 0.5256]

plt.figure(figsize=(10, 6))

# Plotting the lines
plt.plot(epochs, vanilla_rnn_loss, label='Vanilla RNN', marker='o', linestyle='-')
plt.plot(epochs, blstm_loss, label='BLSTM (Overfitted)', marker='s', linestyle='--')
plt.plot(epochs, attention_loss, label='RNN + Attention', marker='^', linestyle='-.')

# Formatting the plot for a formal report
plt.title('Training Loss Convergence: Problem 2 Architectures', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Cross-Entropy Loss', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Save the figure for LaTeX
plt.savefig('data/problem2_loss_curves.png', dpi=300)
print("Plot saved as 'data/problem2_loss_curves.png'. Include this in your LaTeX report!")
plt.show()