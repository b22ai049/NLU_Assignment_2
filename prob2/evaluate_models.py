def calculate_metrics(gen_file, train_file="data/TrainingNames.txt"):
    with open(train_file, "r") as f:
        train_names = set([line.strip().lower() for line in f.readlines()])
    with open(gen_file, "r") as f:
        gen_names = [line.strip().lower() for line in f.readlines()]
    
    unique_gen = set(gen_names)
    
    # Novelty: Percentage of generated names not in training set
    novel_count = sum(1 for name in unique_gen if name not in train_names)
    novelty_rate = (novel_count / len(gen_names)) * 100
    
    # Diversity: Unique generated names / total generated
    diversity = len(unique_gen) / len(gen_names)
    
    print(f"Metrics for {gen_file}:")
    print(f"  Novelty Rate: {novelty_rate:.2f}%")
    print(f"  Diversity: {diversity:.4f}\n")

calculate_metrics("data/VanillaRNN_generated.txt")
calculate_metrics("data/BLSTM_generated.txt")
calculate_metrics("data/AttentionRNN_generated.txt")