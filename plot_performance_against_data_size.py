import os
import json
import matplotlib
matplotlib.use('TkAgg') #uncomment for Mac users (maybe)
import matplotlib.pyplot as plt

if __name__ == '__main__':
    missing_metrics_paths = []
    training_commands = []
    seq2vec_choices = ["dan", "gru"]
    data_sizes = ["5k", "10k", "15k"]
    validation_accuracies = {"dan": [], "gru": []}
    for seq2vec_choice in seq2vec_choices:
        epochs = 8 if seq2vec_choice == "dan" else 4 # gru is slow, use only 4 epochs
        for size in data_sizes:

            serialization_dir = os.path.join("serialization_dirs",
                                             f"main_{seq2vec_choice}_{size}_with_emb")

            metrics_path = os.path.join(serialization_dir, "metrics.json")
            if not os.path.exists(metrics_path):
                training_command = (f"python train.py main "
                                    f"data/imdb_sentiment_train_{size}.jsonl "
                                    f"data/imdb_sentiment_dev.jsonl "
                                    f"--seq2vec-choice {seq2vec_choice} "
                                    f"--embedding-dim 50 "
                                    f"--num-layers 4 "
                                    f"--num-epochs {epochs} "
                                    f"--suffix-name _{seq2vec_choice}_{size}_with_emb "
                                    f"--pretrained-embedding-file data/glove.6B.50d.txt ")
                training_commands.append(training_command)
                missing_metrics_paths.append(metrics_path)
            else:
                with open(metrics_path) as file:
                    metrics = json.load(file)
                validation_accuracies[seq2vec_choice].append(metrics["best_epoch_validation_accuracy"])

    if missing_metrics_paths:
        print("\nFollowing metrics could not be found:")
        print("\n".join(missing_metrics_paths))

        print("\nBefore generate plot, you will need to run the "
              "following training commands to generate them:")
        print("\n".join(training_commands))
        exit()

    # Make the plots
    plt.style.use('seaborn-whitegrid')
    fig, axs = plt.subplots(2, 1, figsize=(8,6), sharex=False, sharey=False, constrained_layout=False)
    for ax, seq2vec_choice in zip(axs.flat, seq2vec_choices):
        ax.plot(data_sizes, validation_accuracies[seq2vec_choice])
        ax.set_xlabel("Training Data Used")
        ax.set_ylabel("Validation Accuracy")
        title = f"{seq2vec_choice.upper()}+glove: Best validation accuracies vs Training data used"
        ax.set_title(title)
    
    fig.subplots_adjust(hspace = 0.5)
    
    plt.savefig(os.path.join("plots", f"performance_against_data_size_with_glove.png"))
    plt.clf()
