import wandb
import torch
import sys
import os

# Add the src_dataset directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../src_dataset/')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../utils/')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../src_model/')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../src_loss/')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../')))
os.environ["PYTHONIOENCODING"] = "utf-8"

from cnn_lstm import CNN_LSTM
from trainer import train_model_with_logging
from dataloader import train_dataset, val_dataset
from customLoss import CustomLoss

# Sweep configuration
sweep_config = {
    "method": "grid",  # Can be 'random', 'grid', or 'bayes'
    "metric": {
        "name": "val_loss",  # Metric to optimize
        "goal": "minimize"   # Minimize validation loss
    },
    "parameters": {
        "epochs": {
            "values": [400, 600, 800]  # Define the number of epochs here
        },
        "lr": {
            "values": [0.001, 0.005, 0.0001,0.0005]
        },

        "hidden_size": {
            "values": [64, 128, 256]
        },
        "lambda_val": {
            "values": [0.3, 0.6, 1, 1.5]
        },
        "batch_size": {
            "values": [16,32, 64]
        }
    }
}

def train():
    # Initialize W&B
    wandb.init()

    # Access hyperparameters
    config = wandb.config

    # Setup model, optimizer, and loss
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    model = CNN_LSTM(input_channels=3, hidden_size=config.hidden_size, output_dim=6).to(device)
    criterion = CustomLoss(lambda_val=config.lambda_val)j
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Train the model
    trained_model, train_losses, val_losses = train_model_with_logging(
        model,
        train_dataloader,
        val_dataloader,
        criterion,
        optimizer,
        config.epochs
    )

    # Log metrics
    for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="cnn_lstm_project")

# Run the sweep
wandb.agent(sweep_id, function=train)
wandb.finish()