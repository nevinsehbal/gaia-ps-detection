import torch

# Function to train any model
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_dataloader:
            # inputs.shape Should be (batch_size, channels, depth, height, width)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            print(f'Batch Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_dataloader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Training Loss: {epoch_loss:.4f}')
        train_losses.append(epoch_loss)
        # Validate the model
        val_losses.append(validate_model(model, val_dataloader, criterion))
    print('Training complete')
    return model, train_losses, val_losses

def validate_model(model, val_dataloader, criterion):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        for inputs, labels in val_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(val_dataloader.dataset)
        print(f'Validation Loss: {epoch_loss:.4f}')
    model.train()
    return epoch_loss