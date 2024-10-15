import torch

# Function to train any model
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_samples = 0
        for inputs, labels in train_dataloader:
            # inputs.shape Should be (batch_size, channels, depth, height, width)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(inputs.size(0))
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0) # batch size
            print(f'Batch Loss: {loss.item():.4f}')

        print(f"Train epoch loss = {running_loss} divided by {total_samples}")
        epoch_loss = running_loss / total_samples
        print(f'Epoch {epoch}/{num_epochs - 1}, Training Loss: {epoch_loss:.4f}')
        train_losses.append(epoch_loss)
        # Validate the model
        val_losses.append(validate_model(model, val_dataloader, criterion))
    print('Training complete')
    return model, train_losses, val_losses

# a new train model function that also writes the predicted and ground truth values to a file
def train_model_with_logging(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, log_file_p):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_samples = 0
        for inputs, labels in train_dataloader:
            # inputs.shape Should be (batch_size, channels, depth, height, width)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0) # batch size
            # print(f'Batch Loss: {loss.item():.4f}')
            # Write the predicted and ground truth values to a csv file, output is dimension 6 , p_idx, s_idx, no_ps_conf, p_conf, s_conf, both_ps_conf
            with open(log_file_p, 'a') as log_file:
                # first write titles of 12 columns
                log_file.write("p_idx_gt, p_idx_pred, s_idx_gt, s_idx_pred, no_ps_conf_gt, no_ps_conf_pred, p_conf_gt, p_conf_pred, s_conf_gt, s_conf_pred, both_ps_conf_gt, both_ps_conf_pred\n")
                for i in range(inputs.size(0)):
                    log_file.write(f"{labels[i, 0]}, {outputs[i, 0]}, {labels[i, 1]}, {outputs[i, 1]}, {labels[i, 2]}, {outputs[i, 2]}, {labels[i, 3]}, {outputs[i, 3]}, {labels[i, 4]}, {outputs[i, 4]}, {labels[i, 5]}, {outputs[i, 5]}\n")
        epoch_loss = running_loss / total_samples
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
        total_samples = 0
        for inputs, labels in val_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0) # batch size
        
        epoch_loss = running_loss / total_samples
        print(f'Validation Loss: {epoch_loss:.4f}')
    model.train()
    return epoch_loss