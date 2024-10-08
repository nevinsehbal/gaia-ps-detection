# Import the ConvLSTM class, implementation available at: precipitation-nowcasting/ConvLSTM
from baseconvlstmcopy import ConvLSTM
import torch
import torch.nn as nn

# ConvLSTM Model for Seismic Data
class  SeismicConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, output_dim, num_windows, data_points):
        super(SeismicConvLSTM, self).__init__()
        self.convlstm = ConvLSTM(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size,
                                 num_layers=num_layers, batch_first=True, bias=True, return_all_layers=False)
        self.fc1 = nn.Linear(hidden_dim[-1] * num_windows * 1 * data_points, 512)  # Adding a new intermediate layer
        self.fc2 = nn.Linear(512, output_dim)  # Existing final output layer
    
    def forward(self,x):
        # x: [batch, sequence_length, channels, height, width]
        batch_size = x.size(0)
        
        # Forward through ConvLSTM
        layer_output_list, last_state_list = self.convlstm(x)
        
        # Get the output from the last ConvLSTM layer
        convlstm_out = layer_output_list[-1]  # [batch, sequence_length, hidden_dim, height, width]
        
        # Flatten the ConvLSTM output
        convlstm_out = convlstm_out.view(batch_size, -1)
        
        # Pass through fully connected layers
        x = torch.relu(self.fc1(convlstm_out))  # Apply ReLU activation
        output = self.fc2(x)
        # Assume output is [batch_size, 4] where [P_index, S_index, P_confidence, S_confidence]
        predicted_indices = output[:, :2]  # Linear for indices
        predicted_confidence = torch.sigmoid(output[:, 2:])  # Sigmoid for confidence
        
        return torch.cat([predicted_indices, predicted_confidence], dim=1)