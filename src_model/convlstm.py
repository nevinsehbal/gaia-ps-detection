# Import the ConvLSTM class, implementation available at: precipitation-nowcasting/ConvLSTM
from baseconvlstm import ConvLSTM, ModifiedConvLSTM
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
    
# ConvLSTM Model for Seismic Data
class ModifiedSeismicConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_sizes, strides, paddings, num_layers, output_dim, num_windows, data_points):
        super(ModifiedSeismicConvLSTM, self).__init__()
        
        # Initialize ConvLSTM with variable kernel sizes, strides, paddings, and hidden dimensions
        self.convlstm = ModifiedConvLSTM(
            input_dim=input_dim, 
            hidden_dim=hidden_dims, 
            kernel_size=kernel_sizes, 
            stride=strides,  
            padding=paddings,  # Padding added
            num_layers=num_layers, 
            batch_first=True, 
            bias=True, 
            return_all_layers=False
        )
        self.num_windows = num_windows
        self.data_points = data_points
        self.hidden_dims = hidden_dims

        # Adding a fully connected layer after ConvLSTM layers
        # self.fc1 = nn.Linear(hidden_dims[-1] * num_windows * 1 * data_points, 512)  # Adjusting dimensions according to hidden_dims[-1]
        # self.fc1 = nn.Linear(345600, 512)  # Adjusting dimensions according to hidden_dims[-1]
        # print("FC1 input size = hidden_dims[-1] * num_windows * 1 * data_points = ", hidden_dims[-1] * num_windows * 1 * data_points)
        # print("Hidden dims[-1]: ", hidden_dims[-1])
        # print("Num windows: ", num_windows)
        # print("Data points: ", data_points)
        self.fc1 = None
        self.fc2 = nn.Linear(512, output_dim)  # Final output layer
    
    def forward(self, x):
        # x: [batch, sequence_length, channels, height, width]
        batch_size = x.size(0)
        # print(f'Input shape: {x.shape}')
        
        # Forward through ConvLSTM
        layer_output_list, last_state_list = self.convlstm(x)
        
        # Get the output from the last ConvLSTM layer
        convlstm_out = layer_output_list[-1]  # [batch, sequence_length, hidden_dim, height, width]
        # print(f'ConvLSTM output shape: {convlstm_out.shape}')

        # Flatten the ConvLSTM output
        convlstm_out = convlstm_out.view(batch_size, -1)
        # print(f'Flattened ConvLSTM output shape: {convlstm_out.shape}')


        if self.fc1 is None:
            self.fc1 = nn.Linear(self.hidden_dims[-1] * self.num_windows * 1 * self.data_points, 512)
            print(f'Initialized FC1 with input size: {convlstm_out.size(1)}')

        # Pass through fully connected layers
        x = torch.relu(self.fc1(convlstm_out))  # Apply ReLU activation
        # print(f'After FC1 shape: {x.shape}')
        output = self.fc2(x)
        # print(f'Final output shape: {output.shape}')
        
        # Assume output is [batch_size, 4] where [P_index, S_index, P_confidence, S_confidence]
        predicted_indices = output[:, :2]  # Linear for indices
        predicted_confidence = torch.sigmoid(output[:, 2:])  # Sigmoid for confidence
        
        return torch.cat([predicted_indices, predicted_confidence], dim=1)


# Seismic ConvLSTM Model with 5 Conv2D layers
# import torch
# import torch.nn as nn

# class SeismicConvLSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, output_dim, num_windows, data_points):
#         super(SeismicConvLSTM, self).__init__()
        
#         # ConvLSTM
#         self.convlstm = ConvLSTM(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size,
#                                  num_layers=num_layers, batch_first=True, bias=True, return_all_layers=False)
        
#         # # Convolutional block: 5 Conv2d layers
#         # self.conv_block = nn.Sequential(
#         #     nn.Conv2d(hidden_dim[-1], 64, kernel_size=3, stride=1, padding=1),  # 1st Conv2D
#         #     nn.BatchNorm2d(64),
#         #     nn.ReLU(),
#         #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 2nd Conv2D
#         #     nn.BatchNorm2d(128),
#         #     nn.ReLU(),
#         #     nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 3rd Conv2D
#         #     nn.BatchNorm2d(256),
#         #     nn.ReLU(),
#         #     nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # 4th Conv2D
#         #     nn.BatchNorm2d(512),
#         #     nn.ReLU(),
#         #     nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),  # 5th Conv2D
#         #     nn.BatchNorm2d(1024),
#         #     nn.ReLU()
#         # )
        
#         # Fully connected layers
#         self.fc1 = nn.Linear(1024 * num_windows * 1 * data_points, 512)  # Adjust based on conv output
#         self.fc2 = nn.Linear(512, output_dim)
    
#     def forward(self, x):
#         batch_size = x.size(0)
        
#         # Forward through ConvLSTM
#         layer_output_list, last_state_list = self.convlstm(x)
        
#         # Get the output from the last ConvLSTM layer
#         convlstm_out = layer_output_list[-1]  # [batch, sequence_length, hidden_dim, height, width]
        
#         # Forward through the convolutional block
#         conv_block_out = self.conv_block(convlstm_out[:, -1, :, :, :])  # Use the last sequence output of ConvLSTM
        
#         # Flatten the conv block output
#         conv_block_out = conv_block_out.view(batch_size, -1)
        
#         # Pass through fully connected layers
#         x = torch.relu(self.fc1(conv_block_out))
#         output = self.fc2(x)
        
#         # Output [batch_size, 4] where [P_index, S_index, P_confidence, S_confidence]
#         predicted_indices = output[:, :2]  # Linear for indices
#         predicted_confidence = torch.sigmoid(output[:, 2:])  # Sigmoid for confidence
        
#         return torch.cat([predicted_indices, predicted_confidence], dim=1)

