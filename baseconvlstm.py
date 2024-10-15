import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
 
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
 
        super(ConvLSTMCell, self).__init__()
 
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
 
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2 # Ensure that (h,w) remains unchanged during the transmission process
        self.bias = bias
        # Conv2d layer is defined with:
        # input channels as the sum of input_dim and hidden_dim.
        # Output channels are 4 times the hidden_dim (for i, f, g, o gates).
        # Kernel size is the user-defined kernel size.
        # Padding is defined as above. 
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim, # i gate, f gate, o gate, g gate are calculated together, and then split
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias,)
    
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state # Each timestamp contains two state tensors: h and c

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis 

        combined_conv = self.conv(combined) # i gate, f gate, o gate, g gate are calculated together, and then split
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g # c state tensor update
        h_next = o * torch.tanh(c_next) # h state tensor update

        return h_next, c_next # Output two state tensors of the current timestamp
    
    def init_hidden(self, batch_size, image_size):
        """
        Initial state tensor initialization. The state tensor 0 of the first timestamp is initialized
        :param batch_size:
        :param image_size:
        :return:
        """
        height, width = image_size
        init_h = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        init_c = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        return (init_h,init_c)
    
class ConvLSTM(nn.Module):
 
    """
    Parameters: parameter introduction
    @input_dim: the number of channels of the input tensor
    @hidden_dim: the number of channels of the two state tensors h and c, which can be a list
    @kernel_size: the size of the convolution kernel. By default, the convolution kernel size of all layers is the same. 
                You can also set the convolution kernel size of different lstm layers to be different
    @num_layers: the number of convolution layers, which needs to be equal to len(hidden_dim)
    @batch_first: Whether or not dimension 0 is the batch or not
    @bias: Bias or no bias in Convolution
    @return_all_layers: whether to return the h state of all lstm layers
    Note: the same convolution kernel size, the same padding size
    
    Input: input introduction
    A tensor of size [B, T, C, H, W] or [T, B, C, H, W]# needs to be 5-dimensional
    
    Output: output introduction
    Two lists are returned: layer_output_list, last_state_list
    List 0: layer_output_list--single-layer list, each element represents the output h state of a LSTM layer, each element's size = [B, T, hidden_dim, H, W]
    List 1: last_state_list--double-layer list, each element is a binary list [h, c], representing the output state of the 
            last timestamp of each layer [h, c], h.size = c.size = [B, hidden_dim, H, W]
    A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
    0 - layer_output_list is the list of lists of length T of each output
    1 - last_state_list is the list of last states
    each element of the list is a tuple (h, c) for hidden state and memory
    Example: Usage example
    >> x = torch.rand((32, 10, 64, 128, 128))
    >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
    >> _, last_states = convlstm(x)
    >> h = last_states[0][0] # 0 for layer index, 0 for h index

    """
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()
 
        self._check_kernel_size_consistency(kernel_size)
 
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers) # Convert to a list
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers) # Convert to a list
        if not len(kernel_size) == len(hidden_dim) == num_layers: # Determine consistency
            raise ValueError('Inconsistent list length.')
 
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
 
        cell_list = []
        for i in range(0, self.num_layers): # Multi-layer LSTM settings
            # Input dimension of the current LSTM layer
            # if i==0:
            # cur_input_dim = self.input_dim
            # else:
            # cur_input_dim = self.hidden_dim[i - 1]
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1] # Equivalent to above
            if i == 0:
                print("Layer 1")
                cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                              hidden_dim=self.hidden_dim[i],
                                              kernel_size=(11,1),
                                              bias=self.bias))
            elif i == 1:
                print("Layer 2")
                cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                              hidden_dim=self.hidden_dim[i],
                                              kernel_size=(5,1),
                                              bias=self.bias))
            elif i == 2:
                print("Layer 3")
                cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                              hidden_dim=self.hidden_dim[i],
                                              kernel_size=(3,1),
                                              bias=self.bias))
            elif i == 3:
                print("Layer 4")
                cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                              hidden_dim=self.hidden_dim[i],
                                              kernel_size=(3,1),
                                              bias=self.bias))
            elif i == 4:
                print("Layer 5")
                cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                              hidden_dim=self.hidden_dim[i],
                                              kernel_size=(3,1),
                                              bias=self.bias))
            else:
                print("Error: Number of layers must be equal to 5")



            # cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
            #                               hidden_dim=self.hidden_dim[i],
            #                               kernel_size=self.kernel_size[i],
            #                               bias=self.bias))
 
        self.cell_list = nn.ModuleList(cell_list) # Connect multiple LSTM layers into a network model
    
    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: 5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
 
        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            b, _, _, h, w = input_tensor.size()  # Automatically obtain b,h,w information
            hidden_state = self._init_hidden(batch_size=b,image_size=(h, w))
 
        layer_output_list = []
        last_state_list = []
 
        seq_len = input_tensor.size(1) # Get the length of lstm based on the input tensor
        cur_layer_input = input_tensor
 
        for layer_idx in range(self.num_layers): # Calculate layer by layer
 
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len): # Calculate stamp by stamp
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],cur_state=[h, c])
                output_inner.append(h) # Output state of the tth stamp of the layer_idx layer
 
            layer_output = torch.stack(output_inner, dim=1) # Output state concatenation of all stamps of the layer_idx layer
            cur_layer_input = layer_output # Prepare the input tensor of the layer_idx+1 layer
 
            layer_output_list.append(layer_output) # Concatenation of the h state of all timestamps of the current layer
            last_state_list.append([h, c]) # The output state of the last stamp of the current layer [h,c]
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        """
        Initialize the input state of the first timestamp of all lstm layers to 0
        :param batch_size:
        :param image_size:
        :return:
        """
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states
 
    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        """
        Check whether the input kernel_size meets the requirements. The format of kernel_size is required to be list or tuple
        :param kernel_size:
        :return:
        """
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')
 
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        """
        Extended to multi-layer lstm case
        :param param:
        :param num_layers:
        :return:
        """
        if not isinstance(param, list):
            param = [param] * num_layers
        return param