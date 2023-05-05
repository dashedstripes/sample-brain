import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# The wavenet network has the following: 
# 3 layers (what are layers?)
# 2 blocks (what are blocks?)
# dilation_base is set to 2 (what is a dilation base?)

# the sequence length is then padded in order to make it a power of 2
# padding is then set to 0?

# we create a 1 dimensional convolutional layer with 1 channel, 32 filters, with a kernel size of 1

# we then create a list of residual blocks (what's a residual block?)

# we loop over number of blocks we have set
# for each block, we set a dilation value equal to the dilation base value to the exponent of the current index value of the block
# then, for each layer:
# we set an "in_filters" variables to the number of filters, and an "out_filters" variable to the number of filters
# finally, we append a residual block to the list of residual blocks
# we use the in+out filters, the kernal size, dilation, and padding as input params

# this means we have 2 residual blocks with these properties.

# next, we set a ReLI
# finally, we create a final 1d convolution filter that takes in 32 filters, over 1 channel, with a kernal size of 1.




class WaveNet(nn.Module):
    def __init__(self, sequence_length, n_channels=1, n_layers=10, n_blocks=2, n_filters=2, kernel_size=2, dilation_base=2):
        super(WaveNet, self).__init__()
        self.n_layers = n_layers
        self.n_blocks = n_blocks
        self.dilation_base = dilation_base
        
        # Determine the padding required to make the sequence length a power of 2
        sequence_length = 2 ** math.ceil(math.log2(sequence_length))
        padding = sequence_length - sequence_length
        
        # Input convolution layer
        self.conv_in = nn.Conv1d(n_channels, n_filters, kernel_size=1)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for b in range(n_blocks):
            dilation = dilation_base ** b
            for l in range(n_layers):
                in_filters = n_filters
                out_filters = n_filters
                self.res_blocks.append(ResidualBlock(in_filters, out_filters, kernel_size, dilation, padding))
        
        # Output layers
        self.relu = nn.ReLU()
        self.conv_out = nn.Conv1d(n_filters, n_channels, kernel_size=1)
        
# this is the forward function:
# the forward function takes in our input tensor
# it passes it to "conv_in", which is the 1d convolution vector we set in __init__
# we then generate a list of skip_connections (what's a skip connection?)
# we loop over the length of the number of residual blocks
# for each block, we grab the current block and set it to "res_block"
# we then pass the current value of x (our input tensor passed through conv_in) to the res_block
# this returns both x, and skip (what's skip?)
# we then append skip to our list of skip connections

    def forward(self, x):
        x = self.conv_in(x)
        
        # Residual blocks
        skip_connections = []
        for i in range(len(self.res_blocks)):
            res_block = self.res_blocks[i]
            x, skip = res_block(x)
            skip_connections.append(skip)
        
        # we defined a "max_length" var that is the minimum value of the array of skip connections?
        # Sum the skip connections and pass through activation
        max_length = min([len(s) for s in skip_connections])
        # for each element in skip connections, truncate the length of the third dimension to "max_length"
        skip_connections = [s[:, :, :max_length] for s in skip_connections]

        # add up all the skip connections
        x = sum(skip_connections)

        # run x through a relu (turns negative numbers to 0)
        x = self.relu(x)
        
        # run x through conv_out, which is another 1d convolutional filter
        # Output layers
        x = self.conv_out(x)

        # return the logits
        return x


# this is the residual block we've heard so much about, (what is a residual block?)
class ResidualBlock(nn.Module):
    
    # we define a bunch of layers
    def __init__(self, in_filters, out_filters, kernel_size, dilation, padding):
        super(ResidualBlock, self).__init__()
        # conv_dilated is a 1d convolutional layer that takes in filters, kernal size and dilation, it also adds some padding (// is floor division in python) the padding is an integer value
        self.conv_dilated = nn.Conv1d(in_filters, out_filters, kernel_size=kernel_size, dilation=dilation, padding=(kernel_size - 1) * dilation // 2 + padding // 2)

        # conv_gate is identical to conv_dilated
        self.conv_gate = nn.Conv1d(in_filters, out_filters, kernel_size=kernel_size, dilation=dilation, padding=(kernel_size - 1) * dilation // 2 + padding // 2)

        # sigmoid maps input values to output values between 0 and 1, smoothly, S shape curve
        self.sigmoid = nn.Sigmoid()
        # tanh maps input values to output values between 0 and 1
        self.tanh = nn.Tanh()
        
    
    # this is the forward step through our residual block, all of the values here 
    # will be updated during back propogation
    def forward(self, x):
        # run x through the conv_dilated 1d filter
        conv_dilated = self.conv_dilated(x)
        # run x through the conv_gate 1d filter (not sure why there are 2 filters? see next step)
        conv_gate = self.conv_gate(x)
        # we then run the ouput of the conv_gate through the sigmoid
        # we multiply this value by the result of conv_dilated
        conv_gated = conv_dilated * self.sigmoid(conv_gate)
        # we store the value of conv_gated for later
        skip = conv_gated

        # finally, i don't really get this line ngl
        res = x[:, :, -conv_gated.size(2):] + conv_gated
        return res, skip
