import torch
import torch.nn.functional as F


class Split_SFCN(torch.nn.Module):
    def __init__(self):
        super(Split_SFCN, self).__init__()
            """
            TODO: Initialize the required layers for this network
            3 convolution layers with 7x7 kernels 
            """ 
            self.layer1= torch.nn.Conv2d(in_channels=2, out_channels=18, kernel_size=7) 
            self.layer2= torch.nn.Conv2d(in_channels=18, out_channels=18, kernel_size=7) 
            self.layer3= torch.nn.Conv2d(in_channels=18, out_channels=18, dilation=2, kernel_size=7) 
        
    def forward(self, x):
        """
        TODO: Implement Forward Functionality
        activation functions
        """ #applying activation function on each layer
        c1=F.relu(self.layer1(x))  
        c2=F.relu(self.layer2(c1))
        c3=F.relu(self.layer3(c2))
        return c3
        #pass

#we have 5 blocks
class Split_RPN(torch.nn.Module):
    def __init__(self):
        super(Split_RPN, self).__init__()
            """
            TODO: Initialize the required layers for this network
            3 convolutional layers
            1x2 max pooling layer
            1x1 projection pooling for top and bottom branch
            """
            b_count=1
            self.b_count=b_count #block count starting from 1
            self.b_inputs=[18,55,55,55,55] #the first block gets 18 input channels, the rest are set manually 
            self.b_outputs=55 
            
            self.conv_2= torch.nn.Conv2d( in_channels=self.b_inputs[b_count-1], out_channels=6, kernel_size=7, dilation=2)
            #the in_channels will differ since only the first block gets to have 18 inputs
            self.conv_3= torch.nn.Conv2d( in_channels=self.b_inputs[b_count-1], out_channels=6, kernel_size=7, dilation=3)
            self.conv_4= torch.nn.Conv2d( in_channels=self.b_inputs[b_count-1], out_channels=6, kernel_size=7, dilation=4)
            
            
            
            
            
    def forward(self, x):
        """
        TODO: Implement Forward Functionality
        """
        pass

class Split_CPN(torch.nn.Module):
    def __init__(self):
        super(Split_CPN, self).__init__()
        """
        TODO: Initialize the required layers for this network
        """

    def forward(self, x):
        """
        TODO: Implement Forward Functionality
        """
        pass

class SplitModel(torch.nn.Module):
    def __init__(self):
        super(SplitModel, self).__init__()
        """
        TODO: Initialize SFCN, RPNs and CPNs
        """

    def forward(self, x):
        """
        TODO: Implement Forward Functionality
        """
        pass
