import torch
import torch.nn.functional as F


class Split_SFCN(torch.nn.Module):
    def __init__(self):
        super(Split_SFCN, self).__init__()
        
            #TODO: Initialize the required layers for this network
            #3 convolution layers with 7x7 kernels 

            self.layer1= torch.nn.Conv2d(in_channels=2, out_channels=18, kernel_size=7) 
            self.layer2= torch.nn.Conv2d(in_channels=18, out_channels=18, kernel_size=7) 
            self.layer3= torch.nn.Conv2d(in_channels=18, out_channels=18, dilation=2, kernel_size=7) 
        
    def forward(self, x):
        
        #TODO: Implement Forward Functionality
        #applying activation function on each layer
        c1=F.relu(self.layer1(x))  
        c2=F.relu(self.layer2(c1))
        c3=F.relu(self.layer3(c2))
        return c3
        #pass

#we have 5 blocks
class Split_RPN(torch.nn.Module):
    def __init__(self,b_count): #b_count is the current block number and also the parameter of this class
        super(Split_RPN, self).__init__()
            
            #TODO: Initialize the required layers for this network
            #3 convolutional layers
            #1x2 max pooling layer
            #1x1 projection pooling for top and bottom branch
           
            self.b_count=b_count #block count starting from 0
            self.b_inputs=[18,55,55,55,55] #the first block gets 18 input channels, the rest are set manually
           
            #the in_channels will differ since only the first block gets to have 18 inputs
            self.conv_2= torch.nn.Conv2d( in_channels=self.b_inputs[b_count], out_channels=6, kernel_size=7, dilation=2)
            self.conv_3= torch.nn.Conv2d( in_channels=self.b_inputs[b_count], out_channels=6, kernel_size=7, dilation=3)
            self.conv_4= torch.nn.Conv2d( in_channels=self.b_inputs[b_count], out_channels=6, kernel_size=7, dilation=4)
            
            self.conv1x2= torch.nn.MaxPool2d(kernel_size=(1,2)) #max pooling 1x2 
          
            #there are 2 paths for the output after this point 
            #bottom path: 1-d output meaning a single sheet/slice of outputs
            self.conv1x1_bot= torch.nn.Conv2d(in_channels=18, out_channels=1,kernel_size=7) 
            
            #top path mantaining the spatial size of input
            self.conv1x1_top= torch.nn.Conv2d(in_channels=18, out_channels=55,kernel_size=7)
            
    def forward(self, x):
        #to this point we have constructed the layers required to progress further
        #first concat the feature maps of the 3 layers 
        output= torch.cat([self.conv2(x), self.conv3(x), self.conv4(x)],dim=1)
        
        #then apply max pooling to the layer (of the first 3 blocks)
        if self.b_count<=2:
            output=self.conv1x2(output)
           
        #TOP BRANCH
        top_result= self.conv1x1_top(output) #convolution layer
        #top_result=  pls help i cant get the projection pool set up. How do i replace every input by its mean? i know that we have to use the .mean function for the mean 
        #but how do i "maintain the spatial size of the input"
            
        #BOTTOM BRANCH
        bot_result= self.conv1x1_bot(output) #convolution layer
        #bot_result= 
        bot_probabilties= torch.sigmoid(bot_result)
        
        block_output= torch.cat([top_result, output, bot_probabilities], dim=1)
        
        if self.b_count>=2: return(block_output, bot_probabilities) 
        else return(block_output, None)
        
        
        #pass

class Split_CPN(torch.nn.Module):
    def __init__(self,b_count): #here we need b_count as a parameter as well
        super(Split_CPN, self).__init__()        
        #TODO: Initialize the required layers for this network
        #NOTE: I COPY PASTED FROM ABOVE AND EDITED SOME THINGS AS THE CODE WAS ALMOST THE SAME
        #3 convolutional layers
        #1x2 max pooling layer
        #1x1 projection pooling for top and bottom branch

        #bcount cant be initialized since it will return to 1 everytime the function is called and the blocks will not be updated
        self.b_count=b_count #block count starting from 0
        self.b_inputs=[18,55,55,55,55] #the first block gets 18 input channels, the rest are set manually

        #the in_channels will differ since only the first block gets to have 18 inputs
        self.conv_2= torch.nn.Conv2d( in_channels=self.b_inputs[b_count], out_channels=6, kernel_size=7, dilation=2)
        self.conv_3= torch.nn.Conv2d( in_channels=self.b_inputs[b_count], out_channels=6, kernel_size=7, dilation=3)
        self.conv_4= torch.nn.Conv2d( in_channels=self.b_inputs[b_count], out_channels=6, kernel_size=7, dilation=4)

        self.conv2x1= torch.nn.MaxPool2d(kernel_size=(2,1)) #max pooling 2x1 

        #there are 2 paths for the output after this point 
        #bottom path: 1-d output meaning a single sheet/slice of outputs
        self.conv1x1_bot= torch.nn.Conv2d(in_channels=18, out_channels=1,kernel_size=7) 

        #top path mantaining the spatial size of input
        self.conv1x1_top= torch.nn.Conv2d(in_channels=18, out_channels=55,kernel_size=7)

    def forward(self, x):
        #TODO: Implement Forward Functionality
        #to this point we have constructed the layers required to progress further
        #first concat the feature maps of the 3 layers 
        output= torch.cat([self.conv2(x), self.conv3(x), self.conv4(x)],dim=1)
        
        #then apply max pooling to the layer (of the first 3 blocks)
        if self.b_count<=2:
            output=self.conv2x1(output)
           
        #TOP BRANCH
        top_result= self.conv1x1_top(output) #convolution layer
        #top_result=  pls help i cant get the projection pool set up. How do i replace every input by its mean? i know that we have to use the .mean function for the mean 
        #but how do i "maintain the spatial size of the input"
            
        #BOTTOM BRANCH
        bot_result= self.conv1x1_bot(output) #convolution layer
        #bot_result= 
        bot_probabilties= torch.sigmoid(bot_result)
        
        block_output= torch.cat([top_result, output, bot_probabilities], dim=1)
        
        if self.b_count>=2: return(block_output, bot_probabilities) 
        else: return(block_output, None)
        
        
        #pass

class SplitModel(torch.nn.Module):
    def __init__(self, infer=FALSE):
        super(SplitModel, self).__init__()   
        #TODO: Initialize SFCN, RPNs and CPNs
        
        self.sfcn = Split_SFCN()
        
        #we have 5 blocks and so we need 5 diff attributes
        self.b0RPN= Split_RPN(b_count=0)
        self.b1RPN= Split_RPN(b_count=1)
        self.b2RPN= Split_RPN(b_count=2)
        self.b3RPN= Split_RPN(b_count=3)
        self.b4RPN= Split_RPN(b_count=4)
        
        self.b0CPN= Split_CPN(b_count=0)
        self.b1CPN= Split_CPN(b_count=1)
        self.b2CPN= Split_CPN(b_count=2)
        self.b3CPN= Split_CPN(b_count=3)
        self.b4CPN= Split_CPN(b_count=4)
      
    def forward(self, x):        
        #TODO: Implement Forward Functionality
        x=self.sfcn(x)
        #5 blocks for each RPN and CPN to run
        output_rpn = self.b0RPN(x)
        output_rpn = self.b1RPN(output_rpn)
        output_rpn, prob1_rpn = self.b2RPN(output_rpn)
        output_rpn, prob2_rpn = self.b3RPN(output_rpn)
        output_rpn, prob3_rpn = self.b4RPN(output_rpn)
        
        output_cpn = self.b0CPN(x)
        output_cpn = self.b1CPN(output_cpn)
        output_cpn, prob1_cpn = self.b2CPN(output_cpn)
        output_cpn, prob2_cpn = self.b2CPN(output_cpn)
        output_cpn, prob3_cpn = self.b2CPN(output_cpn)
        #collecting the result in a reformed format
        rpn= (prob1_rpn, prob2_rpn, prob3_rpn)
        cpn= (prob1_cpn, prob2_cpn, prob3_cpn)
         
        #we need only last block probabilities for inference hence we introduce a new boolean attribute by the name of infer
        
        if (infer): return(prob3_rpn, prob3_cpn)
        else: return(rpn, cpn)
        
        #pass
