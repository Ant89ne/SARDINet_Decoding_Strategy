#############################################################
#                    IMPORT LIBRAIRIES                      #
#############################################################

#Pytorch librairies
import torch
from torch import nn

#############################################################
#                  SEPARABLE CONVOLUTIONS                   #
#############################################################

class SepConvBlock(nn.Module):
    """
    Class for the separable convolution block
    """

    def __init__(self, inputChannels, outputChannels, kernel) -> None:
        """
        Initialization

        @input inputChannels:       Number of channels as input
        @input outputChannels:      Number of willing output channels
        @input kernel:              Size of the convolutional kernels
        """

        super(SepConvBlock, self).__init__()

        #Dephtwise convolution : each channel is convolved separately and independently (groups argument)
        self.depthwiseConv = nn.Conv2d(inputChannels, inputChannels, kernel, 1,1, groups=inputChannels, padding_mode="reflect")
        #Pointwise convolution : along the channel dimension        
        self.pointwise = nn.Conv2d(inputChannels, outputChannels, 1)


    def forward(self, x):
        """
        Forward pass of the block

        @input x :      Data to be forwarded
        """
        ret = self.depthwiseConv(x)
        ret = self.pointwise(ret)

        return ret        

#############################################################
#                    DECODER BRANCHES                       #
#############################################################

class DecBranch(nn.Module):
    """
    Class for a single branch of the decoder of SARDNINet
    This decoder is the original SARDINet decoder containing post-upsampling convolutions
    """

    def __init__(self, decKernel) -> None:
        """
        Initialization

        @input decKernel :        Number of kernels for each layer including the number of channel in the input
        """
        super(DecBranch, self).__init__()
        
        #List of the layers of the branch
        self.layers = []     
        self.decKernel = decKernel   

        #Creation of the layers
        for k in range(2):
            self.layers.append(nn.Upsample(scale_factor=2, mode = "bilinear"))                               
            self.layers.append(nn.Conv2d(self.decKernel[k], self.decKernel[k+1], 3, 1, 1, padding_mode="reflect"))   
            self.layers.append(nn.Conv2d(self.decKernel[k+1], self.decKernel[k+1], 3, 1, 1, padding_mode="reflect"))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm2d(self.decKernel[k+1]))
        
        #Final layer block (no ReLU)
        self.layers.append(nn.Upsample(scale_factor=2, mode = "bilinear"))
        self.layers.append(nn.Conv2d(self.decKernel[-2], self.decKernel[-2], 3, 1, 1, padding_mode="reflect"))
        self.layers.append(nn.Conv2d(self.decKernel[-2], self.decKernel[-1], 3, 1, 1, padding_mode="reflect"))
        self.layers.append(nn.Sigmoid())
        
        #Create a sequential network to easy the forward pass
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        """
        Forward pass
        
        @input x :      Data to be forwarded
        """
        ret = self.layers(x)
        return ret

class DecTransposeBranch(nn.Module):
    """
    Class for a single branch of the decoder of SARDNINet
    This decoder is based on Transposed convolutions
    """

    def __init__(self, decKernel) -> None:
        """
        Initialization

        @input decKernel :        Number of kernels for each layer including the number of channel in the input
        """
        super(DecTransposeBranch, self).__init__()
        
        #List of the layers of the branch
        self.layers = []
        self.decKernel = decKernel

        #Creation of the layers
        for k in range(2):
            self.layers.append(nn.ConvTranspose2d(self.decKernel[k], self.decKernel[k+1], 3, stride = 2, padding = 1, output_padding=1))   
            self.layers.append(nn.Conv2d(self.decKernel[k+1], self.decKernel[k+1], 3, 1, 1, padding_mode="reflect"))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm2d(self.decKernel[k+1]))
        
        #Final layer block (no ReLU)
        self.layers.append(nn.ConvTranspose2d(self.decKernel[-2], self.decKernel[-2], 3, stride = 2, padding = 1, output_padding=1))
        self.layers.append(nn.Conv2d(self.decKernel[-2], self.decKernel[-1], 3, 1, 1, padding_mode="reflect"))
        self.layers.append(nn.Sigmoid())
        
        #Create a sequential network to easy the forward pass
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        """
        Forward pass
        
        @input x :      Data to be forwarded
        """
        ret = self.layers(x)
        return ret

class DecSubPixConvUpsampleBranch(nn.Module):
    """
    Class for a single branch of the decoder of SARDNINet
    This decoder is based on subpixel convolutions for the last layer and on post upsampling convolution for the low resolution feature reconstructions
    """

    def __init__(self, decKernel) -> None:
        """
        Initialization

        @input decKernel :        Number of kernels for each layer including the number of channel in the input
        """
        super(DecSubPixConvUpsampleBranch, self).__init__()
        
        #List of the layers of the branch
        self.layers = []
        self.decKernel = decKernel

        #Creation of the layers
        for k in range(2):
            self.layers.append(nn.Upsample(scale_factor=2, mode = "bilinear"))                               
            self.layers.append(nn.Conv2d(self.decKernel[k], self.decKernel[k+1], 3, 1, 1, padding_mode="reflect"))   
            self.layers.append(nn.Conv2d(self.decKernel[k+1], self.decKernel[k+1], 3, 1, 1, padding_mode="reflect"))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm2d(self.decKernel[k+1]))
        
       
        #Final layer block (no ReLU)
        self.layers.append(nn.Conv2d(self.decKernel[-2], self.decKernel[-1]*2*2, 3, padding="same"))
        self.layers.append(nn.PixelShuffle(2))
        self.layers.append(nn.Sigmoid())        
        
        #Create a sequential network to easy the forward pass
        self.layers = nn.Sequential(*self.layers)


    def forward(self, x):
        """
        Forward pass
        
        @input x :      Data to be forwarded
        """
        ret = self.layers(x)
        return ret

class DecSubPixConvTransposeBranch(nn.Module):
    """
    Class for a single branch of the decoder of SARDNINet
    This decoder is based on transpose convolution for low resolution feature reconstruction and a sub pixel convolution in the last layer
    """

    def __init__(self, decKernel) -> None:
        """
        Initialization

        @input decKernel :        Number of kernels for each layer including the number of channel in the input
        """
        super(DecSubPixConvTransposeBranch, self).__init__()
        
        #List of the layers of the branch
        self.layers = []
        self.decKernel = decKernel

        #Creation of the layers
        for k in range(2):
            self.layers.append(nn.ConvTranspose2d(self.decKernel[k], self.decKernel[k+1], 3, stride = 2, padding = 1, output_padding=1))   
            self.layers.append(nn.Conv2d(self.decKernel[k+1], self.decKernel[k+1], 3, 1, 1, padding_mode="reflect"))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm2d(self.decKernel[k+1]))
        
        #Final layer block (no ReLU)
        self.layers.append(nn.Conv2d(self.decKernel[-2], self.decKernel[-1]*2*2, 3, padding="same"))
        self.layers.append(nn.PixelShuffle(2))
        self.layers.append(nn.Sigmoid())        
        
        #Create a sequential network to easy the forward pass
        self.layers = nn.Sequential(*self.layers)


    def forward(self, x):
        """
        Forward pass
        
        @input x :      Data to be forwarded
        """
        ret = self.layers(x)
        
        return ret

class DecSubPixConvBranch(nn.Module):
    """
    Class for a single branch of the decoder of SARDNINet

    """

    def __init__(self, decKernel) -> None:
        """
        Initialization

        @input decKernel :        Number of kernels for each layer including the number of channel in the input
        """
        super(DecSubPixConvBranch, self).__init__()
        
        #List of the layers of the branch
        self.layers = []

        #Number of kernels for each layer
        self.decKernel = decKernel

        #Creation of the layers
        for k in range(2):
            self.layers.append(nn.Conv2d(self.decKernel[k], self.decKernel[k+1]*2*2, 3, padding = "same"))
            self.layers.append(nn.PixelShuffle(2))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm2d(self.decKernel[k+1]))
        
        #Final layer block (no ReLU)
        self.layers.append(nn.Conv2d(self.decKernel[-2], self.decKernel[-1]*2*2, 3, padding="same"))
        self.layers.append(nn.PixelShuffle(2))
        self.layers.append(nn.Sigmoid())        
        
        #Create a sequential network to easy the forward pass
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        """
        Forward pass
        
        @input x :      Data to be forwarded
        """
        ret = self.layers(x)        
        return ret


#############################################################
#                       SARDINet                            #
#############################################################

class SARDINet(nn.Module):
    """
    Class for SARDINet architecture
    """

    def __init__(self, decoderNumber = 5, inpCondKernels = [2,96,128], encKernels = [128,256,512], decKernels = [256,96,1]):
        """
        Initialization

        @input decoderNumber:       Number of the decoder to be tested (default: 5):
                                        1 - standing for SARDINet with post-upsampling convolutions
                                        2 - standing for SARDINet with transposed convolutions
                                        3 - standing for SARDINet with sub-pixel convolutions
                                        4 - standing for SARDINet with post-upsampling convolutions and last layer of sub-pixel convolutions
                                        5 - standing for SARDINet with transposed convolutions and last layer of sub-pixel convolutions
        @input inpConKernels:       Number of kernels for the input conditionning step including the number of input channels (default: [2,96,128])
        @input encKernels:          Number of kernels for the encoder branches (default: [128, 256, 512])
        @input decKernels:          Number of kernels for the decoder branches including the number of output channels for a single decoder (default: [256, 96, 1])
        """

        super(SARDINet, self).__init__()
        
        #Attribute saving
        self.inpCondKernels = inpCondKernels
        self.encKernels = encKernels
        self.decKernels = decKernels
        self.decNumber = decoderNumber

        #Input conditionning
        self.inputCond = []
        for k in range(2):
            self.inputCond.append(nn.Conv2d(inpCondKernels[k],inpCondKernels[k+1],3,k+1,1, padding_mode="reflect"))
            self.inputCond.append(nn.ReLU())
            self.inputCond.append(nn.BatchNorm2d(inpCondKernels[k+1]))
        self.inputCond = nn.Sequential(*self.inputCond)

        #Top encoder branch
        self.encoderB1 = []
        self.encoderB1.append(SepConvBlock(encKernels[0], encKernels[1], 3))
        self.encoderB1.append(nn.BatchNorm2d(encKernels[1]))
        self.encoderB1.append(nn.MaxPool2d(2))
        self.encoderB1.append(nn.Conv2d(encKernels[1], encKernels[2], 1))
        self.encoderB1.append(nn.BatchNorm2d(encKernels[2]))
        self.encoderB1.append(nn.MaxPool2d(2))
        self.encoderB1 = nn.Sequential(*self.encoderB1)

        #Middle encoder branch
        self.encoderB2 = []
        self.encoderB2.append(nn.Conv2d(encKernels[0], encKernels[1], 1))
        self.encoderB2.append(nn.ReLU())
        self.encoderB2.append(nn.BatchNorm2d(encKernels[1]))
        self.encoderB2.append(nn.MaxPool2d(2))
        self.encoderB2.append(SepConvBlock(encKernels[1], encKernels[1], 3))
        self.encoderB2.append(nn.ReLU())
        self.encoderB2.append(nn.BatchNorm2d(encKernels[1]))
        self.encoderB2.append(SepConvBlock(encKernels[1], encKernels[2], 3))
        self.encoderB2.append(nn.BatchNorm2d(encKernels[2]))
        self.encoderB2.append(nn.MaxPool2d(2))
        self.encoderB2 = nn.Sequential(*self.encoderB2)

        #Bottom encoder branch
        self.encoderB3 = []
        self.encoderB3.append(SepConvBlock(encKernels[0], encKernels[1], 3))
        self.encoderB3.append(nn.ReLU())
        self.encoderB3.append(nn.BatchNorm2d(encKernels[1]))
        self.encoderB3.append(SepConvBlock(encKernels[1], encKernels[1], 3))
        self.encoderB3.append(nn.BatchNorm2d(encKernels[1]))
        self.encoderB3.append(nn.MaxPool2d(2))
        self.encoderB3.append(SepConvBlock(encKernels[1], encKernels[2], 3))
        self.encoderB3.append(nn.BatchNorm2d(encKernels[2]))
        self.encoderB3.append(nn.MaxPool2d(2))
        self.encoderB3 = nn.Sequential(*self.encoderB3)


        #Decoder branches - choice of the type of decoder depending on the decNumber specified
        decKernels = [encKernels[-1]] + decKernels
        if self.decNumber == 1 :    #Post-upsampling convolution
            self.decoderB1 = DecBranch(decKernels)
            self.decoderB2 = DecBranch(decKernels)
            self.decoderB3 = DecBranch(decKernels)
        elif self.decNumber == 2 :  #Transposed Convolutions
            self.decoderB1 = DecTransposeBranch(decKernels)
            self.decoderB2 = DecTransposeBranch(decKernels)
            self.decoderB3 = DecTransposeBranch(decKernels)
        elif self.decNumber == 3 :  #Sub-pixel convolutions
            self.decoderB1 = DecSubPixConvBranch(decKernels)
            self.decoderB2 = DecSubPixConvBranch(decKernels)
            self.decoderB3 = DecSubPixConvBranch(decKernels)
        elif self.decNumber == 4 :  #Post upsampling convolution with a last sub-pixel convolution layer
            self.decoderB1 = DecSubPixConvUpsampleBranch(decKernels)
            self.decoderB2 = DecSubPixConvUpsampleBranch(decKernels)
            self.decoderB3 = DecSubPixConvUpsampleBranch(decKernels)
        elif self.decNumber == 5 :  #Transposed convolution with a last sub-pixel convolution layer
            self.decoderB1 = DecSubPixConvTransposeBranch(decKernels)
            self.decoderB2 = DecSubPixConvTransposeBranch(decKernels)
            self.decoderB3 = DecSubPixConvTransposeBranch(decKernels)
        else :  #Raise an error if the decoder number is incorrect
            raise Exception("Wrong decoder number, processus aborted")
            
            

    def forward(self, x):
        """
        Forward pass

        @input x :      Data to be forwarded
        """

        #Input conditionning
        ret = self.inputCond(x)

        #Encoder branches
        ret1 = self.encoderB1(ret)
        ret2 = self.encoderB2(ret)
        ret3 = self.encoderB3(ret)

        #Fusion at the latent space
        fus = ret1 + ret2 + ret3

        #Decoder reconstruction
        d1 = self.decoderB1(fus)
        d2 = self.decoderB2(fus)
        d3 = self.decoderB3(fus)

        #Concatenation of the reconstructed channels
        finalRet = torch.cat((d1,d2,d3), dim = 1)
        
        return finalRet
