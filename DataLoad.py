############################################################
#                       IMPORT LIBRAIRIES                   #
#############################################################

#Pytorch librairies
import torch
from torch.utils.data import Dataset


#Usefull librairies
import os
import numpy as np
import random
import rioxarray
import json


#Project librairies
from utils import checkDir


#############################################################
#                       CLASS CREATION                      #
#############################################################
    
class BEN_SarOpt_Dataset(Dataset):
    """
    Class for the data loading process
    Data used here : BigEarthNet-MM
    """

    def __init__(self, sarImFolder, optImFolder, percentages, id, seed = 28) -> None:
        """
        Initialization function
        @input sarImFolder:     Path to the folder with SAR images
        @input optImFolder:     Path to the folder with corresponding optical images
        @input percentage:      List of the percentages used for training, evaluation and test
        @input id:              Identifier of the dataset to be created (training = 0, evaluation = 1, test = 2)
        """

        #Seed for the shuffling of SAR images
        random.seed(seed)
        torch.manual_seed(seed)

        super(BEN_SarOpt_Dataset, self).__init__()
        
        #Create temporary directory to speed the reading process
        checkDir("./temp/")

        #Save attributes
        self.optPath = optImFolder
        self.sarPath = sarImFolder

        #List optical and radar images
        self.listOpt = [f for f in os.listdir(optImFolder) if f.startswith("S2")]
        self.listSar = [f for f in os.listdir(sarImFolder) if f.startswith("S1")]

        #Shuffle SAR images for a random sampling of SAR images
        random.shuffle(self.listSar)
        
        #Keep only images part of the dataset numbered "id"
        i = int(np.sum(percentages[:id])*len(self.listSar))
        j = int(np.sum(percentages[:id+1]) * len(self.listSar))
        self.listSar = self.listSar[i:j]

        #Size of dataset images
        self.sSar = (2,120,120)

        #Same for optical image
        self.sOpt = (3,120,120)


    def __len__(self):
        """
        Get the size of the dataset

        @return :       Size of the dataset
        """
        return len(self.listSar)

    def __getitem__(self, index) :
        """
        Function used to access a sample of the dataset
        
        @input index:   number of the sample to select

        @return:        Couple of SAR-Optical images corresponding to the same zone
        """
        
        #Read the selected radar image
        sarImFolder = self.listSar[index]
        
        
        #Initialize output images
        newDataOpt = np.zeros(self.sOpt)
        newDataSar = np.zeros(self.sSar)
        
        #Read SAR images
        for k, pol in enumerate(["VH","VV"]):
            pathIm = self.sarPath + sarImFolder + '/' + sarImFolder + f'_{pol}.tif'
            os.system(f"cp {pathIm} ./temp/")            
            im = rioxarray.open_rasterio("./temp/" + sarImFolder + f'_{pol}.tif')
            newDataSar[k,:,:] = im[0,:,:]
            os.system(f"rm ./temp/" + sarImFolder + f'_{pol}.tif')
        newDataSar = torch.Tensor(newDataSar)

        #Identify the corresponding Optical image
        pathLab = self.sarPath + sarImFolder + '/' + sarImFolder + '_labels_metadata.json'
        a = open(pathLab, "r")
        labs = json.load(a)
        optImFolder = labs["corresponding_s2_patch"]
        
        #Read the optical image
        for k in range(2,5):
            pathIm = self.optPath + optImFolder + '/' + optImFolder + f'_B0{k}.tif'
            os.system(f"cp {pathIm} ./temp/")            
            im = rioxarray.open_rasterio("./temp/" + optImFolder + f'_B0{k}.tif')
            newDataOpt[k-2,:,:] = im[0,:,:]/4096.0
            os.system(f"rm ./temp/" + optImFolder + f'_B0{k}.tif')
        newDataOpt = torch.Tensor(newDataOpt)
        newDataOpt = -torch.threshold( -newDataOpt, -1, -1)

        return newDataSar, newDataOpt
