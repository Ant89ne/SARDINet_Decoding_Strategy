#############################################################
#                       IMPORT LIBRAIRIES                   #
#############################################################

#Pytorch librairies
import torch
from torch import nn 
from torch.utils.data import DataLoader
import torchgeometry as geo

#Usefull librairies
import numpy as np
from datetime import datetime
import random
import argparse


#My own functions
#My own functions
from DataLoad import BEN_SarOpt_Dataset
from TransNet import SARDINet
from utils import *
from Routine import TrainRoutine

#############################################################
#                     INITIALIZATION                        #
#############################################################
seed = 27
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

#Initialization
pathInput = "/home/abralet//NAS/Bralet/DataBigEarth200/BigEarthNet-S1-v1.0/"
pathGT = "/home/abralet//NAS/Bralet/DataBigEarth200/BigEarthNet-v1.0_chosen/"
outputDir = "/bigdata1/abralet/pilou/"

#Create directory for checkpoints
currDate = datetime.now()
saveDir = outputDir + str(currDate).replace(' ', '_').replace(':',"-") + "/" 
checkDir(saveDir)

device = "cuda" if (torch.cuda.is_available()) else "cpu"      #Whether to use gpu or cpu
num_workers = 40 #Number of subprocesses in parallel to load data

paths = {
    "pathInput" : pathInput,
    "pathGT" : pathGT,
    "saveDir" : saveDir
}

#############################################################
#                     HYPERPARAMETERS                       #
#############################################################

epochs = 30                     #Number of epochs
percentages = [0.7,0.2,0.1]     #Percentage of training and evaluation
batch_size = 8                 #Batch size
lr = 5e-5                  #learning rate for SARDINet

hyperparameters = {
    "epochs" : epochs,
    "percentages" : percentages,
    "batch_size" : batch_size,
    "lr_main" : lr
}

#############################################################
#               DATASETS AND DATALOADERS                    #
#############################################################

d_train = BEN_SarOpt_Dataset(pathInput, pathGT, percentages, 0)         #Training dataset
d_evalu = BEN_SarOpt_Dataset(pathInput, pathGT, percentages, 1)         #Evaluation dataset
d_tests = BEN_SarOpt_Dataset(pathInput, pathGT, percentages, 2)         #Test dataset
dataloader = DataLoader(d_train, 1, shuffle = True, num_workers = num_workers)                                                             #Training dataloader
dataloader_eval = DataLoader(d_evalu, 1, shuffle = False)                                                       #Evaluation dataloader
dataloader_test = DataLoader(d_tests, 1, shuffle = False)

dataloaders = {
    "dataloader": dataloader,
    "dataloader_eval": dataloader_eval,
    "dataloader_test": dataloader_test
}

#############################################################
#                         NETWORK                           #
#############################################################

modelPath = ""
decoderNumber = 5

modelparams = {
    "modelPath": modelPath,
    "decoderNumber": decoderNumber
}

#############################################################
#                     OPTIMIZATION                          #
#############################################################

#Loss function
loss_fn = lambda pred,lab: nn.L1Loss(reduction = "mean")(pred,lab) + geo.losses.SSIM(5, reduction="mean")(pred,lab)

optimization = {
    "loss_fn" : loss_fn,
}

#############################################################
#                   NETWORK TRAINING                        #
#############################################################

routine = TrainRoutine(paths, hyperparameters, dataloaders, modelparams, optimization, device)
routine.trainNet()

#############################################################
#                   NETWORK TRAINING                        #
#############################################################

routine.inferNet()