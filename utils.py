#############################################################
#                    IMPORT LIBRAIRIES                      #
#############################################################

#Pytorch librairies
import torch
from torch import nn 
import torchgeometry as geo

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics import MeanSquaredError


#Usefull librairies
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import copy
import cv2
import tqdm

#############################################################
#                    METRICS FUNCTIONS                      #
#############################################################

def getNewMetrics(pred, l, device):
    """
    Function used to get the metrics to evaluate the performances of the network
    @input pred:        Prediction made by SARDINet
    @input l:           Ground truth labels
    @input device:      Whether "cpu" or "cuda", device on which to compute the metrics

    @return:            Fréchet Inception Distance result, Peak Signal to Noise Ratio result, MSE results
    """

    #Mean Squared Error calculation
    mse = MeanSquaredError().to(device)
    final_mse = mse(pred, l)

    #Quantization on 8 bits for following metrics
    pred = (255*pred).to(dtype = torch.uint8)
    l = (255*l).to(dtype = torch.uint8)

    #Fréchet Inception Distance calculation
    fid = FrechetInceptionDistance(feature = 64).to(device)
    fid.update(pred, False)
    fid.update(l, True)
    final_fid = fid.compute()

    #Peak Signal to Noise Ratio calculation
    psnr = PeakSignalNoiseRatio().to(device)
    final_psnr = psnr(pred, l)

    return copy.deepcopy(final_fid.item()), copy.deepcopy(final_psnr.item()), copy.deepcopy(final_mse.item())


#############################################################
#                    TRAINING LOOPS                         #
#############################################################

def training_loop(dataloader, model, loss_fn, optimizer, device = "cpu"):
    """
    Training loop for classical network
    @input dalaoader :          Dataloader to be used
    @input model :              Neural network to be trained
    @input loss_fn :            Loss to be minimized
    @input optimizer :          Optimizer to be used
    @input device :             Device to be used (default : cpu)
    
    @return :                   Loss and metrics (MSE, PSNR and FID) on training data
    """
    
    #Switch model to training mode
    model.train()

    #Initialization
    TLoss = 0
    fid, psnr, mse = 0,0,0

    #Zeroing gradient (avoid accumulation)
    optimizer.zero_grad()

    #Routine
    for i, (d, l) in enumerate(tqdm.tqdm(dataloader)) :
        
        #Copy data and label to the selected device
        d = d.to(device)
        l = l.to(device)

        #Prediction
        pred = model(d)

        #Loss calculation        
        loss = loss_fn(pred,l)

        #Save Loss
        TLoss += loss.item()

        #Loss backpropagation
        loss.backward()
        
        #Metrics calculation
        fidc, psnrc, msec = getNewMetrics(pred, l, "cpu")
        fid += fidc
        psnr += psnrc
        mse += msec

        #Optimization when batches are ready
        optimizer.step()

        #Zeroing the gradient
        optimizer.zero_grad()

    #Averaging
    fid /= len(dataloader)
    psnr /= len(dataloader)
    mse /= len(dataloader)
    TLoss /= len(dataloader)
    print("Training : Loss : " + str(TLoss)+ " | MSE : " + str(mse) + " | PSNR : " + str(psnr) + " | FID : " + str(fid))
    
    return [TLoss, mse, psnr, fid]

#############################################################
#                   EVALUATION LOOPS                        #
#############################################################

def eval_loop(dataloader, model, loss_fn, device):
    """
    Main loop routine for the evaluation of the network

    @input dataloader :         Dataloader to be used
    @input model :              Model to be evaluated
    @input loss_fn :            Loss used for training
    @input device :             Device to be used

    @return :                   Loss and metrics (MSE, PSNR, FID) on the evaluation dataset
    """

    #Switch to evaluation mode
    model.eval()
    ELoss = 0
    mse, psnr, fid = 0,0, 0

    #Routine
    for i, (d, l) in enumerate(tqdm.tqdm(dataloader)) :

        #Copy images and ground truth on selected device
        d = d.to(device)
        l = l.to(device)

        #Prediction
        pred = model(d)

        #Loss calculation
        loss = loss_fn(pred,l)

        #Get metrics MAE and RMSE
        fidc, psnrc, msec = getNewMetrics(pred, l)
        fid += fidc
        psnr += psnrc     
        mse += msec

        #Save Loss
        ELoss += loss.item()

    #Average metrics
    ELoss /= len(dataloader)
    fid /= len(dataloader)
    psnr /= len(dataloader)
    mse /= len(dataloader)

    print("Evaluation : Loss : " + str(ELoss) + " | MSE : " + str(mse) + " | PSNR : " + str(psnr) + " | FID : " + str(fid))
    
    return [ELoss, mse, psnr, fid]

#############################################################
#                    VISUALIZATION                          #
#############################################################

def visLoss(TrainLosses, EvalLosses):
    """
    Function used to visualize the final loss evolution

    @input TrainLosses :        Training loss and metrics
    @input EvalLosses :         Evaluation loss and metrics
    """

    plt.figure(figsize=(20,10))
    
    #Loss visualization
    plt.subplot(232)
    plt.plot([k for k in range(len(TrainLosses))], TrainLosses[:,0], 'r-')
    plt.plot([k for k in range(len(EvalLosses))], EvalLosses[:,0], 'b-')
    plt.title("Loss over epochs")
    plt.legend(["Training Loss", "Evaluation Loss"])
    
    #Loss zoomed
    if len(TrainLosses) > 5 :
        plt.subplot(131)
        plt.plot([k for k in range(len(TrainLosses))], TrainLosses[:,0], 'r-')
        plt.plot([k for k in range(len(EvalLosses))], EvalLosses[:,0], 'b-')
        plt.title("Loss over epochs - zoom")
        plt.axis([0,len(TrainLosses), 0, np.maximum(np.sort(EvalLosses[:,0])[-5], np.sort(TrainLosses[:,0])[-5])])
        plt.legend(["Training Loss", "Evaluation Loss"])

        #MSE visualization
        plt.subplot(233)
        plt.plot([k for k in range(len(TrainLosses))], TrainLosses[:,1], 'r-')
        plt.plot([k for k in range(len(EvalLosses))], EvalLosses[:,1], 'b-')
        plt.title("MSE over epochs")
        plt.legend(["Training MSE", "Evaluation MSE"])
        plt.axis([0,len(TrainLosses), 0, np.maximum(np.sort(EvalLosses[:,1])[-5], np.sort(TrainLosses[:,1])[-5])])
    
        #PSNR visualization
        plt.subplot(235)
        plt.plot([k for k in range(len(TrainLosses))], TrainLosses[:,2], 'r-')
        plt.plot([k for k in range(len(EvalLosses))], EvalLosses[:,2], 'b-')
        plt.title("PSNR over epochs")
        plt.legend(["Training PSNR", "Evaluation PSNR"])
        plt.axis([0,len(TrainLosses), 0, np.maximum(np.sort(EvalLosses[:,2])[-5], np.sort(TrainLosses[:,2])[-5])])

        #FID visualization
        plt.subplot(236)
        plt.plot([k for k in range(len(TrainLosses))], TrainLosses[:,3], 'r-')
        plt.plot([k for k in range(len(EvalLosses))], EvalLosses[:,3], 'b-')
        plt.title("FID over epochs")
        plt.legend(["Training FID", "Evaluation FID"])
        plt.axis([0,len(TrainLosses), 0, np.maximum(np.sort(EvalLosses[:,3])[-5], np.sort(TrainLosses[:,3])[-5])])


def visIm(model, dataset, epoch, dir, nbIm = 4):
    """
    Function used to visualize images after training
    
    @input model :          Network to be used
    @input dataset :        Dataset to be used
    @input epoch :          Number of training epochs already done
    @input dir :            Directory where to save the images
    @input nbIm :           Number of images to visualize (defualt : 4)
    """
    
    #TODO vérifier si besoin de random.seed(25) ou pas

    #Create directory if not yet existing
    checkDir(dir)

    with torch.no_grad():

        for k in range(nbIm):
            
            #Get an image
            d,l = dataset[k]

            #Copy network on cpu
            model.cpu()

            #Reshape image
            s = d.shape
            d = torch.reshape(d, (1, s[0], s[1], s[2]))
            
            #Prediction
            pred = model(d)

            #Transpose for visualization purposes
            pred = torch.transpose(pred,1,3)
            l = torch.transpose(l, 0,2)
            d = torch.transpose(d, 1,3)

            #Optical prediction
            plt.figure(figsize=(20,10))
            plt.clf()
            plt.subplot(211)
            plt.imshow(np.uint8(cv2.cvtColor(np.array(pred[0,:,:,:]),cv2.COLOR_BGR2RGB)*255))
            plt.title("RGB Prediction")

            #Optical ground truth
            plt.subplot(212)
            plt.imshow(np.uint8(cv2.cvtColor(np.array(l),cv2.COLOR_BGR2RGB)*255))
            plt.title("RGB Ground Truth")

            #Save images
            plt.savefig(dir + "im" + str(k) + "_" + str(epoch))
            plt.close()

            #Input visualization
            plt.figure(figsize=(20,10))
            plt.clf()
            plt.subplot(121)
            plt.imshow(d[0,:,:,0], cmap="gray")
            plt.title("VH")

            plt.subplot(122)
            plt.imshow(d[0,:,:,1], cmap="gray")
            plt.title("VV")

            plt.savefig(dir + "input" + str(k))
            plt.close()



#############################################################
#                         UTILS                             #
#############################################################


def checkpoint(model, TLoss, ELoss, epoch, dir):
    """
    Save the network

    @input model :          Network to be saved
    @input TLoss :          Training loss evolution to be saved
    @input ELoss :          Evaluation loss evolution to be saved
    @input dir :            Directory where to save
    """

    #Network saving
    torch.save(model, dir + "model_" + str(epoch) + ".pt")

    #Loss visualization
    visLoss(np.array(TLoss), np.array(ELoss))
    
    #Loss saving
    plt.savefig(dir + "Loss_" + str(epoch) + ".png")
    plt.close()

def checkDir(path): 
    """
    Check if the path exists, if not, creates it

    @input path :           Path to be checked
    """
    if not os.path.isdir(path): 
        os.mkdir(path)
