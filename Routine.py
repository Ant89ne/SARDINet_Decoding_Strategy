import torch
from torch.optim import Adam

from TransNet import SARDINet
from utils import eval_loop, training_loop, visIm, checkpoint

from datetime import datetime

class TrainRoutine :
    """
    Class used to ensure a correct training routine
    """

    def __init__(self, paths, hyperparameters, dataloaders, modelParams, optimization, device = "gpu"):
        """
        Initialization - for the detail of the components of the dictionaries, see the function init_params

        @input paths:               Dictionnary with the paths used for the training
        @input hyperparameters:     Dictionnary of the hyperparameters used for the training
        @input dataloaders:         Dictionnary of the train, eval and test dataloaders
        @input modelParams:         Dictionnary of the parameters of the model
        @input optimization:        Dictionnary of the parameters for the optimization (especially loss function here)
        @input device:              Device used to compute the training (default: "gpu")
        """
        self.init_params(paths, hyperparameters, dataloaders, modelParams, optimization)
        self.device = device

        #Model creation
        if len(self.modelPath)>0 :
            self.model = torch.load(self.modelPath)
        else :
            self.model = SARDINet(decoderNumber=self.decoderNumber)
        self.model.to(self.device)        #Transfer the model to the selected device
        
        #Optimizer for the training
        self.optim = Adam(self.model.parameters(), lr = self.lr_main)


    def init_params(self, paths, hyperparameters, dataloaders, modelParams, optimization):
        """
        Function used to unwrap the dictionnaries
        
        @input paths:               Dictionnary with the paths used for the training
        @input hyperparameters:     Dictionnary of the hyperparameters used for the training
        @input dataloaders:         Dictionnary of the train, eval and test dataloaders
        @input modelParams:         Dictionnary of the parameters of the model
        @input optimization:        Dictionnary of the parameters for the optimization (especially loss function here)
        """

        self.pathInput = paths["pathInput"]
        self.pathGT = paths["pathGT"]
        self.saveDir = paths["saveDir"]
        
        self.epochs = hyperparameters["epochs"]
        self.percentages = hyperparameters["percentages"]
        self.batch_size = hyperparameters["batch_size"]
        self.lr_main = hyperparameters["lr_main"]
        
        self.dataloader = dataloaders["dataloader"]
        self.dataloader_eval = dataloaders["dataloader_eval"]
        self.dataloader_test = dataloaders["dataloader_test"]
        
        self.modelPath = modelParams["modelPath"]
        self.decoderNumber = modelParams["decoderNumber"]

        self.loss_fn = optimization["loss_fn"]
    
    def trainNet(self):
        """
        Function used to train the network in its whole
        """
        currDate = datetime.now()

        #Evaluation of the untrained network
        self.TrainLosses = [eval_loop(self.dataloader, self.model, self.loss_fn, self.device)]
        self.EvalLosses = [eval_loop(self.dataloader_eval, self.model, self.loss_fn, self.device)]


        #Starting training
        for e in range(self.epochs) :
            print("\nEpoch " + str(e+1) + "/" + str(self.epochs))

            #Move network to the selected device
            self.model.to(self.device)
            #Time measurement
            t1 = datetime.now()
            
            #One step of training
            TLoss = training_loop(self.dataloader, self.model, self.loss_fn, self.optim, self.device, fid_on = False)
            
            #Save metrics
            self.TrainLosses.append(TLoss)
            
            #Time measurement
            t2 = datetime.now()
            
            #Evaluate the training
            ELoss = eval_loop(self.dataloader_eval, self.model, self.loss_fn, self.device, fid_on = False)
            
            #Save metrics
            self.EvalLosses.append(ELoss)
            
            #Time measurement
            t3 = datetime.now()

            print("Elapsed time Training : " + str(t2-t1) + " | Elapsed time Evaluation : " + str(t3-t2))

            #Save the model every 10 epochs
            if not (e+1) % 10 :
                checkpoint(self.model, self.TrainLosses, self.EvalLosses, e+1, self.saveDir)
                visIm(self.model, self.dataloader.dataset, e+1, self.saveDir+"Training/")
                visIm(self.model, self.dataloader_eval.dataset, e+1, self.saveDir+"Eval/")

        #Save the final model
        checkpoint(self.model, self.TrainLosses, self.EvalLosses, e+1, self.saveDir)
        visIm(self.model, self.dataloader.dataset, e+1, self.saveDir+"Training/")
        visIm(self.model, self.dataloader_eval.dataset, e+1, self.saveDir+"Eval/")

        #Time measurement
        tfin = datetime.now()

        print("Full Training Elapsed Time : " + str(tfin - currDate))

    def inferNet(self):
        """
        Function used to test the model
        """
        self.model.to(self.device)
        TestLoss = eval_loop(self.dataloader_test, self.model, self.loss_fn, self.device, fid_on=True)
        visIm(self.model, self.dataloader_test.dataset, -1, self.saveDir+"Test/", nbIm = -1)