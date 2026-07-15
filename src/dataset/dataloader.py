import torch
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import mgzip
import pickle
import matplotlib.pyplot as plt
import random
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, condition):
        # Get simulation data
        POS,FORCE,VISION,OPT_FLOW,ACT = self.get_data(path, condition)
        # Split data in input and output
        # input = self.assemble_input(VISION,POS)
        input = VISION[:-1] #self.assemble_input(VISION[:-1],POS[:-1])
        cond_input = ACT[:-1]
        # output = self.assemble_output(OPT_FLOW,POS,FORCE)
        output = VISION[1:] #self.assemble_output(VISION[1:],POS[1:],FORCE[1:]) #VISION[1:]
        # Get training dataset
        self.get_training_dataset(input,cond_input,output)
        
    def get_data(self, path, condition):                
        for trial in range(1):
            # Load data
            data = self.load_data(path,trial,condition)
            # Manipulate data  
            pos,force,vision,opt_flow,act = self.shift_data(data)
            # Merge trials (if more than 1)
            if trial == 0:
                POS = pos
                FORCE = force
                VISION = vision
                OPT_FLOW = opt_flow
                ACT = act
            else:
                POS = np.vstack((POS,pos))
                FORCE = np.vstack((FORCE,force))
                VISION = np.vstack((VISION,vision))
                OPT_FLOW = np.vstack((OPT_FLOW,opt_flow))
                ACT = np.vstack((ACT,act))
        # Data pre-processing
        data = (POS,FORCE,VISION,OPT_FLOW,ACT)
        return self.preprocess_data(data)
        
    def load_data(self,path,trial,condition):
        # Build the complete path and open the file
        path_final = os.path.join(path,"trial_"+str(trial+1)+"_"+condition+".pkl")
        f = mgzip.open(path_final,"rb")
        # Read the file contents
        time = pickle.load(f)
        pos = pickle.load(f)
        force = pickle.load(f)
        act = pickle.load(f)
        vision = pickle.load(f)
        # Close the file
        f.close()
        return (time,pos,force,act,vision)
    
    def shift_data(self,data):
        _,pos,force,act,vision = data
        # Manipulate data to current or next time-step 
        pos = pos[:-1]
        force = force[:-1]
        opt_flow = vision[1:] - vision[:-1] #self.build_optical_flow_dataset(vision)
        vision = vision[:-1]
        act = np.hstack((act[:-1],act[1:])) # it can also contain previous information (recurrency)
        return (pos,force,vision,opt_flow,act)
    
    def preprocess_data(self,data):
        POS,FORCE,VISION,OPT_FLOW,ACT = data
        # Process data with scaling (standardized scaling)
        self.sc_POS = StandardScaler()
        POS = self.sc_POS.fit_transform(POS)
        self.sc_FORCE = StandardScaler()
        FORCE = self.sc_FORCE.fit_transform(FORCE)
        self.sc_ACT = StandardScaler()
        ACT = self.sc_ACT.fit_transform(ACT)
        VISION = VISION/255.        
        OPT_FLOW = OPT_FLOW/255.
        return (POS,FORCE,VISION,OPT_FLOW,ACT)
    
    def assemble_input(self, inp_1, inp_2): # inp_1: VISION, inp_2: POS
        h, w = inp_1.shape[1], inp_1.shape[2]
        rep_h = int(h/inp_2.shape[1])+1
        rep_w = int(w)
        
        input_new = np.zeros((inp_1.shape[0],inp_1.shape[1],inp_1.shape[2],inp_1.shape[3]+1))
        for sample in range(len(inp_2)):
            input_new[sample,:,:,0:3] = inp_1[sample]
            inp_2_mod = np.tile(inp_2[sample],(rep_w,rep_h))
            inp_2_mod = inp_2_mod[0:h,0:w]
            input_new[sample,:,:,3] = inp_2_mod         
        return input_new
    
    def assemble_output(self, inp_1, inp_2, inp_3): # inp_1: OPT_FLOW, inp_2: POS, inp_3: FORCE
        h, w = inp_1.shape[1], inp_1.shape[2]
        rep_h = int(h/inp_2.shape[1])+1
        rep_w = int(w)
        
        output_new = np.zeros((inp_1.shape[0],inp_1.shape[1],inp_1.shape[2],inp_1.shape[3]+2))
        for sample in range(len(inp_1)):
            output_new[sample,:,:,0:3] = inp_1[sample]
            out_2_mod = np.tile(inp_2[sample],(rep_w,rep_h))
            out_2_mod = out_2_mod[0:h,0:w]
            output_new[sample,:,:,3] = out_2_mod
            out_3_mod = np.tile(inp_3[sample],(rep_w,rep_h))
            out_3_mod = out_3_mod[0:h,0:w]
            output_new[sample,:,:,4] = out_3_mod            
        return output_new
    
    def build_optical_flow_dataset(self,vision):
        opt_flow = np.zeros((vision.shape[0]-1,vision.shape[1],vision.shape[2],vision.shape[3]))
        for sample in range(1,vision.shape[0]):
            opt_flow[sample-1] = self.get_optical_flow(vision[sample-1],vision[sample])
        return opt_flow
    
    def get_optical_flow(self,img1,img2):
        # Convert frames to float32
        prev = np.float32(img1)
        next = np.float32(img2)
        # Compute optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(prev,cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(next,cv2.COLOR_RGB2GRAY),
            None,  # Flow map to be filled, set to None to create a new array
            0.2,   # Pyramids scale factor
            4,     # Number of pyramid layers
            1,    # Window size for each pyramid layer
            50,     # Number of iterations at each pyramid layer
            7,     # Size of pixel neighborhood used to find polynomial expansion
            0.3,   # Standard deviation of the Gaussian that is used to smooth derivatives
            0      # Flags
        )
        # Calculate the polar coordinates and magnitude
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Create an RGB image representing the flow
        hsv = np.zeros_like(prev)
        hsv[..., 1] = 255
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        # Convert HSV to BGR
        flow_rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
        
        plt.imshow(flow_rgb)
        plt.show()
        return flow_rgb
    
    def get_training_dataset(self,input,cond_input,output):
        # Convert data to tensors
        X = torch.tensor(input,dtype=torch.float32).to(device)
        X = X.permute(0, 3, 1, 2)
        # X = X[:,3,:,:]
        # X = X.reshape(X.size(0), -1, X.size(1), X.size(2))
        cX = torch.tensor(cond_input,dtype=torch.float32).to(device)
        Y = torch.tensor(output,dtype=torch.float32).to(device)
        Y = Y.permute(0, 3, 1, 2)
        # Y = Y[:,3:5,:,:]
        # X = X.reshape(X.size(0), -1, X.size(1), X.size(2))
        # Find training and validation data indexes
        size_train = int(0.7*len(X))
        size_val = len(X) - size_train
        idx_val = np.random.random_integers(0,len(X)-1,size_val)
        idx_train = np.arange(0,len(X))
        idx_train = np.delete(idx_train,idx_val)
        # Split in train and validation
        self.X_train, self.cX_train, self.Y_train = X[idx_train], cX[idx_train], Y[idx_train]
        self.X_train = self.X_train
        self.cX_train = self.cX_train
        self.Y_train = self.Y_train
        self.X_val, self.cX_val, self.Y_val = X[idx_val], cX[idx_val], Y[idx_val]
    
    def rescale(self, data, type):
        if type == "VISION" or type == "FLOW":
            data = data*255.
            return data
        elif type == "POS":
            sc = self.sc_POS
        elif type == "FORCE":
            sc = self.sc_FORCE
        elif type == "ACT":
            sc = self.sc_ACT
        return sc.inverse_transform(data.detach().numpy())
    
    def get_samples(self,start,end):
        return self.X[start:end], self.Y[start:end], self.ACT[start:end]
    
    def get_training_set(self):
        return self.X_train, self.Y_train, self.cX_train
    
    def get_validation_set(self):
        return self.X_val, self.Y_val, self.cX_val
    
    def __len__(self):
        return len(self.Y_train)
    
    def __getitem__(self, index):
        return self.X_train[index], self.Y_train[index], self.cX_train[index]
    
def main():
    # Path to data location
    path = "Workspace\data"
    scenario = "no_obj"
    
    # Build the training and validation datasets
    feature_set = Dataset(path,scenario)
    
# if __name__ == "__main__":
#     main()