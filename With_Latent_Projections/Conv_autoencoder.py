import pandas as pd
import numpy as npy
import os
import time
import DTW as mypair
import scipy as math
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
import csv 
from keras import regularizers
from keras import layers 
import pickle
import tensorflow as tf
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
class conv_auto_encoder:
    data=npy.array
    data_average=npy.array
    File_loc=''
    File_name=''
    Model_save_path=''
    Model_save_name=''
    classif_mode=0
    uniques=[]
    counts=[]
    confusion_matrix=npy.array
    Encoder_layer=Model
    Decoder_layer=Model
    Classifier_layer=Model
    Warper_layer=Model
    Right_encoder=Model
    Left_encoder=Model
    mymodel=Model
    training_time=0
    final_estimate=npy.array
    hist_dict={}
    my_loss_count=0
    pridicted=0
    temp=''
    temps=''
    rejected=[]
    time_loss=0
    encoder_node_size=0
    validation=0
    validation_data=npy.array
    my_uniqes_copy=npy.array
    def __init__(self,File_loc,File_name,Model_save_path, Model_save_name,classif_mode,select_all,class_label,validation):
        self.File_loc=File_loc
        self.File_name=File_name
        self.Model_save_path=Model_save_path
        self.Model_save_name=Model_save_name
        self.classif_mode=classif_mode
        self.Model_save_path=Model_save_path
        self.Model_save_name=Model_save_name
        self.temp=pd.read_csv(File_loc+File_name+'_TRAIN.tsv',sep='\t',header=None)
        self.temps=pd.read_csv(File_loc+File_name+'_TEST.tsv',sep='\t',header=None)
        self.data=npy.zeros((1,self.temp.shape[1]),dtype=npy.float32)
        data_copy=npy.array(self.temp.iloc[:,:])
        uniques=npy.unique(data_copy[:,0])
        count=0
        max_length=max(self.temps.shape[0],self.temp.shape[0])
        if classif_mode==0 and select_all==1:
            for k in range(len(uniques)):
                for i in range(max_length):
                    if i<self.temp.shape[0] and self.temp.iloc[i,0]==uniques[k]:
                        if npy.isnan(npy.sum(self.temp.iloc[i,1:])):
                            self.rejected.append(i)
                        else:
                            count=count+1
                            temps_c=npy.array([self.temp.iloc[i,:]])
                            temps_c=temps_c.reshape((1,temps_c.shape[1]))
                            self.data=npy.concatenate((self.data,temps_c),axis=0)
                    if i<self.temps.shape[0] and self.temps.iloc[i,0]==uniques[k]:
                        if npy.isnan(npy.sum(self.temps.iloc[i,1:])):
                            self.rejected.append(i)
                        else:
                            count=count+1
                            temps_c=npy.array([self.temps.iloc[i,:]]).reshape(1,self.temps.shape[1])
                            self.data=npy.concatenate((self.data,temps_c),axis=0)
                self.uniques.append(int(uniques[k]))
                self.counts.append(count)
                count=0    
        else:
            if classif_mode==0 and select_all==0:
                for k in range(len(uniques)):
                    for i in range(self.temp.shape[0]):
                        if i<self.temp.shape[0] and self.temp.iloc[i,0]==uniques[k]:
                            if npy.isnan(npy.sum(self.temp.iloc[i,1:])):
                                 self.rejected.append(i)
                            else:
                                count=count+1
                                temps_c=npy.array([self.temp.iloc[i,:]])
                                temps_c=temps_c.reshape((1,temps_c.shape[1]))
                                self.data=npy.concatenate((self.data,temps_c),axis=0)
                    self.uniques.append(int(uniques[k]))
                    self.counts.append(count)
                    count=0   
            else:
                if classif_mode==1 and select_all==1:
                    for i in range(max_length):
                        if i<self.temp.shape[0] and self.temp.iloc[i,0]==uniques[class_label]:
                            if npy.isnan(npy.sum(self.temp.iloc[i,1:])):
                                 self.rejected.append(i)
                            else:
                                count=count+1
                                temps_c=npy.array([self.temp.iloc[i,:]])
                                temps_c=temps_c.reshape((1,temps_c.shape[1]))
                                self.data=npy.concatenate((self.data,temps_c),axis=0)
                        if i<self.temps.shape[0] and self.temps.iloc[i,0]==uniques[class_label]:
                            if npy.isnan(npy.sum(self.temps.iloc[i,1:])):
                                 self.rejected.append(i)
                            else:
                                count=count+1
                                temps_c=npy.array([self.temps.iloc[i,:]]).reshape(1,self.temps.shape[1])
                                self.data=npy.concatenate((self.data,temps_c),axis=0)
                    self.uniques.append(int(uniques[class_label]))
                    self.counts.append(count)
                    count=0
                else:
                    for i in range(self.temp.shape[0]):
                        if i<self.temp.shape[0] and self.temp.iloc[i,0]==uniques[class_label]:
                            if npy.isnan(npy.sum(self.temp.iloc[i,1:])):
                                 self.rejected.append(i)
                            else:
                                count=count+1
                                temps_c=npy.array([self.temp.iloc[i,:]])
                                temps_c=temps_c.reshape((1,temps_c.shape[1]))
                                self.data=npy.concatenate((self.data,temps_c),axis=0)
                    self.uniques.append(int(uniques[class_label]))
                    self.counts.append(count)
                    count=0  
        self.final_estimate=npy.zeros((len(self.uniques),self.data.shape[1]-1))
        self.data=self.data[1:,:]
        data=npy.zeros((1,self.data.shape[1]))
        self.my_uniqes_copy=npy.copy(self.uniques)
        self.validation_data=npy.zeros((1,self.data.shape[1]))
        for i in range(len(self.uniques)):
            temping=self.data[self.data[:,0]==self.uniques[i]]
            temping[:,0]=i
            data=npy.concatenate((data,temping))
            self.uniques[i]=i
        self.validation=validation
        self.data=data[1:,:]
        print('new data labels',self.data[:,0])
        self.copy_of_data=npy.copy(self.data)
        self.my_loss_count=0
        calc_val=int(self.temp.shape[0]*self.validation)
        print('validation',calc_val)
        if calc_val>0:
            count=0
            if calc_val>=len(self.uniques):
                for i in range(len(self.uniques)):
                    data=self.data[self.data[:,0]==i]
                    vals=int(data.shape[0]*self.validation)
                    if vals==0:
                            vals=1
                    if vals>=1:
                        for j in range(vals):
                            data=self.data[self.data[:,0]==i]
                            ind=npy.random.randint(0,data.shape[0]-1)
                            self.validation_data=npy.concatenate((self.validation_data,data[ind,:].reshape(1,data.shape[1])),axis=0)
                            res=npy.where(npy.all(self.data==data[ind,:],axis=1))
                            data=npy.delete(self.data,ind,0)
                            self.data=npy.delete(self.data,res[0][0],0)
                self.validation_data=self.validation_data[1:,:]
            else:
                self.validation_data=self.data[self.data.shape[0]-calc_val:,:]
                self.data=self.data[0:self.data.shape[0]-calc_val,:]
        print('My validation classes:',self.validation_data[:,0])
        npy.random.shuffle(self.data)
        self.confusion_matrix=npy.zeros((len(self.uniques),len(self.uniques)))
        print('Total Data sets in train file:',self.temp.shape[0])
        print('Total Data set in test file:',self.temps.shape[0])
        print('Totoal Unique classes in the data sets:',len(uniques))
        print('Unique Class labels:',uniques)
        print('selected unique class for the experiment:',self.uniques)
        print('Data sets in the selected unique class:',self.counts)
        print('Selected Centroid Mode:',self.classif_mode)
        print('Data sets rejected due to NA',len(self.rejected))
        print('Data size selected for training:',self.data.shape[0])
        print('Data size selected for validation',self.validation_data.shape[0])
    def build_network_vgg(self,encoder_node_size,filter_size, polling_size,En_L1_reg,En_L2_reg, De_L1_reg,De_L2_reg,Cl_L1_reg,Cl_L2_reg,Input_activ, Hidden_activ,Learning_rate):
        print('underconstruction')
        En_inputs=Input(shape=(self.data.shape[1]-1,))
        self.Encoder_layer=Sequential()
        self.encoder_node_size=encoder_node_size
        self.Encoder_layer=layers.Reshape((self.data.shape[1]-1,1), input_shape=(self.data.shape[1]-1,),name='EL1')(En_inputs)
        self.Encoder_layer=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Input_activ,name='EL2')(self.Encoder_layer)
        self.Encoder_layer=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,name='EL3')(self.Encoder_layer)
        self.Encoder_layer=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,name='EL4')(self.Encoder_layer)
        self.Encoder_layer=layers.MaxPooling1D(polling_size,name='EL5')(self.Encoder_layer)
        self.Encoder_layer=layers.Conv1D(64,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ, padding='same',name='EL6')(self.Encoder_layer)
        self.Encoder_layer=layers.Conv1D(64,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ,padding='same',name='EL7')(self.Encoder_layer)
        self.Encoder_layer=layers.Conv1D(64,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ, padding='same',name='EL8')(self.Encoder_layer)
        self.Encoder_layer=layers.MaxPooling1D(polling_size,name='EL9')(self.Encoder_layer)
        self.Encoder_layer=layers.Conv1D(128,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ, padding='same',name='EL10')(self.Encoder_layer)
        self.Encoder_layer=layers.Conv1D(128,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ,padding='same',name='EL11')(self.Encoder_layer)
        self.Encoder_layer=layers.Conv1D(128,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ, padding='same',name='EL12')(self.Encoder_layer)
        self.Encoder_layer=layers.MaxPooling1D(polling_size,name='EL13')(self.Encoder_layer)
        self.Encoder_layer=layers.Flatten(name='EL14')(self.Encoder_layer)
        E_out=layers.Dense(encoder_node_size,kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ,name='Encoder_output')(self.Encoder_layer)
        self.Decoder_layer=layers.Reshape((encoder_node_size,1),input_shape=(encoder_node_size,),name='DL1')(E_out)
        self.Decoder_layer=layers.Conv1D(128,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',name='DL2')(self.Decoder_layer)
        self.Decoder_layer=layers.Conv1D(128,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',name='DL3')(self.Decoder_layer)
        self.Decoder_layer=layers.Conv1D(128,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',name='DL4')(self.Decoder_layer)
        self.Decoder_layer=layers.UpSampling1D(polling_size,name='DL5')((self.Decoder_layer))
        self.Decoder_layer=layers.Conv1D(64,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',name='DL6')(self.Decoder_layer)
        self.Decoder_layer=layers.Conv1D(64,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',name='DL7')(self.Decoder_layer)
        self.Decoder_layer=layers.Conv1D(64,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',name='DL8')(self.Decoder_layer)
        self.Decoder_layer=layers.UpSampling1D(polling_size,name='DL9')((self.Decoder_layer))
        self.Decoder_layer=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',name='DL10')(self.Decoder_layer)
        self.Decoder_layer=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',name='DL11')(self.Decoder_layer)
        self.Decoder_layer=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',name='DL12')(self.Decoder_layer)
        self.Decoder_layer=layers.Flatten(name='DL13')((self.Decoder_layer))
        D_out=layers.Dense(self.data.shape[1]-1,kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Input_activ,name="Decoder_output")((self.Decoder_layer))
        self.mymodel=Model(En_inputs,D_out)
        self.mymodel.compile(optimizer=optimizers.Adam(lr=Learning_rate),loss='mse',metrics=['mse'])
        self.build_multi_Encoder_Decoder_separately_two()
    def Multi_task_encoder_two(self,encoder_node_size,filter_size, polling_size,En_L1_reg,En_L2_reg, De_L1_reg,De_L2_reg,Cl_L1_reg,Cl_L2_reg,Input_activ, Hidden_activ, Learning_rate):
        print('underconstruction')
        En_inputs=Input(shape=(self.data.shape[1]-1,))
        self.Encoder_layer=Sequential()
        self.encoder_node_size=encoder_node_size
        self.Encoder_layer=layers.Reshape((self.data.shape[1]-1,1), input_shape=(self.data.shape[1]-1,),name='EL1')(En_inputs)
        self.Encoder_layer=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Input_activ,name='EL2')(self.Encoder_layer)
        self.Encoder_layer=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,name='EL3')(self.Encoder_layer)
        self.Encoder_layer=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,name='EL4')(self.Encoder_layer)
        self.Encoder_layer=layers.MaxPooling1D(polling_size,name='EL5')(self.Encoder_layer)
        self.Encoder_layer=layers.Conv1D(64,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ, padding='same',name='EL6')(self.Encoder_layer)
        self.Encoder_layer=layers.Conv1D(64,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ,padding='same',name='EL7')(self.Encoder_layer)
        self.Encoder_layer=layers.Conv1D(64,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ, padding='same',name='EL8')(self.Encoder_layer)
        self.Encoder_layer=layers.MaxPooling1D(polling_size,name='EL9')(self.Encoder_layer)
        self.Encoder_layer=layers.Conv1D(128,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ, padding='same',name='EL10')(self.Encoder_layer)
        self.Encoder_layer=layers.Conv1D(128,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ,padding='same',name='EL11')(self.Encoder_layer)
        self.Encoder_layer=layers.Conv1D(128,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ, padding='same',name='EL12')(self.Encoder_layer)
        self.Encoder_layer=layers.MaxPooling1D(polling_size,name='EL13')(self.Encoder_layer)
        self.Encoder_layer=layers.Flatten(name='EL14')(self.Encoder_layer)
        E_out=layers.Dense(encoder_node_size,kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ,name='Encoder_output')(self.Encoder_layer)
        self.Decoder_layer=layers.Reshape((encoder_node_size,1),input_shape=(encoder_node_size,),name='DL1')(E_out)
        self.Decoder_layer=layers.Conv1D(128,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',name='DL2')(self.Decoder_layer)
        self.Decoder_layer=layers.Conv1D(128,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',name='DL3')(self.Decoder_layer)
        self.Decoder_layer=layers.Conv1D(128,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',name='DL4')(self.Decoder_layer)
        self.Decoder_layer=layers.UpSampling1D(polling_size,name='DL5')((self.Decoder_layer))
        self.Decoder_layer=layers.Conv1D(64,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',name='DL6')(self.Decoder_layer)
        self.Decoder_layer=layers.Conv1D(64,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',name='DL7')(self.Decoder_layer)
        self.Decoder_layer=layers.Conv1D(64,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',name='DL8')(self.Decoder_layer)
        self.Decoder_layer=layers.UpSampling1D(polling_size,name='DL9')((self.Decoder_layer))
        self.Decoder_layer=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',name='DL10')(self.Decoder_layer)
        self.Decoder_layer=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',name='DL11')(self.Decoder_layer)
        self.Decoder_layer=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',name='DL12')(self.Decoder_layer)
        self.Decoder_layer=layers.Flatten(name='DL13')((self.Decoder_layer))
        D_out=layers.Dense(self.data.shape[1]-1,kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Input_activ,name="Decoder_output")((self.Decoder_layer))
        self.Classifier_layer=layers.Dense(int(encoder_node_size/2),kernel_regularizer=regularizers.l1_l2(l1=Cl_L1_reg,l2=Cl_L2_reg),activation=Hidden_activ,name='CL2')(E_out)
        self.Classifier_layer=layers.Dense(int(encoder_node_size/4),kernel_regularizer=regularizers.l1_l2(l1=Cl_L1_reg,l2=Cl_L2_reg),activation=Hidden_activ,name='CL3')(self. Classifier_layer)
        cl_out=layers.Dense(int(len(self.uniques)),kernel_regularizer=regularizers.l1_l2(l1=Cl_L1_reg,l2=Cl_L2_reg),activation='softmax',name="Classifier_output")(self. Classifier_layer)
        self.mymodel=Model(En_inputs,[D_out,cl_out])
        losses = {"Decoder_output":"mse", "Classifier_output": "categorical_crossentropy"}
        mertics={"Decoder_output": "mse", "Classifier_output": 'accuracy'}
        lossWeights = {"Decoder_output": 1, "Classifier_output": 1}
        self.mymodel.compile(optimizer=optimizers.Adam(lr=Learning_rate),loss=losses,loss_weights=lossWeights,metrics=mertics)
        self.build_multi_Encoder_Decoder_separately_two()
    def estimate_mean_separately(self):
        my_test=npy.array(self.temps.iloc[:,:])
        my_uniques=npy.unique(my_test[:,0])
        data=npy.zeros((1,my_test.shape[1]))
        for i in range(len(my_uniques)):
            temp=my_test[my_test[:,0]==my_uniques[i]]
            temp[:,0]=i
            data=npy.concatenate((data,temp))
        my_test=data[1:,:]
        my_lat_rep=npy.zeros((1,self.encoder_node_size+1))
        my_uniques=self.uniques
        est_mean=npy.zeros((len(my_uniques),self.data.shape[1]-1))
        my_lat_mean=npy.zeros((len(my_uniques),self.encoder_node_size))
        for k in range(len(my_uniques)):
            temp=my_test[my_test[:,0]==my_uniques[k]]
            lat_predict=self.Encoder_layer.predict(temp[:,1:])
            class_lab=temp[:,0].reshape((temp.shape[0],1))
            lat_mean=npy.mean(lat_predict,0)
            my_lat_mean[k,:]=lat_mean.reshape((1,self.encoder_node_size))
            lat_predict=npy.concatenate((class_lab,lat_predict),axis=1)
            my_lat_rep=npy.concatenate((my_lat_rep,lat_predict),axis=0)
            est_mean[k,:]=self.Decoder_layer.predict(npy.array([lat_mean]))
        return my_lat_mean,my_lat_rep[1:,:],est_mean,my_test
    def build_multi_Encoder_Decoder_separately_two(self):
        self.mymodel.summary()
        self.encoder_node_size=self.mymodel.get_layer('Encoder_output').output_shape[1]
        En_inputs=Input(shape=(self.data.shape[1]-1,))
        self.Encoder_layer=self.mymodel.get_layer('EL1')(En_inputs)
        self.Encoder_layer=self.mymodel.get_layer('EL2')(self.Encoder_layer)
        self.Encoder_layer=self.mymodel.get_layer('EL3')(self.Encoder_layer)
        self.Encoder_layer=self.mymodel.get_layer('EL4')(self.Encoder_layer)
        self.Encoder_layer=self.mymodel.get_layer('EL5')(self.Encoder_layer)
        self.Encoder_layer=self.mymodel.get_layer('EL6')(self.Encoder_layer)
        self.Encoder_layer=self.mymodel.get_layer('EL7')(self.Encoder_layer)
        self.Encoder_layer=self.mymodel.get_layer('EL8')(self.Encoder_layer)
        self.Encoder_layer=self.mymodel.get_layer('EL9')(self.Encoder_layer)
        self.Encoder_layer=self.mymodel.get_layer('EL10')(self.Encoder_layer)
        self.Encoder_layer=self.mymodel.get_layer('EL11')(self.Encoder_layer)
        self.Encoder_layer=self.mymodel.get_layer('EL12')(self.Encoder_layer)
        self.Encoder_layer=self.mymodel.get_layer('EL13')(self.Encoder_layer)
        self.Encoder_layer=self.mymodel.get_layer('EL14')(self.Encoder_layer)
        self.Encoder_layer=self.mymodel.get_layer('Encoder_output')(self.Encoder_layer)
        self.Encoder_layer=Model(En_inputs, self.Encoder_layer)
        self.Encoder_layer.summary()
        decoder_inp_shape=Input(shape=(self.mymodel.get_layer('Encoder_output').output_shape[1],))
        self.Decoder_layer=Sequential()
        self.Decoder_layer=self.mymodel.get_layer('DL1')(decoder_inp_shape)
        self.Decoder_layer=self.mymodel.get_layer('DL2')(self.Decoder_layer)
        self.Decoder_layer=self.mymodel.get_layer('DL3')(self.Decoder_layer)
        self.Decoder_layer=self.mymodel.get_layer('DL4')(self.Decoder_layer)
        self.Decoder_layer=self.mymodel.get_layer('DL5')(self.Decoder_layer)
        self.Decoder_layer=self.mymodel.get_layer('DL6')(self.Decoder_layer)
        self.Decoder_layer=self.mymodel.get_layer('DL7')(self.Decoder_layer)
        self.Decoder_layer=self.mymodel.get_layer('DL8')(self.Decoder_layer)
        self.Decoder_layer=self.mymodel.get_layer('DL9')(self.Decoder_layer)
        self.Decoder_layer=self.mymodel.get_layer('DL10')(self.Decoder_layer)
        self.Decoder_layer=self.mymodel.get_layer('DL11')(self.Decoder_layer)
        self.Decoder_layer=self.mymodel.get_layer('DL12')(self.Decoder_layer)
        self.Decoder_layer=self.mymodel.get_layer('DL13')(self.Decoder_layer)
        self.Decoder_layer=self.mymodel.get_layer('Decoder_output')(self.Decoder_layer)
        self.Decoder_layer=Model(decoder_inp_shape, self.Decoder_layer)
        self.Decoder_layer.summary()
    def load_model(self):
        if os.path.isfile:
            #self.mymodel.load_weights(self.Model_save_path+self.Model_save_name)
            historys={}
            checkpath=self.Model_save_path+self.Model_save_name
            my_mod=checkpath.replace('.ckpt','')
            print(my_mod)
            self.mymodel=load_model(my_mod+'.h5')
            print(self.mymodel)
            with(open(checkpath+'train_hsitoryconv '+'.txt', 'rb')) as file_pi:
                historys=pickle.load(file_pi)
            self.hist_dict=historys
            print('Loading complete.')
        else:
            print('Model weights not found! Check model path and name. ')
    def train_model(self,user_epochs,bathc_size_factor,unique_id=''):   
         checkpath=self.Model_save_path+self.Model_save_name+str(unique_id)
         cp_store=ModelCheckpoint(checkpath,save_weights_only=True,verbose=1)
         start_time=time.time()
         calc_valid=int(self.temp.shape[0]*self.validation)
         historys=''
         if calc_valid==0: 
             historys=self.mymodel.fit(self.data[:,1:],self.data[:,1:],epochs=user_epochs,verbose=1,batch_size=int((self.data.shape[0])/bathc_size_factor),callbacks=[cp_store])
             with(open(checkpath+'train_hsitoryconv '+'.txt', 'wb')) as file_pi:
                 pickle.dump(historys.history, file_pi)
             self.hist_dict=historys.history
             self.training_time=time.time()-start_time
             model_paths=checkpath.replace('.ckpt','')
             self.mymodel.save(model_paths+'.h5')
         else:
             historys=self.mymodel.fit(self.data[:,1:],self.data[:,1:],validation_data=(self.validation_data[:,1:],self.validation_data[:,1:]),epochs=user_epochs,verbose=1,batch_size=int((self.data.shape[0])/bathc_size_factor),callbacks=[cp_store])
             with(open(checkpath+'train_hsitoryconv '+'.txt', 'wb')) as file_pi:
                 pickle.dump(historys.history, file_pi)
             self.hist_dict=historys.history
             self.training_time=time.time()-start_time
             model_paths=checkpath.replace('.ckpt','')
             self.mymodel.save(model_paths+'.h5')
         if os.path.isfile(checkpath+' Comment.txt'):
             with(open(checkpath+' Comment.txt', 'a')) as file_pi:
                 file_pi.write('Time taken to train the network:'+str(self.training_time))
         else:
              with(open(checkpath+' Comment.txt', 'w')) as file_pi:
                 file_pi.write('Time taken to train the network:'+str(self.training_time))
         if os.path.isfile(checkpath+' losses.txt'):
             leng=len(self.hist_dict['val_loss'])
             with(open(checkpath+' losses.txt', 'a')) as file_pi:
                 if self.validation>0:
                     file_pi.write('Final Validation Loss'+str(self.hist_dict['val_loss'][leng-1])+'Final training loss'+str(self.hist_dict['loss'][leng-1])+'Final Decoder mse'+str(self.hist_dict['mse'][leng-1])+'Final Decoder val mse'+str(self.hist_dict['val_mse'][leng-1]))
                 else:
                     file_pi.write('Final training loss'+str(self.hist_dict['loss'][leng-1])+'Final Decoder mse'+str(self.hist_dict['mse'][leng-1]))
         else:
              leng=len(self.hist_dict['val_loss'])
              with(open(checkpath+' losses.txt', 'w')) as file_pi:
                 if self.validation>0:
                     file_pi.write('Final Validation Loss'+str(self.hist_dict['val_loss'][leng-1])+'Final training loss'+str(self.hist_dict['loss'][leng-1])+'Final Decoder mse'+str(self.hist_dict['mse'][leng-1])+'Final Decoder val mse'+str(self.hist_dict['val_mse'][leng-1]))
                 else:
                     file_pi.write('Final training loss'+str(self.hist_dict['loss'][leng-1])+'Final Decoder mse'+str(self.hist_dict['mse'][leng-1]))
         print('Time taken for training in seconds: ', self.training_time)
    def train_multi_model(self,user_epochs,bathc_size_factor,unique_id=''):   
         checkpath=self.Model_save_path+self.Model_save_name+str(unique_id)
         cp_store=tf.keras.callbacks.ModelCheckpoint(checkpath,save_weights_only=True,verbose=1)
         start_time=time.time()
         calc_valid=int(self.temp.shape[0]*self.validation)
         historys=''
         if calc_valid==0: 
             class_labels=self.data[:,0].reshape((self.data.shape[0],))
             class_labels=to_categorical(class_labels,num_classes=len(self.uniques))
             historys=self.mymodel.fit(self.data[:,1:],{"Decoder_output": self.data[:,1:], "Classifier_output": class_labels},epochs=user_epochs,verbose=1,batch_size=int((self.data.shape[0])/bathc_size_factor),callbacks=[cp_store])
             with(open(checkpath+'train_hsitoryconv '+'.txt', 'wb')) as file_pi:
                 pickle.dump(historys.history, file_pi)
             self.hist_dict=historys.history
             self.training_time=time.time()-start_time
             model_paths=checkpath.replace('.ckpt','')
             self.mymodel.save(model_paths+'.h5')
         else:
             class_labels=self.data[:,0].reshape((self.data.shape[0],))
             class_labels=to_categorical(class_labels,num_classes=len(self.uniques))
             val_class_labels=self.validation_data[:,0].reshape((self.validation_data.shape[0],))
             val_class_labels=to_categorical(val_class_labels,num_classes=len(self.uniques))
             historys=self.mymodel.fit(self.data[:,1:],{"Decoder_output": self.data[:,1:], "Classifier_output": class_labels},validation_data=(self.validation_data[:,1:],{"Decoder_output": self.validation_data[:,1:], "Classifier_output": val_class_labels}),epochs=user_epochs,verbose=1,batch_size=int((self.data.shape[0])/bathc_size_factor),callbacks=[cp_store])
             with(open(checkpath+'train_hsitoryconv '+'.txt', 'wb')) as file_pi:
                 pickle.dump(historys.history, file_pi)
             self.hist_dict=historys.history
             self.training_time=time.time()-start_time
             model_paths=checkpath.replace('.ckpt','')
             self.mymodel.save(model_paths+'.h5')
         with(open(checkpath+'train_hsitoryconv '+'.txt', 'wb')) as file_pi:
             pickle.dump(historys.history, file_pi)
         self.hist_dict=historys.history
         self.training_time=time.time()-start_time
         if os.path.isfile(checkpath+' Comment.txt'):
             with(open(checkpath+' Comment.txt', 'a')) as file_pi:
                 file_pi.write('Time taken to train the network:'+str(self.training_time))
         else:
              with(open(checkpath+' Comment.txt', 'w')) as file_pi:
                 file_pi.write('Time taken to train the network:'+str(self.training_time))
         print('Time taken for training in seconds: ', self.training_time)
         if os.path.isfile(checkpath+' losses.txt'):
             leng=len(self.hist_dict['val_Decoder_output_loss'])
             with(open(checkpath+' losses.txt', 'a')) as file_pi:
                 if self.validation>0:
                     file_pi.write('Final Validation Loss'+str(self.hist_dict['val_loss'][leng-1])+'Final training loss'+str(self.hist_dict['loss'][leng-1])+'Final Decoder loss'+str(self.hist_dict['Decoder_output_loss'][leng-1])+'Final Classifer loss'+str(self.hist_dict['Classifier_output_loss'][leng-1])+'Final Decoder val_loss'+str(self.hist_dict['val_Decoder_output_loss'][leng-1])+'Final Classifier_val_loss'+str(self.hist_dict['val_Classifier_output_loss'][leng-1]))
                 else:
                     file_pi.write('Final training loss'+str(self.hist_dict['loss'][leng-1])+'Final Decoder loss'+str(self.hist_dict['Decoder_output_loss'][leng-1])+'Final Classifer loss'+str(self.hist_dict['Classifier_output_loss'][leng-1]))
         else:
              leng=len(self.hist_dict['val_Decoder_output_loss'])
              with(open(checkpath+' losses.txt', 'w')) as file_pi:
                 if self.validation>0:
                     file_pi.write('Final Validation Loss'+str(self.hist_dict['val_loss'][leng-1])+'Final training loss'+str(self.hist_dict['loss'][leng-1])+'Final Decoder loss'+str(self.hist_dict['Decoder_output_loss'][leng-1])+'Final Classifer loss'+str(self.hist_dict['Classifier_output_loss'][leng-1])+'Final Decoder val_loss'+str(self.hist_dict['val_Decoder_output_loss'][leng-1])+'Final Classifier_val_loss'+str(self.hist_dict['val_Classifier_output_loss'][leng-1]))
                 else:
                     file_pi.write('Final training loss'+str(self.hist_dict['loss'][leng-1])+'Final Decoder loss'+str(self.hist_dict['Decoder_output_loss'][leng-1])+'Final Classifer loss'+str(self.hist_dict['Classifier_output_loss'][leng-1]))
         print('Time taken for training in seconds: ', self.training_time)
    def estimate_class_avg(self,multiple):
        my_lat_means=npy.zeros((1,self.encoder_node_size))
        if multiple==0:
            for k in range(len(self.uniques)):
                data=self.data[self.data[:,0]==self.uniques[k]]
                data=data.reshape((data.shape[0],self.data.shape[1]))
                latents=self.Encoder_layer.predict(data[:,1:])
                latent_means=npy.mean(latents[:,:],axis=0)
                self.final_estimate[k,:]=self.Decoder_layer.predict(npy.array([latent_means]))
                latent_means=latent_means.reshape((1,self.encoder_node_size))
                my_lat_means=npy.concatenate((my_lat_means,latent_means),axis=0)
        else:
            data=self.data[:,1:]
            latents=self.Encoder_layer.predict(data[:,:])
            latent_means=npy.mean(latents,axis=0)
            self.final_estimate[0,:]=self.Decoder_Layer.predict(latent_means)
            latent_means=latent_means.reshape((1,self.encoder_node_size))
            my_lat_means=npy.concatenate((my_lat_means,latent_means),axis=0)
        print('Estimation complete')
        return my_lat_means[1:,:]
    def Calculate_WGSS(self,series,multiple):
            means=self.final_estimate
            series_means=npy.mean(series[:,1:],axis=1).reshape((series.shape[0],1))
            series_sd=npy.std(series[:,1:],axis=1).reshape((series.shape[0],1))
            means_mean=npy.mean(means,axis=1).reshape((means.shape[0],1))
            means_sd=npy.std(means,axis=1).reshape((means.shape[0],1))
            normalized_seri=(series[:,1:]-series_means)/series_sd
            noramlized_mean=(means-means_mean)/means_sd
            temp=npy.zeros((1,len(self.uniques)))
            res=0
            curr_index=0
            calc=0
            if multiple==0:
                for i in range(len(self.uniques)):
                    for j in range(normalized_seri[curr_index:curr_index+self.counts[i],:].shape[0]):
                         aligned=mypair.DTW(normalized_seri[curr_index+j,:],noramlized_mean[i,:])
                         aligned.calcglobalcost_UDTW()
                         aligned.findwarppath()
                         res=res+math.sum(math.square(aligned.Warpedfv-aligned.Warpedsv))
                    temp[0,i]=res
                    res=0
                    curr_index=curr_index+self.counts[i]
                calc=math.sum(temp,axis=1)/(self.data.shape[1]-1)
            else:
                for j in range(normalized_seri.shape[0]):
                     aligned=mypair.DTW(normalized_seri[j,:],noramlized_mean[0,:])
                     aligned.calcglobalcost_UDTW()
                     aligned.findwarppath()
                     res=res+math.sum(math.square(aligned.Warpedfv-aligned.Warpedsv))
                calc=res/(self.data.shape[1]-1)
            return calc
    def write_means(self,WGSS_mean=1,res=0,save_name=' estimated_means.tsv'):
        means=self.final_estimate
        Filename=self.Model_save_path+self.File_name+save_name
        if os.path.isfile(Filename):
             with open(Filename, 'a', newline='') as tsv_file:
                print('opened file in',Filename )
                tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
                if WGSS_mean==1:
                    for i in range(means.shape[0]):
                        tsv_writer.writerow(self.final_estimate[i,:])
                else:
                    tsv_writer.writerow([res])
        else:
            with open(Filename, 'w', newline='') as tsv_file:
                print('opened file in',Filename )
                tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
                if WGSS_mean==1:
                    for i in range(means.shape[0]):
                        tsv_writer.writerow(self.final_estimate[i,:])
                else:
                    tsv_writer.writerow([res])
    def write_atsv_file(self,array,file_loc,file_name):
        Filename=file_loc+file_name
        for i in range(array.shape[0]):
            with open(Filename, 'a', newline='') as tsv_file:
                  print('opened file in',Filename )
                  tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
                  tsv_writer.writerow(array[i,:])
    def display_results(self,model):
        fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2)
        fig.suptitle('Number of data sets'+str(self.data.shape)+' ,'+'Validation data size'+
                      'Training time in minutes:'+
                     str(self.training_time/60)+' ,'+'Number of classes:'+str(len(self.uniques)))
        hist_dic=self.hist_dict
        if model==0:
            ax1.plot(range(len(hist_dic['loss'])),hist_dic['loss'],'r',label='Training Loss')
            if self.validation>0:
                ax1.plot(range(len(hist_dic['val_loss'])),hist_dic['val_loss'],'r',label='Validation Loss')
        else:
            ax1.plot(range(len(hist_dic['Decoder_output_loss'])),hist_dic['Decoder_output_loss'],label='Decoder_output_loss')
            ax1.plot(range(len(hist_dic['Classifier_output_loss'])),hist_dic['Classifier_output_loss'],label='Classifier_output_loss')
            if self.validation>0:
                ax1.plot(range(len(hist_dic['val_Decoder_output_loss'])),hist_dic['val_Decoder_output_loss'],label='Validation Decoder_output_loss')
                ax1.plot(range(len(hist_dic['val_Classifier_output_loss'])),hist_dic['val_Classifier_output_loss'],label='Validation Classifier_output_loss')
                leng=len(hist_dic['val_Decoder_output_loss'])
                print('Final Validation Loss',hist_dic['val_loss'][leng-1])
                print('Loss',hist_dic['loss'][leng-1])
                print('Classifier_loss',hist_dic['Classifier_output_loss'][leng-1])
                print('Decoder_loss',hist_dic['Decoder_output_loss'][leng-1])
                print('Classifier_val_loss',hist_dic['val_Classifier_output_loss'][leng-1])
                print('Decoder_val_loss',hist_dic['val_Decoder_output_loss'][leng-1])
        ax1.grid(axis='both')
        ax1.legend()
        ax1.title.set_text('Validation and Training Loss')
        for k in range(self.data.shape[0]):
           ax2.plot(self.data[k,1:],'r')
        ax2.grid(axis='both')
        ax2.legend()
        ax2.title.set_text('Data sets')
        for k in range(self.final_estimate.shape[0]):
            print(k)
            ax3.plot(self.final_estimate[k,:],label='Average for class '+str(k+1))
        ax3.grid(axis='both')
        ax3.legend()
        ax3.title.set_text('Estimated averages')
        calc_valid=int(self.validation*self.temp.shape[0])
        my_data=npy.zeros((1,1))
        if calc_valid>0:
            my_data=npy.concatenate((self.data,self.validation_data))
        else:
            my_data=self.data
        for j in range(len(self.uniques)):
            plt.figure()
            data=my_data[my_data[:,0]==self.uniques[j]]
            for k in range(data.shape[0]):
                plt.plot(data[k,:],'r')
            plt.title('Class '+str(j+1)+' Data and estimated average')
            plt.grid(axis='both')
            plt.plot(self.final_estimate [j,:],'b',linewidth=4)
        data2=npy.array(self.temps.iloc[:,:])
        temp=npy.zeros((1,self.data.shape[1]))
        for i in range(len(self.uniques)):
            temps=data2[data2[:,0]==self.my_uniqes_copy[i]]
            temps[:,0]=i
            temp=npy.concatenate((temp,temps),axis=0)
        data2=temp[1:,:]
        for j in range(len(self.uniques)):
            plt.figure()
            data=data2[data2[:,0]==self.uniques[j]]
            for k in range(data.shape[0]):
                plt.plot(data[k,:],'r')
            plt.title('Test Class '+str(j+1)+' Data and estimated average')
            plt.grid(axis='both')
            plt.plot(self.final_estimate [j,:],'b',linewidth=4)
        for j in range(len(self.uniques)):
            plt.figure()
            data=data2[data2[:,0]==self.uniques[j]]
            for k in range(data.shape[0]):
                plt.plot(data[k,:],'r')
            plt.title('Test Class '+str(j+1)+' Data')
            plt.grid(axis='both')
        for j in range(len(self.uniques)):
            plt.figure()
            data=my_data[my_data[:,0]==self.uniques[j]]
            data=self.Encoder_layer.predict(data[:,1:])
            data=self.Decoder_layer.predict(data)
            for k in range(data.shape[0]):
                plt.plot(data[k,:],'r')
            plt.title('Decoder output for train class '+str(j+1))
            plt.grid(axis='both')
        for j in range(len(self.uniques)):
            plt.figure()
            data=data2[data2[:,0]==self.uniques[j]]
            data=self.Encoder_layer.predict(data[:,1:])
            data=self.Decoder_layer.predict(data)
            for k in range(data.shape[0]):
                plt.plot(data[k,:],'r')
            plt.title('Decoder output for test class '+str(j+1))
            plt.grid(axis='both')
    def disp_compressed_dim(self,classes,data,avgs,strg='time domain'):
        plt.figure()
        count=0
        temps=StandardScaler().fit_transform(npy.concatenate((avgs,data[:,1:]),axis=0))
        count=count+1
        temp=TSNE(n_components=2).fit_transform(temps)
        temp_data=temp[len(classes):,:]
        temp_mean=temp[0:len(classes),:]
        for j in classes:
            my_temp=temp_data[data[:,0]==j]
            plt.scatter(my_temp[:,0],my_temp[:,1])
            plt.scatter(temp_mean[j,0],temp_mean[j,1],linewidths=20,label='Class '+ str(j+1)+' avg.')
        plt.xlabel('Dimension 1 (x)')
        plt.ylabel('Dimension 2 (y)')
        plt.grid(axis='both')
        plt.title('Two dimensional tSNE plot of '+strg+' representation')
        plt.legend()
        plt.show()