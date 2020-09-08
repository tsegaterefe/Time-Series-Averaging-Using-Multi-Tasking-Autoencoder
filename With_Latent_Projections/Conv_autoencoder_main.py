# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 09:27:22 2020

@author: Tsega
"""
import Conv_autoencoder as encoder
import Conv_configuration as my_configs
import time
import One_NN as myNN
import numpy as npy
import pandas as pd
from importlib import reload  
import os
def main():
    my_config=my_configs.Configurations()
    Files_to_execute_for=pd.read_csv(my_config.List_of_data_sets+my_config.List_of_data_sets_FName, sep=',',header=None)
    root=my_config.File_loc
    model_save_root=my_config.Model_save_path
    for i in range(Files_to_execute_for.shape[0]):
        temp=Files_to_execute_for.iloc[i]
        temp=temp.to_string(index=False)
        temp=temp.replace(' ','')
        my_config.File_loc=root+temp+'/'
        my_config.File_name=temp
        my_config.Model_save_name=temp+'.ckpt'
        my_config.Model_save_path=model_save_root
        try:
            os.mkdir(my_config.Model_save_path+my_config.File_name)
        except OSError:
            print ("Creation of the directory %s failed" % my_config.Model_save_path+my_config.File_name)
        else:
            print ("Successfully created the directory %s " % my_config.Model_save_path+my_config.File_name)
        for j in range(my_config.iterations):
            my_config.Model_save_path=model_save_root
            try:
                os.mkdir(my_config.Model_save_path+my_config.File_name+'/'+'Trial'+str(j+1))
            except OSError:
                print ("Creation of the directory %s failed" % my_config.Model_save_path+my_config.File_name+'/'+'Trial'+str(j+1))
            else:
                print ("Successfully created the directory %s " %my_config.Model_save_path+my_config.File_name+'/'+'Trial'+str(j+1))
            my_config.Model_save_path=my_config.Model_save_path+my_config.File_name+'/'+'Trial'+str(j+1)+'/'
            my_config.Cl_L2_reg=my_config.reg_config2[j]
            my_config.De_L2_reg=my_config.reg_config1[j]
            my_config.En_L2_reg=my_config.reg_config1[j]
            my_config.Epoch=my_config.Epochs[j]
            print('Analyzing data sets........................')
            mydatas=encoder.conv_auto_encoder(my_config.File_loc,my_config.File_name,my_config.Model_save_path, my_config.Model_save_name,my_config.classif_mode,my_config.select_all,my_config.class_label,my_config.validation_size)
            print('Total data sets in train file:',mydatas.temp.shape[0],mydatas.temp.shape[1])
            print('Total data sets in test file:',mydatas.temps.shape[0],mydatas.temp.shape[1])
            print('Unique classes:', mydatas.uniques)
            print('Total members of classes in the concatinated data set:', mydatas.counts)
            print('Finished analyzing data sets.')
            my_config.encoder_node_size=int((mydatas.temp.shape[1]-1)/4)
            if my_config.First_time_train==1:
                if my_config.model_type==0:
                    mydatas.build_network_vgg(my_config.encoder_node_size,my_config.filter_size, my_config.polling_size, my_config.En_L1_reg, my_config.En_L2_reg,my_config.De_L1_reg,my_config.De_L2_reg,my_config.Cl_L1_reg,my_config.Cl_L2_reg, my_config.Input_activ, my_config.Hidden_activ,my_config.Learning_rate)
                    mydatas.train_model(my_config.Epoch,my_config.batch_size)
                else:
                    mydatas.Multi_task_encoder_two(my_config.encoder_node_size,my_config.filter_size, my_config.polling_size, my_config.En_L1_reg, my_config.En_L2_reg,my_config.De_L1_reg,my_config.De_L2_reg,my_config.Cl_L1_reg,my_config.Cl_L2_reg, my_config.Input_activ, my_config.Hidden_activ,my_config.Learning_rate)
                    mydatas.train_multi_model(my_config.Epoch,my_config.batch_size)
            else:
                if my_config.model_type==0:
                    mydatas.load_model()
                    mydatas.build_multi_Encoder_Decoder_separately_two()
                else:
                    mydatas.load_model()
                    mydatas.build_multi_Encoder_Decoder_separately_two()
            if my_config.classif_mode==0:
                start=time.time()
                Train_lat_means=mydatas.estimate_class_avg(my_config.classif_mode)
                end=time.time()
                mydatas.write_means(0,end-start,' time taken to estimate mean.tsv')
                mydatas.write_means()
                class_labels=mydatas.data[:,0].reshape(mydatas.data.shape[0],1)
                my_lat=mydatas.Encoder_layer.predict(mydatas.data[:,1:])
                my_lat=npy.concatenate((class_labels,my_lat),axis=1)
                mydatas.disp_compressed_dim(mydatas.uniques,my_lat,Train_lat_means,'Train latent space')
                mydatas.disp_compressed_dim(mydatas.uniques,mydatas.data, mydatas.final_estimate,'Train time domain')
                display=[]
                if my_config.select_all==0 and my_config.classif_mode==0:
                    lat_mean,latents,test_mean,test_data=mydatas.estimate_mean_separately()
                    myNN_classif=myNN.One_NN(mydatas.final_estimate,'null','null',0,data=test_data)
                    res=myNN_classif.classify_DTW()
                    display.append(res)
                    print('Begnining Latent NCC classification........')
                    myNN_classif=myNN.One_NN(Train_lat_means,'null','null',0,data=latents)
                    res=myNN_classif.classify_Euclidean()
                    display.append(res)
                    print("Latent and Time domain Results:",display)
                    mydatas.write_atsv_file(npy.array(display).reshape((1,2)),my_config.Model_save_path,'Time domain Latent classification result.tsv')
                    mydatas.disp_compressed_dim(mydatas.uniques,latents,Train_lat_means,'Test latent space')
                    mydatas.disp_compressed_dim(mydatas.uniques,test_data,mydatas.final_estimate,'Test time domain')
            else:
                start=time.time()
                Train_lat_means=mydatas.estimate_class_avg(my_config.classif_mode)
                end=time.time()
                mydatas.write_means(0,end-start,' time taken to estimate mean.tsv')
                mydatas.write_means()
                my_lat=mydatas.Encoder_layer.predict(mydatas.data[:,1:])
                my_lat=npy.concatenate((class_labels,my_lat),axis=1)
                mydatas.disp_compressed_dim(mydatas.uniques,my_lat,Train_lat_means,'Train latent space')
                mydatas.disp_compressed_dim(mydatas.uniques,mydatas.data, mydatas.final_estimate,'Train time domain')
                display=[]
                if my_config.select_all==0 and my_config.classif_mode==0:
                    lat_mean,latents,test_mean,test_data=mydatas.estimate_mean_separately()
                    myNN_classif=myNN.One_NN(mydatas.final_estimate,'null','null',0,data=test_data)
                    res=myNN_classif.classify_DTW()
                    display.append(res)
                    print('Begnining Latent NCC classification........')
                    myNN_classif=myNN.One_NN(Train_lat_means,'null','null',0,data=latents)
                    res=myNN_classif.classify_Euclidean()
                    display.append(res)
                    print("Latent and Time domain Results:",display)
                    mydatas.write_atsv_file(npy.array(display).reshape((1,2)),my_config.Model_save_path,'Time domain Latent classification result.tsv')
                    mydatas.disp_compressed_dim(mydatas.uniques,latents,Train_lat_means,'Test latent space')
                    mydatas.disp_compressed_dim(mydatas.uniques,test_data,mydatas.final_estimate,'Test time domain')
            reload(encoder)
            reload(myNN)
if __name__ == "__main__":
        main()
