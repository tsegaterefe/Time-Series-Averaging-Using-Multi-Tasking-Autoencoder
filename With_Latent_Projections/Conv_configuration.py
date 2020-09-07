# -*- coding: utf-8 -*-
"""
@author: Tsegamlak Terefe
"""
"""
The configuration parameters are described as follows:
    encoder_node_size-> The dimension of the latent space
    Input_activ -> Input layer activation
    Hidden_activ-> Hidden layers activation 
    filter_size -> Convolutional layers kernel size
    pooling_size-> pooling size for max polling layers
    batch_size-> Batch factor for training, i.e., Batch=input data row dimension/batch_size
    File_loc-> location of file for training, i.e., the UCR data set locatin
    File_name-> Name of file for training, i.e., excluding the _TRAIN.tst or _TEST.tsv substring (it is automatically rade from a list from a CSV file) 
    classif_mode-> 0= All classes within the data set are used for training, 
                   1= Training is conducted on a selected class
    select_all -> 0=only data sets within the xxx_TRAIN.tsv are used for training the network
                  1=data sets witin xxx_TEST.tsv and xxx_TRAIN.tsv are mearged together to train the network
    Class_lable-> index of the class selected for training, if per class training is issued
    First_time_training -> 1= Model is built and traind from scratch
                           0= Model is built but weights are loaded with a xxx.ckpt file located in model save path directory
    Model_Save_Path -> Directory to save models and their corresponding files which could be used to load model with weights 
    Model_save_name -> The name of the xxx.h5 and other files, i.e., model and other files will be saved as /Model_save_name/xxxx.h5 (the name is automaitcally rade from the csv list)
    Model_type -> Type of selected model 0:VGG plain encoder 1:VGG multitasking encoder
    List_of_data_sets -> CSV file with the names of UCR data sets to run the loop on
    List_of_data_sets_FName -> The name of the CSV file with the list (.csv included)
    Epochs -> List of epoches for the five trials
    reg_config1 and reg_config2 -> Regulaizarions per epoch for the classifier and encoder-decoder
"""
class Configurations:
    encoder_node_size=0
    Input_activ=''
    Hidden_activ=''
    encoder_node_size=0
    filter_size=3
    polling_size=3
    En_L1_reg=0
    En_L2_reg=0
    De_L1_reg=0
    De_L2_reg=0
    Cl_L1_reg=0
    Cl_L2_reg=0
    batch_size=1
    Epoch=600
    Learning_rate=0.00001
    File_loc=''
    File_name=''
    Model_save_path=''
    Model_save_name=''
    classif_mode=1
    select_all=1
    class_label=0
    First_time_train=0
    model_type=0
    validation_size=0
    reg_config1=[0.0000,0.0001,0.001,0.001,0.01]
    reg_config2=[0.0000,0.001,0.001,0.01,0.01]
    Epochs=[600,2500,2500,2500,2500]
    List_of_data_sets=''
    List_of_data_sets_FName=''
    def __init__(self):
            self.reg_config1=[0.0000,0.0001,0.001,0.001,0.01]
            self.reg_config2=[0.0000,0.001,0.001,0.01,0.01]
            self.Epochs=[600,2500,2500,2500,2500]
            self.encoder_node_size=0
            self.Input_activ='linear'
            self.Hidden_activ='relu'
            self.encoder_node_size=0
            self.filter_size=3
            self.polling_size=3
            self.En_L1_reg=0.0
            self.En_L2_reg=0.0000
            self.De_L1_reg=0.0
            self.De_L2_reg=0.0000
            self.Cl_L1_reg=0.0
            self.Cl_L2_reg=0.000
            self.batch_size=2
            self.Epoch=600
            self.Learning_rate=0.0001
            self.File_loc='D:/Deep learning data/UCR_Data_sets/UCRArchive_2018/'
            self.File_name=''
            self.Model_save_path='D:/2020 papers/UCR datas/UCR Ncentroid Multilearning Auto/'
            self.List_of_data_sets='D:/2020 papers/Clean version code/'
            self.List_of_data_sets_FName='Dataset_List.csv'
            self.Model_save_name=''
            self.classif_mode=0
            self.select_all=0
            self.class_label=1
            self.First_time_train=0
            self.validation_size=0.20
            self.model_type=1