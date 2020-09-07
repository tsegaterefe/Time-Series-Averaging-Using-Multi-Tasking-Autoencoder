# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 08:18:44 2019

@author: Tsega
"""
#import DTW as mypair
import numpy as npy
import seaborn as sn
import pandas as pd
import os
import scipy as mycalc
import scipy as math
from sklearn.decomposition import PCA
from tslearn import metrics
class One_NN:
    get_ipython().run_line_magic('matplotlib','qt')
    mytestseries=npy.array
    myclassavgs=npy.array
    target_class=npy.array
    zero_indexed_traget_class=npy.array
    confusion_matrix=npy.array
    class_distance=npy.array
    countof_classmember=npy.array
    file_location=''
    file_name=''
    mypath=''
    unique_classes=[]
    def __init__(self,class_centroids,f_loc,f_name,types,data=npy.array):
      if types==1:
        print('Classification Mode:',1)
        self.file_location=f_loc
        self.file_name=f_name
        self.mypath=os.path.join(self.file_location,self.file_name)
        test_series=pd.read_csv(self.mypath,sep='\t',header=None)
        self.mytestseries=npy.array(test_series.iloc[0:,1:])
        self.myclassavgs=class_centroids
        self.target_class=npy.array(test_series.iloc[0:,0],dtype=int)
        self.zero_indexed_traget_class=npy.copy(self.target_class)
        self.unique_classes=list(npy.unique(self.target_class))
        self.countof_classmember=npy.zeros((1,len(self.unique_classes)))
        for i in range(self.target_class.shape[0]):
            self.zero_indexed_traget_class[i]=self.unique_classes.index(self.target_class[i])
        for i in range(self.unique_classes.shape[0]):
            self.countof_classmember[0,i]=list(self.zero_indexed_traget_class).count(i)
        self.confusion_matrix=npy.zeros((len(self.unique_classes),len(self.unique_classes)))
        self.class_distance=npy.zeros((1,len(self.unique_classes)))
      else:
        print('Classification Mode:',types)
        self.mytestseries=data[0:,1:]
        self.myclassavgs=class_centroids
        self.target_class=npy.array(data[0:,0],dtype=int)
        self.zero_indexed_traget_class=npy.copy(self.target_class)
        self.unique_classes=list(npy.unique(self.target_class))
        self.countof_classmember=npy.zeros((1,len(self.unique_classes)))
        for i in range(self.target_class.shape[0]):
            self.zero_indexed_traget_class[i]=self.unique_classes.index(self.target_class[i])
        for i in range(len(self.unique_classes)):
            self.countof_classmember[0,i]=list(self.zero_indexed_traget_class).count(i)
        self.confusion_matrix=npy.zeros((len(self.unique_classes),len(self.unique_classes)))
        self.class_distance=npy.zeros((1,len(self.unique_classes)))
      print('*********************1NN information****************************')
      print('Total Data sets:',self.mytestseries.shape)
      print('Total Averages:',self.myclassavgs.shape)
      print('Tota target classes:',self.target_class.shape)
      print('Target class values:',self.target_class)
      print('Unique target class values:',self.unique_classes)
      print('zero indexed target class:',self.zero_indexed_traget_class)
      print('number of different classes:',self.class_distance.shape[1])
      print('Size of confusion matrix:',self.confusion_matrix.shape[0])
    #if problems are encountered with tslearn we have provided implmentation of DTW. Uncomment the lines of codes and the import line to acivate the DTW script.
    def classify_DTW(self):
        correct_count=0
        for i in range(self.mytestseries.shape[0]):
            for j in range(self.class_distance.shape[1]):
                #align=mypair.DTW(self.mytestseries[i,:],self.myclassavgs[j,:])
                #align.calcglobalcost_UDTW()
                #align.findwarppath()
                self.class_distance[0,j]=metrics.dtw(self.mytestseries[i,:],self.myclassavgs[j,:])#math.sqrt(sum(math.square(align.Warpedfv-align.Warpedsv)))
            classified_as=list(self.class_distance[0]).index(min(self.class_distance[0]))
            print('distance from each classes:',self.class_distance)
            if classified_as==self.zero_indexed_traget_class[i]:
                print('Data:',i,' target:',self.zero_indexed_traget_class[i],' classified as:',classified_as)
                self.confusion_matrix[classified_as,classified_as]+=1
                correct_count=correct_count+1
            else:
                print('Data:',i,' target:',self.zero_indexed_traget_class[i],' classified as:',classified_as)
                self.confusion_matrix[self.zero_indexed_traget_class[i],classified_as]+=1
        return (correct_count/self.mytestseries.shape[0])*100
    def classify_Euclidean(self):
        correct_count=0
        for i in range(self.mytestseries.shape[0]):
            for j in range(self.class_distance.shape[1]):
                self.class_distance[0,j]=mycalc.sqrt(sum(mycalc.square(self.mytestseries[i,:]-self.myclassavgs[j,:])))
            #print('distance from each classes:',self.class_distance)
            classified_as=list(self.class_distance[0]).index(min(self.class_distance[0]))
            print('distance from each classes:',self.class_distance)
            if classified_as==self.zero_indexed_traget_class[i]:
                print('Data:',i,' target:',self.zero_indexed_traget_class[i],' classified as:',classified_as)
                self.confusion_matrix[classified_as,classified_as]+=1
                correct_count=correct_count+1
            else:
                print('Data:',i,' target:',self.zero_indexed_traget_class[i],' classified as:',classified_as)
                self.confusion_matrix[self.zero_indexed_traget_class[i],classified_as]+=1
        return (correct_count/self.mytestseries.shape[0])*100
    def classify_Euclidean2(self,train_latents):
        prevdistance=-1
        for i in range(self.mytestseries.shape[0]):
            print('Target class:',self.zero_indexed_traget_class[i])
            for j in range(self.mytestseries.shape[0]):
                if i!=j:
                    currentdistance=mycalc.sqrt(sum(mycalc.square(self.mytestseries[i,:]-train_latents[j,1:])))
                    #print('current distance',currentdistance,'Prev distance',prevdistance)
                    if currentdistance<prevdistance or prevdistance<0:
                        print('I am in if','current distance',currentdistance,'Prev distance',prevdistance)
                        print('Target class',self.zero_indexed_traget_class[i],'classified as',self.zero_indexed_traget_class[j])
                        prevdistance=currentdistance
                        classified_as=self.unique_classes.index(train_latents[j,0])
                    #classified_as=list(self.class_distance[0]).index(min(self.class_distance[0]))
            if classified_as==self.zero_indexed_traget_class[i]:
                        print('Data:',i,' target:',self.zero_indexed_traget_class[i],' classified as:',classified_as)
                        self.confusion_matrix[classified_as,classified_as]+=1
            else:
                        print('Data:',i,' target:',self.zero_indexed_traget_class[i],' classified as:',classified_as)
                        self.confusion_matrix[self.zero_indexed_traget_class[i],classified_as]+=1
            prevdistance=-1
