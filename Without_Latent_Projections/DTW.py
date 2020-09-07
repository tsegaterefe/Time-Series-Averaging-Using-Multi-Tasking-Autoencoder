# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 15:59:39 2019
The code aims at calculating variant implementations of DTW: DTW,WDTW,SDTW,DDTW
@author: Tsega
"""
import numpy as npy
import scipy as math
class DTW:
    Globalcost_=npy.array       #DTW cost matrix
    WarpingPath_=list           #DTW selected Warping Path
    FirstVector_=npy.array      #First time sereis 
    SecondVector_=npy.array     #Second time series 
    Warpindexoffv_=list         #Waring path index of first series  
    Warpindexofsv_=npy.array    #Warping path index of second series
    Warpedfv=npy.array          #Warped attribute values of first series 
    Warpedsv=npy.array          #Warped attribute  values of second series 
    PairwiseDBA_=npy.array
    SDTW_gradient=npy.array
    def __init__(self,vector1,vector2):
        self.Globalcost_=npy.zeros((vector2.shape[0],vector1.shape[0]))
        self.SDTW_gradient=npy.zeros((vector2.shape[0],vector1.shape[0]))
        self.FirstVector_=npy.array(vector1)
        self.SecondVector_=npy.array(vector2)
        self.WarpingPath_=[]
        self.Warpindexoffv_=[]
        self.Warpindexofsv_=[]
        self.PairwiseDBA_=[]
        self.Warpedfv=npy.array([])
        self.Warpedsv=npy.array([])
    def calcglobalcost_DDTW(self):
        derv1=npy.array(self.FirstVector_)
        derv2=npy.array(self.SecondVector_)
        for indx in range(1,derv1.shape[0]-1):
            derv1[indx]=((self.FirstVector_[indx+1]-self.FirstVector_[indx-1])/2+(self.FirstVector_[indx]-self.FirstVector_[indx-1]))/2
        for indx in range(1,derv2.shape[0]-1):
             derv2[indx]=((self.SecondVector_[indx+1]-self.SecondVector_[indx-1])/2+(self.SecondVector_[indx]-self.SecondVector_[indx-1]))/2
        derv1[0]=derv1[1]
        derv1[derv1.shape[0]-1]=derv1[derv1.shape[0]-2]
        derv2[0]=derv2[1]
        derv2[derv2.shape[0]-1]=derv2[derv2.shape[0]-2]
        self.Globalcost_[0,0]=math.square(derv1[0]-derv2[0])
        temp=0;
        for i in range(1,self.FirstVector_.shape[0]):
            self.Globalcost_[0,i]=math.square(derv1[i]-derv2[0])+self.Globalcost_[0,i-1]
        for i in range(1,self.SecondVector_.shape[0]):
            self.Globalcost_[i,0]=math.square(derv2[i]-derv1[0])+self.Globalcost_[i-1,0]
        for i in range(1,self.SecondVector_.shape[0]):
            for j in range(1,self.FirstVector_.shape[0]):
                self.Globalcost_[i,j]=math.square(derv1[j]-derv2[i])
                temp=math.minimum(self.Globalcost_[i-1,j],self.Globalcost_[i,j-1])
                temp=math.minimum(self.Globalcost_[i-1,j-1],temp)
                self.Globalcost_[i,j]+=temp
                
                
#########################################################################################################################
    def Predict_UDTW(self):
        self.WarpingPath_.append((0,0))
        i=0
        j=0
        while((i<self.FirstVector_.shape[0]) or (j<self.SecondVector_.shape[0])):
            if i!=self.FirstVector_.shape[0]-1 and j!=self.SecondVector_.shape[0]-1:
                sum1=math.square(self.FirstVector_[i+1]-self.SecondVector_[j+1])
                sum2=math.square(self.FirstVector_[i+1]-self.SecondVector_[j])+sum1
                sum3=math.square(self.FirstVector_[i]-self.SecondVector_[j+1])+sum1
                if sum2>=sum1:
                    if sum3>=sum1:
                        j+=1
                        i+=1
                        self.WarpingPath_.append((i,j))
                    else:
                        j+=1
                        self.WarpingPath_.append((i,j))
                else:
                    if sum2>=sum3:
                        i+=1
                        self.WarpingPath_.append((i,j))
                    else:
                        j+=1
                        self.WarpingPath_.append((i,j))
            else:
                if i==self.FirstVector_.shape[0]-1 and j!=self.SecondVector_.shape[0]-1:
                    while(j<self.SecondVector_.shape[0]-1):
                        j+=1
                        self.WarpingPath_.append((i,j))
                else:
                    if i!=self.FirstVector_.shape[0]-1 and j==self.SecondVector_.shape[0]-1:
                        while(i<self.FirstVector_.shape[0]-1):
                            i+=1
                            self.WarpingPath_.append((i,j))
                    else:
                        i+=1
                        j+=1
        self.Warpindexoffv_,self.Warpindexofsv_=map(list,zip(*self.WarpingPath_))
        self.Warpindexoffv_=npy.array(self.Warpindexoffv_)
        self.Warpindexofsv_=npy.array(self.Warpindexofsv_)
        for i in self.Warpindexoffv_:
            self.Warpedfv=npy.append(self.Warpedfv,[self.FirstVector_[i]])
        for i in self.Warpindexofsv_:
            self.Warpedsv=npy.append(self.Warpedsv,[self.SecondVector_[i]])     
############################################################################################
#This function calculates DTW alignment with out calculating the global cost matrix(windowed)    
    def Windowed_UDTW(self):
        self.WarpingPath_.append((0,0))
        i=0
        j=0
        while((i<self.FirstVector_.shape[0]) or (j<self.SecondVector_.shape[0])):
            if i!=self.FirstVector_.shape[0]-1 and j!=self.SecondVector_.shape[0]-1:
                if math.square(self.FirstVector_[i]-self.SecondVector_[j+1])>=math.square(self.FirstVector_[i+1]-self.SecondVector_[j+1]):
                    if math.square(self.FirstVector_[i+1]-self.SecondVector_[j])>=math.square(self.FirstVector_[i+1]-self.SecondVector_[j+1]):
                        j+=1
                        i+=1
                        self.WarpingPath_.append((i,j))
                    else:
                        i+=1
                        self.WarpingPath_.append((i,j))
                else:
                    if math.square(self.FirstVector_[i]-self.SecondVector_[j+1])>=math.square(self.FirstVector_[i+1]-self.SecondVector_[j]):
                        i+=1
                        self.WarpingPath_.append((i,j))
                    else:
                        j+=1
                        self.WarpingPath_.append((i,j))
            else:
                if i==self.FirstVector_.shape[0]-1 and j!=self.SecondVector_.shape[0]-1:
                    while(j<self.SecondVector_.shape[0]-1):
                        j+=1
                        self.WarpingPath_.append((i,j))
                else:
                    if i!=self.FirstVector_.shape[0]-1 and j==self.SecondVector_.shape[0]-1:
                        while(i<self.FirstVector_.shape[0]-1):
                            i+=1
                            self.WarpingPath_.append((i,j))
                    else:
                        i+=1
                        j+=1
        self.Warpindexoffv_,self.Warpindexofsv_=map(list,zip(*self.WarpingPath_))
        self.Warpindexoffv_=npy.array(self.Warpindexoffv_)
        self.Warpindexofsv_=npy.array(self.Warpindexofsv_)
        for i in self.Warpindexoffv_:
            self.Warpedfv=npy.append(self.Warpedfv,[self.FirstVector_[i]])
        for i in self.Warpindexofsv_:
            self.Warpedsv=npy.append(self.Warpedsv,[self.SecondVector_[i]])
    def calcglobalcost_UDTW(self):
        self.Globalcost_[0,0]=math.square(self.FirstVector_[0]-self.SecondVector_[0])
        temp=0;
        for i in range(1,self.FirstVector_.shape[0]):
            self.Globalcost_[0,i]=math.square(self.FirstVector_[i]-self.SecondVector_[0])+self.Globalcost_[0,i-1]
        for i in range(1,self.SecondVector_.shape[0]):
            self.Globalcost_[i,0]=math.square(self.SecondVector_[i]-self.FirstVector_[0])+self.Globalcost_[i-1,0]
        for i in range(1,self.SecondVector_.shape[0]):
            for j in range(1,self.FirstVector_.shape[0]):
                self.Globalcost_[i,j]=math.square(self.FirstVector_[j]-self.SecondVector_[i])
                temp=math.minimum(self.Globalcost_[i-1,j],self.Globalcost_[i,j-1])
                temp=math.minimum(self.Globalcost_[i-1,j-1],temp)
                self.Globalcost_[i,j]+=temp
#Weignted DTW: Cost coordinates are weighted to improve classification
    def calcglobalcost_WDTW(self,Wmax,g):
        """ The function calcualtes the weighted DTW version
        Takes max wieght and restriction constant g (the smaller g is the constrained 
        the DTW becomes)"""
        self.Globalcost_[0,0]=math.square(self.FirstVector_[0]-self.SecondVector_[0])
        temp=0
        for i in range(1,self.FirstVector_.shape[0]):
            weight=Wmax/(1+math.exp(-g))
            self.Globalcost_[0,i]=math.square(weight*(self.FirstVector_[i]-self.SecondVector_[0]))+self.Globalcost_[0,i-1]
        for i in range(1,self.SecondVector_.shape[0]):
            weight=Wmax/(1+math.exp(-g))
            self.Globalcost_[i,0]=math.square(weight*(self.SecondVector_[i]-self.FirstVector_[0]))+self.Globalcost_[i-1,0]
        for i in range(1,self.SecondVector_.shape[0]):
            for j in range(1,self.FirstVector_.shape[0]):
                weight=Wmax/(1+math.exp(-g*math.absolute((i-j))))
                self.Globalcost_[i,j]=math.square(weight*(self.FirstVector_[j]-self.SecondVector_[i]))
                temp=math.minimum(self.Globalcost_[i-1,j],self.Globalcost_[i,j-1])
                temp=math.minimum(self.Globalcost_[i-1,j-1],temp)
                self.Globalcost_[i,j]+=temp
                
#calculates the global cost of Soft DTW
    def calcglobalcost_SDTW(self,gamma):
        """ The function calcualtes the soft DTW version
              Takes  smoothing facto gamma, the larger gamma the better the 
              approximation to the min function)"""
        self.Globalcost_[0,0]=math.square(self.FirstVector_[0]-self.SecondVector_[0])
        for i in range(1,self.FirstVector_.shape[0]):
            self.Globalcost_[0,i]=math.square((self.FirstVector_[i]-self.SecondVector_[0]))+self.Globalcost_[0,i-1]
        for i in range(1,self.SecondVector_.shape[0]):
            self.Globalcost_[i,0]=math.square((self.SecondVector_[i]-self.FirstVector_[0]))+self.Globalcost_[i-1,0]
        for i in range(1,self.SecondVector_.shape[0]):
            for j in range(1,self.FirstVector_.shape[0]):
                self.Globalcost_[i,j]=math.square((self.FirstVector_[j]-self.SecondVector_[i]))
                # for the computational stability this is prefered as compared to direct application
                res1=math.minimum(self.Globalcost_[i,j-1],self.Globalcost_[i-1,j])
                res1=math.minimum(res1,self.Globalcost_[i-1,j-1])
                res2=math.exp(-1/gamma*(self.Globalcost_[i,j-1]-res1))+math.exp(-1/gamma*(self.Globalcost_[i-1,j]-res1))+math.exp(-1/gamma*(self.Globalcost_[i-1,j-1]-res1))
                # Direct: res1=-gamma*math.log(sum(math.exp(-1*(self.Globalcost_[i-1,j-1])/gamma)+math.exp(-1*(self.Globalcost_[i-1,j])/gamma)+math.exp(-1*(self.Globalcost_[i,j-1]))))
                self.Globalcost_[i,j]+=((-gamma*math.log(res2))+res1)
    def gradientof_SDTW(self,gamma):
        n,m=self.SDTW_gradient.shape
        n-=1
        m-=1
        k=0
        a=0
        b=0
        c=0
        for i in range(n,-1,-1):
            for j in range(0,m+1):
                if i==n and m-k==m:
                    self.SDTW_gradient[i,m-k]=1
                else:
                    if i+1>n:
                        a=0
                        c=0
                        if m-j+1<=m:
                            self.SDTW_gradient[i,m-j]=self.SDTW_gradient[i,m-j+1]*npy.exp((-self.Globalcost_[i,m-j])/gamma)
                    else:
                        if m-j+1>m:
                            b=0
                            c=0
                            if i+1<=n:
                                self.SDTW_gradient[i,m-j]=self.SDTW_gradient[i+1,m-j]*npy.exp((-self.Globalcost_[i,m-j])/gamma)
                        else:
                            
                            a=self.SDTW_gradient[i+1,m-j]
                            b=self.SDTW_gradient[i,m-j+1]
                            c=self.SDTW_gradient[i+1,m-j+1]
                            temp1=self.Globalcost_[i,m-j+1]-self.Globalcost_[i,m-j]-(self.FirstVector_[i+1]-self.SecondVector_[m-j])**2
                            temp2=self.Globalcost_[i,m-j+1]-self.Globalcost_[i,m-j]-(self.FirstVector_[i]-self.SecondVector_[m-j+1])**2
                            temp3=self.Globalcost_[i+1,m-j+1]-self.Globalcost_[i,m-j]-(self.FirstVector_[i+1]-self.SecondVector_[m-j+1])**2
                            self.SDTW_gradient[i,m-j]=a*npy.exp(temp1/gamma)+b*npy.exp(temp2/gamma)+c*npy.exp(temp3/gamma)
    def findwarppath(self):
        row=self.Globalcost_.shape[0]-1
        col=self.Globalcost_.shape[1]-1
        self.WarpingPath_.append((row,col))
        while ((row !=0 and row>0) or (col!=0 and col>0)):
            if row==0 and col!=0:
                self.WarpingPath_.append((row,col-1))
                col-=1
            elif col==0 and row!=0:
                self.WarpingPath_.append((row-1,col))
                row-=1;
            else:
                if self.Globalcost_[row-1,col]>self.Globalcost_[row-1,col-1]:
                    if self.Globalcost_[row-1,col-1]>self.Globalcost_[row,col-1]:
                        self.WarpingPath_.append((row,col-1))
                        col-=1
                    else:
                        self.WarpingPath_.append((row-1,col-1))
                        row-=1
                        col-=1
                else:
                    if self.Globalcost_[row-1,col]>self.Globalcost_[row,col-1]:
                        self.WarpingPath_.append((row,col-1))
                        col-=1
                    else:
                        self.WarpingPath_.append((row-1,col))
                        row-=1
        self.WarpingPath_.reverse()
        self.Warpindexofsv_,self.Warpindexoffv_=map(list,zip(*self.WarpingPath_))
        self.Warpindexoffv_=npy.array(self.Warpindexoffv_)
        self.Warpindexofsv_=npy.array(self.Warpindexofsv_)
        for i in self.Warpindexoffv_:
            self.Warpedfv=npy.append(self.Warpedfv,[self.FirstVector_[i]])
        for i in self.Warpindexofsv_:
            self.Warpedsv=npy.append(self.Warpedsv,[self.SecondVector_[i]])
    def calc_alignedcost(self):
        sums=0;
        for tupelem in self.WarpingPath_:
            sums+=math.square(self.FirstVector_[tupelem[1]]-self.SecondVector_[tupelem[0]])
        return math.sqrt(sums)
    def calcpair_corre(self):
        coff=self.Warpedfv.dot(self.Warpedsv)
        return [coff, self.warpedsv]
    '''def show_aligned(self):
        get_ipython().run_line_magic('matplotlib', 'qt')
        plt.figure()
        plt.plot(self.Warpedfv,color='red',label='vector1')
        plt.plot(self.Warpedsv,color='blue',label='vecotr2')
        plt.xlabel('Sample Number'),plt.ylabel('Attribute Value')
        plt.title('Alignement plot of streached sequences'),plt.grid(axis='both')
        plt.legend()
        plt.show()
        plt.figure()
        plt.plot(self.FirstVector_,color='red')
        plt.plot(self.SecondVector_,color='blue')
        plt.xlabel('Sample Number'),plt.ylabel('Attribute Value')
        plt.title('Association plot of orgnial series'),plt.grid(axis='both')
        for elem in self.WarpingPath_:
            plt.plot((elem[1],elem[0]),(self.FirstVector_[elem[1]],self.SecondVector_[elem[0]]),color='gray')
        plt.show()'''