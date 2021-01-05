# Time-Series-Averaging-Using-Multi-Tasking-Autoencoder
The provided scripts are the implmentation for the concepts presented in the paper "Time Series Averaging Using Multi-Tasking Autoencoder"; presented at ICTAI 2020.

# Abstract:
<p align="justify">
The estimation of an optimal time series average has been studied for over three decades. The process is mainly challenging due to temporal distortion. Previous approaches mostly addressed this challenge by using alignment algorithms such as Dynamic Time Warping (DTW). However, the quadratic computational complexity of DTW and its inability to align more than two time series simultaneously complicate the estimation. In this paper, we thus follow a different path and state the averaging problem as a generative problem. To this end, we propose a multi-tasking convolutional autoencoder architecture to extract similar latent features for similarly labeled time series under the influence of temporal distortion. We then take the arithmetic mean of latent features as an estimate of the latent mean. Moreover, we project these estimations and investigate their performance in the time domain.  We evaluate the proposed approach through one nearest centroid classification using 85 data sets obtained from the University of California univariate time series repository. Experimental results show that, in the latent space, the proposed multi-tasking autoencoder achieves competitive accuracies as compared to the state-of-the-art, thus demonstrating that the learned latent space is suitable to compute time series averages. In addition, the time domain projection of latent space means provided superior results as compared to an arithmetic mean.
  
# Demonstration of the Problem:
![Demonstration of the Problem](https://raw.githubusercontent.com/tsegaterefe/Time-Series-Averaging-Using-Multi-Tasking-Autoencoder/master/Images/Discription%20of%20the%20Problem.png)

# Proposed Archtecture: 
![Demonstration of Results](https://github.com/tsegaterefe/Time-Series-Averaging-Using-Multi-Tasking-Autoencoder/blob/master/Images/Proposed%20Archtecture.png)

# Demonstration of Results:
![Demonstration of Results](https://github.com/tsegaterefe/Time-Series-Averaging-Using-Multi-Tasking-Autoencoder/blob/master/Images/Demonstration%20of%20results.png)

# How to Use the Scripts:
<p align="justify">
We have provided two versions of our implemenation. If the scripts are to be run on a server whith graphical displays diabled (if displaying plots is not possible), for instance on Google Colab; scripts within the "Without_Latent_Projection" folder can be used. On the contrary, if displaying plots is possible, scripts wihint the "With_Latent_Projection" folder can be used. Before executing the scripts download the univariate time series data sets from the Unversity of California Univariate Time sereis Repository (UCR), whcih can be found at https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/UCRArchive_2018.zip. After the extraction of the data sets follow the below steps.
</p>

* open the "Conv_configuration.py" 
* Modify "self.File_loc" variable with the location of the UCR data sets. I.e., if location is XXX, then self.File_loc='XXXX/'
* Modify "self.Model_save_path" with a location that is persumed to be suitable to save data generated while training the network. I.e., if prefered location is XXX, then self.Model_save_path='XXX/'
* Within the folders of the scripts there is a "Dataset_List.csv" file; modify "self.List_of_data_sets" variable in the "Conv_configuration.py" script with the location of the "Dataset_List.csv" file. I.e., if the location of "Dataset_List.csv" is XXX, then self.List_of_data_sets='XXX/'
* Use the "Conv_autoencoder_main.py" to procede with network training.
<p align="justify">
We have set the main file to execute network training for five different L2 regularization setups as stated in the paper. If training the network for such iteration is not possible for different reason; then, the training can be executed for a single iteration by changing "self.iterations" variable in "Conv_configuration.py" to one. To furhter modify hyperparameters please refer to the comments in the "Conv_configuration.py" script.
</p>

# Reaserch Fnding:  
<p align="justify">
This research was conducted under the Ethiop France PhD. Program which is financed by:
</p>
<ol>  
<il>The Ethiopian ministery of science and higher Education (MOSHE)</li>
<il>The French Embassy to Ethiopia and African Union.</li>
</ol>  
<p align="justify">  
We would like to acknowledge both parties for their generous contributions. 
</p>
</p>

