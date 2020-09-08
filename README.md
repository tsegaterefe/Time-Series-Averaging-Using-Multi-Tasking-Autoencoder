# Time-Series-Averaging-Using-Multi-Tasking-Autoencoder
The provided scripts are the implmentation for the concepts presented in the paper "Time Series Averaging Using Multi-Tasking Autoencoder"; presented at ICTAI 2020.

# Abstract:

The estimation of an optimal time series average has been studied for over three decades. The process is mainly challenging due to temporal distortion. Previous approaches mostly addressed this challenge by using alignment algorithms such as Dynamic Time Warping (DTW). However, the quadratic computational complexity of DTW and its inability to align more than two time series simultaneously complicate the estimation. In this paper, we thus follow a different path and state the averaging problem as a generative problem. To this end, we propose a multi-tasking convolutional autoencoder architecture to extract similar latent features for similarly labeled time series under the influence of temporal distortion. We then take the arithmetic mean of latent features as an estimate of the latent mean. Moreover, we project these estimations and investigate their performance in the time domain.  We evaluate the proposed approach through one nearest centroid classification using 85 data sets obtained from the University of California univariate time series repository. Experimental results show that, in the latent space, the proposed multi-tasking autoencoder achieves competitive accuracies as compared to the state-of-the-art, thus demonstrating that the learned latent space is suitable to compute time series averages. In addition, the time domain projection of latent space means provided superior results as compared to an arithmetic mean.

# Demonstration of the Problem:
! [Demonstration of the Problem](https://github.com/tsegaterefe/Time-Series-Averaging-Using-Multi-Tasking-Autoencoder/blob/master/Images/Discription%20of%20the%20Problem.png)
# Proposed Archtecture: 
! [Demonstration of Results](https://github.com/tsegaterefe/Time-Series-Averaging-Using-Multi-Tasking-Autoencoder/blob/master/Images/Proposed%20Archtecture.png)
# Demonstration of Results:
! [Demonstration of Results](https://github.com/tsegaterefe/Time-Series-Averaging-Using-Multi-Tasking-Autoencoder/blob/master/Images/Demonstration%20of%20results.png)
