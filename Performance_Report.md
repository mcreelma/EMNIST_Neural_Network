# **EMNIST Network Performance Report**


## _Problem Statement:_

The goal of this project is to construct a convolutional neural network capable of classifying handwritten numbers and letters from the EMNIST dataset. This dataset is commonly used as a benchmark dataset for machine learning exercises and competitions, with some network architectures achieving testing accuracies as high as 97.9% (Achintya, 2020). Additionally, this dataset has become one of the standard test pieces for innovations in neural network design   (Baldominos, 2019)

My goal for this project was not so much to achieve a new accuracy for the dataset, but rather to construct a working neural network which would achieve satisfactory performance, which I defined as achieving over 85% accuracy on the testing dataset based on the accuracy ranges of published data (paperswithcode.com, 2017).


## _Dataset and Training:_

The EMNIST dataset is a collection of handwritten characters including letters and digits (NIST, 2019). It is based on the modified National Institute of Standards and Technology Database which includes 60,000 handwritten digits used for training image recognition systems (LeCun et. al, 1998).  This dataset was then extended to 814,255 characters and digits in order to form the EMNIST dataset (NIST, 2019), however, this dataset does not include an equal probability of each character and thus risks overtraining certain characters. I am training my algorithm on the balanced EMNIST datasets, which includes 131,600 characters from 47 balanced classes (Cohen et. al, 2017). Using the balanced dataset means that there are an equal number of elements from each class, and thus the algorithm will not develop a tendency to identify more frequently occurring characters. 

The dataset was split using a 1/6 training/validation split, which involves 5/6th of the dataset being used to train the model and the remaining 1/6th being used to validate the results after each epoch. The training process was conducted with mini batches of the overall data in order to minimize the number of weights calculated during each epoch by using a randomly selected subsample for training. 


## _Network Architecture:_


## Table 2 - Modification Progression of Neural Network Design


<table>
  <tr>
   <td><span style="text-decoration:underline;">Batch Size</span>
   </td>
   <td><span style="text-decoration:underline;">Learning Rate</span>
   </td>
   <td><span style="text-decoration:underline;">Epochs</span>
   </td>
   <td><span style="text-decoration:underline;">Testing Accuracy</span>
   </td>
   <td><span style="text-decoration:underline;">Avg Loss</span>
   </td>
   <td><span style="text-decoration:underline;">Notes</span>
   </td>
  </tr>
  <tr>
   <td>30
   </td>
   <td>0.00001
   </td>
   <td>60
   </td>
   <td>83.0
   </td>
   <td>0.0490
   </td>
   <td>basic structure as initially outlined
   </td>
  </tr>
  <tr>
   <td>40
   </td>
   <td>0.00005
   </td>
   <td>60
   </td>
   <td>85.5
   </td>
   <td>0.0537
   </td>
   <td>just altered hyper parameters for batch size and learning rate
   </td>
  </tr>
  <tr>
   <td>40
   </td>
   <td>0.00005
   </td>
   <td>60
   </td>
   <td>50.7
   </td>
   <td>3.8100
   </td>
   <td>added softmax dim = -1 at the end
   </td>
  </tr>
  <tr>
   <td>40
   </td>
   <td>0.00005
   </td>
   <td>60
   </td>
   <td>85.9
   </td>
   <td>0.0547
   </td>
   <td>softmax back to relu and final layer dropped from 120to 80
   </td>
  </tr>
  <tr>
   <td>40
   </td>
   <td>0.00005
   </td>
   <td>60
   </td>
   <td>87.0
   </td>
   <td>0.0482
   </td>
   <td>added dropout at the end with p = 0.05
   </td>
  </tr>
  <tr>
   <td>40
   </td>
   <td>0.00005
   </td>
   <td>60
   </td>
   <td>59.5
   </td>
   <td>0.2050
   </td>
   <td>rrelu to relu
   </td>
  </tr>
  <tr>
   <td>40
   </td>
   <td>0.00005
   </td>
   <td>60
   </td>
   <td>59.2
   </td>
   <td>0.1715
   </td>
   <td>lose dropout
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>0.00005
   </td>
   <td>40
   </td>
   <td>86.0
   </td>
   <td>0.0433
   </td>
   <td>rrelu back, added a last layer of 80 to 47
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>0.00005
   </td>
   <td>40
   </td>
   <td>86.2
   </td>
   <td>0.0435
   </td>
   <td>used SGD for optimization
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>0.00005
   </td>
   <td>40
   </td>
   <td>86.9
   </td>
   <td>0.0377
   </td>
   <td>dropout between fc 1 and 2 (300 to 160)
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>0.00005
   </td>
   <td>40
   </td>
   <td>87.1
   </td>
   <td>0.0377
   </td>
   <td>dropout inplace = true
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>0.00005
   </td>
   <td>40
   </td>
   <td>87.2
   </td>
   <td>0.0060
   </td>
   <td>dropout p = 0.1, back to ADAM for optimization
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>0.00005
   </td>
   <td>40
   </td>
   <td>86.0
   </td>
   <td>0.0064
   </td>
   <td>added second dropout between 2 and 3
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>0.00005
   </td>
   <td>60
   </td>
   <td>86.7
   </td>
   <td>0.0061
   </td>
   <td>upped to 60 epochs to let it keep training
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>0.00005
   </td>
   <td>60
   </td>
   <td>87.6
   </td>
   <td>0.0056
   </td>
   <td>dropped dropout and kept 60 count
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>0.00005
   </td>
   <td>80
   </td>
   <td>88.2
   </td>
   <td>0.0053
   </td>
   <td>80 reps
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>0.00005
   </td>
   <td>120
   </td>
   <td>88.4
   </td>
   <td>0.0053
   </td>
   <td>120 reps
   </td>
  </tr>
</table>



## Table 2 - Final Network Design and Total Number of Weights


<table>
  <tr>
   <td colspan="5" >Network Architecture
   </td>
  </tr>
  <tr>
   <td>Layer type
   </td>
   <td>In
   </td>
   <td>Out
   </td>
   <td>Size
   </td>
   <td>Number of Weights
   </td>
  </tr>
  <tr>
   <td>Convolution
   </td>
   <td>1
   </td>
   <td>20
   </td>
   <td>28 x 28
   </td>
   <td>1.57E+04
   </td>
  </tr>
  <tr>
   <td>Convolution
   </td>
   <td>20
   </td>
   <td>30
   </td>
   <td>28 x 28
   </td>
   <td>4.70E+05
   </td>
  </tr>
  <tr>
   <td>maxpool
   </td>
   <td>30
   </td>
   <td>30
   </td>
   <td>14 x 14
   </td>
   <td>1.38E+08
   </td>
  </tr>
  <tr>
   <td>Convolution
   </td>
   <td>30
   </td>
   <td>30
   </td>
   <td>14 x 14
   </td>
   <td>1.76E+05
   </td>
  </tr>
  <tr>
   <td>Convolution
   </td>
   <td>30
   </td>
   <td>10
   </td>
   <td>14 x 14
   </td>
   <td>5.88E+04
   </td>
  </tr>
  <tr>
   <td>maxpool
   </td>
   <td>10
   </td>
   <td>10
   </td>
   <td>7 x 7
   </td>
   <td>9.60E+05
   </td>
  </tr>
  <tr>
   <td>flatten
   </td>
   <td>49
   </td>
   <td>49
   </td>
   <td>1 x L
   </td>
   <td>0
   </td>
  </tr>
  <tr>
   <td>Linear
   </td>
   <td>490
   </td>
   <td>300
   </td>
   <td>1 x L
   </td>
   <td>147000
   </td>
  </tr>
  <tr>
   <td>Linear
   </td>
   <td>300
   </td>
   <td>160
   </td>
   <td>1 x L
   </td>
   <td>48000
   </td>
  </tr>
  <tr>
   <td>Linear
   </td>
   <td>160
   </td>
   <td>80
   </td>
   <td>1 x L
   </td>
   <td>12800
   </td>
  </tr>
  <tr>
   <td>Linear
   </td>
   <td>80
   </td>
   <td>47
   </td>
   <td>1 x L
   </td>
   <td>3760
   </td>
  </tr>
  <tr>
   <td colspan="4" >Total Weights
   </td>
   <td>1.40E+08
   </td>
  </tr>
</table>



## _Network Results:_

Overall, my network achieved a performance of 88.4% which is comparable with the accuracies of many of benchmark networks, which range in performance from 50.93% to 95.96% accuracy on the testing data (paperswithcode.com, 2017). While certainly not as accurate, I would call my accuracy of 88.4% or 16619/18800 to be above average. Additionally, the improvements of accuracy by epoch seemed to reach a plateau around 80 or so training sessions, as can be seen in Figure 1. Obviously these results can always be improved and I will continue to refine my design and implementation techniques through further iterations.

 
![Accuracy_By_Epoch](https://user-images.githubusercontent.com/44550282/202878658-18e56150-361e-419a-a3a3-97e0ea31726c.png)

Figure 1. - Training Accuracy vs. Epoch of final network design


# Citations


 A. Agnes Lydia and , F. Sagayaraj Francis, Adagrad - An Optimizer for Stochastic Gradient Descent, Department of Computer Science and Engineering, Pondicherry Engineering College, May 2019.


Baldominos A, Saez Y, Isasi P. A Survey of Handwritten Character Recognition with MNIST and EMNIST. Applied Sciences. 2019; 9(15):3169. https://doi.org/10.3390/app9153169 Add to Citavi project by DOI


Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from [http://arxiv.org/abs/1702.05373](http://arxiv.org/abs/1702.05373)


“The EMNIST Dataset.” _NIST_, 28 Mar. 2019, https://www.nist.gov/itl/products-and-services/emnist-dataset. 


Nielsen, Michael A. Neural Networks and Deep Learning, Determination Press, 1 Jan. 1970, [http://neuralnetworksanddeeplearning.com/](http://neuralnetworksanddeeplearning.com/). 


Tripathi, Achintya. “EMNIST Letter Dataset 97.9%:ACC & VAL_ACC: 91.78%.” Kaggle, Kaggle, 16 Aug. 2020 ,[https://www.kaggle.com/code/achintyatripathi/emnist-letter-dataset-97-9-acc-val-acc-91-78](https://www.kaggle.com/code/achintyatripathi/emnist-letter-dataset-97-9-acc-val-acc-91-78). 


LeCun, Yann, et al. “The Mnist Database.” _MNIST Handwritten Digit Database, Yann LeCun, Corinna Cortes and Chris Burges_, Nov. 1998, http://yann.lecun.com/exdb/mnist/. 


Paperswithcode.com. (2017). Papers with code - EMNIST-letters benchmark (image classification). EMNIST Benchmark Algorithms. Retrieved November 19, 2022, from [https://paperswithcode.com/sota/image-classification-on-emnist-letters](https://paperswithcode.com/sota/image-classification-on-emnist-letters)  


Li, Fei-Fei. “Convolutional Neural Networks (CNNs / ConvNets).” CS231N Convolutional Neural Networks for Visual Recognition, Stanford University, Jan. 2022, https://cs231n.github.io/convolutional-networks/. 


Xu, B., Wang, N., Chen, T., & Li, M. (2015). Empirical Evaluation of Rectified Activations in Convolutional Network. http://arxiv.org/abs/1505.00853


