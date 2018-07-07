========================
Summary of previous work
========================

This document is a chronolgic summary of previous scientific contributions to the issue of road detection on satellite images. 



`Learning to Detect Roads in High-Resolution Aerial Images <https://link.springer.com/chapter/10.1007/978-3-642-15567-3_16>`_
============================================================================================================================================
:Authors: V Mnih, G E Hinton
:Journal: Daniilidis K., Maragos P., Paragios N. (eds) Computer Vision – ECCV 2010. ECCV 2010. Lecture Notes in Computer Science, vol 6316. Springer, Berlin, Heidelberg
:Date: 2010
:Hardware: consumer GPU
:Data: high-resolution areal images (area ~500 km²)

          vector-formatted road maps

**State of the art**

There were no automatized road detection in commercial use at the time where the paper was published.

Approaches in use:

1. Ad-hoc multistage approaches:

- Establish a priori criteria (contrast, low curvature, constant width, etc.) for the appearance of the road
- Engineer a system which detects objects captured by these criteria
- Some approaches yield good performance on sample data but fail on large real-word datasets

2. Learning-based approaches:

- Failed to scale up to large datasets so far
- Presumed reason for failure: 

a. Very little training data:
Feasibility of large datasets is limited by ground truth for training and testing is obtained by manually labeling each pixel.
b. Very small context is used for feature extraction or only few features are extracted from the context.
c. Independent predictions drawn from each pixel

**Approach (patch-based semantic segmentation task)**

* *learn* road detection from labeled data which is abundant (**according to the authors: universities have libraries of geographic data (? -> contradiction to 2a. ?)**)
* Address all three issues of learning-based approaches at once by means of the following technique:

1. Use vector road maps to generate synthetic road/non-road labels.
2. Implement NN on GPU to efficiently 
a. increase number of extracted features and 
b. enlarge the context.
3. Post-processing procedure: 
Use dependencies present in the vicinity of each map pixel to improve predictions of the NN

* Approach was the first one which works well on large datasets:

- One order of magnitude of increase of the considered area
- Evaluation was performed on urban datasets


**Data**

* High-resolution aerial images
* Road maps in vector format (hand-labeled data is more accurate but harder to obtain):

- Transform vector-formatted road maps into approximate labels per pixel.
- Proposed procedure for label generation per pixel:
- Start: vector map with road centerline location including the area captured by the satellite image S
- Rasterize the road map to obtain a mask C for S:
.. image:: http://quicklatex.com/cache3/a1/ql_694122bc32cb907a4c590caf59090ca1_l3.png
- Use C to define ground truth map M as
.. image:: http://quicklatex.com/cache3/9a/ql_0ba8545148c005e6af6c3809c4eaaf9a_l3.png
where 
:d(i, j): Euclidean distance between location (i, j) and the nearest nonzero pixel in C, 
:sigma: Smoothing parameter, depending on the scale of the areal images and accounts for the uncertainty in road widths and centerline locations. 
2*sigma + 1 corresponds to the width of a typical two-lane road
Interpret M as the probability that the location (i, j) belongs to a road where (i, j) is d(i, j) pixels away from the nearest centerline pixel.

**Model**

NN with a single hidden layer. Both the hidden layer an the output unit have a logistic sigmoid activation. 


**Pre-processing**

It appears to be insufficient to use only *local* image intensity information. It is rather suggested to feed the predictor with as much *context* as possible.

:Aim of pre-processing: Dimensionality reduction of the input data in order to provide a larger context the NN.
:Method of choice: Principal Component Anaysis to wxw RGB aerial image patches and retain the top w*w principal components.
:Outcome: Dimensionality reduction of 2/3 while retaining the most important structures. 
:Further investigations: Experiments with alternative color spaces yield no difference in performance. 
**NOT tried:** Augmentation with other features as edge or texture features. 

Define the map
 .. image:: http://quicklatex.com/cache3/5c/ql_a2b1d658fb0ffa7a095ed0699fbc295c_l3.png
which carries out the reduction of dimensionality. 


**Training**

:Pre-training:    Unsupervised pre-trainig (= pre-initializing the weights of the NN) with the procedure of Hinton and Salakhutdinov (Gaussian-binary Restricted Boltzmann Machines) to increase performance. 
                  According to the authors Gaussian-binary RBM is not a good choice for images as pixels are assumed to be independent of the features.
                  Better choice: include explicit representation of covariance structure (?)
                  
:Model fit:       Minimization of the cross entropy. 
                  Data augmentation by rotation of the images (bias removal). 
:Post-processing: According to Jain and Seung (natural image denoising) to increase performance.
:Metrics:         Completeness (fraction of true roads that were correctly detected)

                  Correctness (fraction of predicted roads that are true roads)



**Error analysis**

Most of the errors are due to the labeling process:

* Lack of information about the width.
* Small roads are not labeled in the vector-formatted road map.

Suggestion for performance improvement:

View the labels as noisy versions of underlying true labels. This allows the NN to override labels that are incorrect (commonly used in the field of object recognition)


**Udate to the paper:**

a) `V. Mnih and G. Hinton, ‘‘Learning to label aerial images from noisy data,’’ Proc. 29th Annual Int’l Conf. on Machine Learning (ICML 2012) <https://www.cs.toronto.edu/~vmnih/docs/noisy_maps.pdf>`_


  Incorporate two different noise models occuring in label images:

   1. omission noise:
     occurs when an object appears in an aerial imagery but not in the corresponding label image
   
   2. registration noise
     inaccurate location of the object in a label image
   
  Proposal:

  Asymmetric Bernoulli distribution and translational noise distribution


 
b) Mnih `PhD thesis <https://www.cs.toronto.edu/~vmnih/docs/Mnih_Volodymyr_PhD_Thesis.pdf>`_ (2013)


`DeepSat – A Learning framework for Satellite Imagery <http://bit.csc.lsu.edu/~saikat/publications/sigproc-sp.pdf>`_
====================================================================================================================

:Authors: S Basu *et al.*
:Journal: Daniilidis K., Maragos P., Paragios N. (eds) Computer Vision – ECCV 2010. ECCV 2010. Lecture Notes in Computer Science, vol 6316. Springer, Berlin, Heidelberg
:Date: 09.2015
:Hardware: consumer GPU
:Data: SAT-4 & SAT-6 (new satellite datasets, four bands (red, green, blue, NIR),  U.S.)
:GitHub: `link <https://github.com/mpapadomanolaki/Training-on-DeepSat>`_

**Approach - NO road detection**

Classification framework consisting of

1. Preprocessing (input data -> normalized data):
 
  * feature extraction
  * normalization 
  
 2. Classification: **Deep Belief Network** (input:  normalized data from 1., unsupervised appraoch)
    Accuracy:  97.95 % (SAT-4) / 93.9% (SAT-6)
    
    
    
 



`Road Extraction from Very High ResolutionRemote Sensing Optical Images based onTexture Analysis and Beamlet Transform <https://ieeexplore.ieee.org/document/7159022/>`_
====================================================================================================================

:Authors: M O Sghaier, R Lepage
:Journal: IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing ( Volume: 9, Issue: 5, May 2016 ) 
:Date: 07.2015
:Data:   `Massachusetts Buildings Dataset (Mass. Buildings) and Massachusetts Roads Dataset (Mass. Roads) <http://www.cs.toronto.edu/~vmnih/data/>`_
:GitHub: `link <https://github.com/mitmul/ssai>`_

**Approach (simultanous building and road detection)**

CNNs trained on publicly available aerial imagery dataset accroding to `Mnih <https://www.cs.toronto.edu/~vmnih/docs/Mnih_Volodymyr_PhD_Thesis.pdf>`_.

No need to

* design image freatures manually
* individual training of multiple classifiers for each terrestrial object 
* consider how to fuse multile decisions 

**Method**
similar to `Learning to Detect Roads in High-Resolution Aerial Images <https://link.springer.com/chapter/10.1007/978-3-642-15567-3_16>`_ but with multi-class output


**Training**

Mini-batch stochastic gradient decent with momentum

Learning rate is reduced during learning by multipliction with a fixed reducing rate every x iterations.

Regularization with L2 weight decay.  

Hyperparamters: 

1. Mini-batch size
2. Learning rate
3. Learning-rate reducing rate
4. Weight of the L2 term


**Result**

Increase of the road detection accuracy. 



    
    
    
 



`Multiple Object Extraction from Aerial Imagery withConvolutional Neural Networks <https://www.ingentaconnect.com/content/ist/jist/2016/00000060/00000001/art00003>`_
====================================================================================================================

:Authors: S Shunta, Y Takayoshi, A Yoshimitsu
:Journal: Society for Imaging Science and Technology
:Date: 01.2016
:Hardware: 
:Data: 
:GitHub: 

    
    
    
 



`Satellite Imagery Classification Based on Deep Convolutional Network <https://waset.org/publications/10004722/satellite-imagery-classification-based-on-deep-convolution-network>`_
====================================================================================================================

:Authors: Z Ma, Z Wang, C Liu, X Liu
:Hardware: 
:Data: 
:GitHub: 

    
    
    
 



`Fully Convolutional Networks for Dense Semantic Labelling of High-Resolution Aerial Imagery <https://ieeexplore.ieee.org/document/7159022/>`_
====================================================================================================================

:Authors: J Sherrah
:Journal: arXiv
:Date: 06.2016
:Hardware: 
:Data: 
:GitHub: 

    
    
    
 



`MRF-based Segmentation and Unsupervised Classification forBuilding and Road Detection in Peri-urban Areas ofHigh-resolution Satellite Images <https://www.sciencedirect.com/science/article/pii/S0924271616304816>`_
====================================================================================================================

:Authors: I Grinias, C Panagiotakis, G Tziritas
:Journal: ISPRS Journal of Photogrammetry and Remote Sensing
:Date: 12.2016
:Hardware: 
:Data: 
:GitHub: 

    
    
    
 



`Creating Roadmaps in Aerial Images with Generative Adversarial Networks and Smoothing-based Optimization <http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w30/Costea_Creating_Roadmaps_in_ICCV_2017_paper.pdf>`_
====================================================================================================================

:Authors: D Costea, A Marcu, E Slusanschi, M Leordeanu
:Journal: IEEE Xplore
:Date: 10.2017
:Hardware: 
:Data: 
:GitHub: 

    
    
    
 



`Road Extraction by Deep Residual U-Net <https://ieeexplore.ieee.org/document/8309343/>`_
====================================================================================================================

:Authors: Z Zhang, Q Liu, Y Wang
:Journal: IEEE Geoscience and Remote Sensing Letters
:Date: 03.2018
:Hardware: 
:Data: 
:GitHub: `link1 <https://github.com/DuFanXin/deep_residual_unet>`_, `link2 <https://github.com/handong1587/handong1587.github.io/blob/master/_posts/deep_learning/2015-10-09-segmentation.md>`_
