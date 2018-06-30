========================
Summary of previous work
========================

This document is a chronolgic summary of previous scientific contributions to the issue of road detection on satellite images.  



`Learning to Detect Roads in High-Resolution Aerial Images <https://link.springer.com/chapter/10.1007/978-3-642-15567-3_16>`_
============================================================================================================================================
:Authors:         V. Mnih and G. E. Hinton
:Journal:         Daniilidis K., Maragos P., Paragios N. (eds) Computer Vision – ECCV 2010. ECCV 2010. Lecture Notes in Computer Science, vol 6316. Springer, Berlin, Heidelberg
:Date:            2010
:Hardware:        consumer GPU
:Data:            High-resolution areal images and vector-formatted road maps

**State of the art**

* No automatized road detection in commercial use

* Approaches in use:

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
        
**Approach proposed in the paper**

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
  
    - Start:  vector map with road centerline location including the area captured by the satellite image S
    - Rasterize the road map to obtain a mask C for S:
    
      .. image:: http://quicklatex.com/cache3/a1/ql_694122bc32cb907a4c590caf59090ca1_l3.png
              
    - Use C to define ground truth map M as
    
      .. image:: http://quicklatex.com/cache3/9a/ql_0ba8545148c005e6af6c3809c4eaaf9a_l3.png
      
      where 
      
      :d(i, j): Euclidean distance between location (i, j)  and the nearest nonzero pixel in C, 
      :sigma:   Smoothing parameter, depending on the scale of the areal images and accounts for the uncertainty in road widths and centerline locations.  
                2*sigma + 1 corresponds to the width of a typical two-lane road
      
      
      
      Interpret M as the probability that the location (i, j) belongs to a road where (i, j) is d(i, j) pixels away from the nearest centerline pixel.
    
**Model**

NN with a single hidden layer.  Both the hidden layer an the output unit have a logistic sigmoid activation.  


**Pre-processing**

Insufficient to use only *local* image intensity information.  It is rather suggested to feed the predictor with as much *context* as possible.

:Aim of pre-processing: Dimensionality reduction of the input data in order to provide a larger context the NN.
:Method of choice:      Principal Component Anaysis to wxw RGB aerial image patches and retain the top w*w principal components.
:Outcome:               Dimensionality reduction of 2/3 while retaining the most important structures.  
:Further investigations: Experiments with alternative color spaces yield no difference in performance. 
                         **NOT tried:** Augmentation with other features as edge or texture features.  
      

Define the map
.. image:: http://quicklatex.com/cache3/5c/ql_a2b1d658fb0ffa7a095ed0699fbc295c_l3.png
which carries out the reduction of dimensionality.  


**Training**



 
