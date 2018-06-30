========================
Summary of previous work
========================

This document is a chronolgic summary of previous scientific contributions to the issue of road detection on satellite images.  



`Learning to Detect Roads in High-Resolution Aerial Images <https://link.springer.com/chapter/10.1007/978-3-642-15567-3_16>`_
============================================================================================================================================
:Authors: V. Mnih and G. E. Hinton
:Journal: Daniilidis K., Maragos P., Paragios N. (eds) Computer Vision – ECCV 2010. ECCV 2010. Lecture Notes in Computer Science, vol 6316. Springer, Berlin, Heidelberg
:Date:    2010

**Summary**

*State of the art*

* No automatized road detection in commercial use
* Ad-hoc multistage approaches:

  - Establish a priori criteria (contrast, low curvature, constant width, etc.) for the appearance of the raod
  - engineer a system which detects objects captured by these criteria
  - some appraoches yield good performance on sample data but fail on large real-word datasets

* Learning-based approaches:

  - so far failed to scale up to large datasets 
  - presumed reason for failure:  

        1. Very little training data:
           Feasibility of large datasets is limited by ground truth for training and testing is obtained by manually labelling each pixel.
        2. Very small context is used for feature extraction or only few features are extracted from the context.
        3. Prediction made 
*Approach suggested in the paper*

* *learn* road detection from labelled data which is abundant (**universities have libraries of geographic data**)
* 


**Hardware**

consumer GPU