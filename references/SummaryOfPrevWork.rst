========================
Summary of previous work
========================

This document is a chronological summary of previous scientific contributions to the issue of road detection on satellite images. 

Most of the previous work can be divided into two categories:

:**road area extraction**:  Generate pixel-level labeling of roads
:**road centerline extraction**:  Aims at detecting skeletons of roads 

Simultaneous area and centerline extraction:   `G. Chen, Y. Wang, S. Xu, H. Wang, S. Jiang, and C. Pan, “Auto-matic road detection and centerline extraction via cascaded end-to-endconvolutional neural network,” TGRS, vol. 55, no. 6, pp. 3322–3337,2017. <https://ieeexplore.ieee.org/document/7873262/>`_

According to 
`Road Extraction by Deep Residual U-Net <https://ieeexplore.ieee.org/document/8309343/>`_ 
road centerline extraction is easily achieved from road areas by using morphological thinning.  

References for **road area extraction** (only partially summarized below):

1. `X. Huang and L. Zhang, “Road centreline extraction from highresolution imagery based on multiscale structural features and support vector machines,” IJRS, vol. 30, no. 8, pp. 1977–1987, 2009. <https://www.tandfonline.com/doi/abs/10.1080/01431160802546837>`_

2. `V. Mnih and G. Hinton, “Learning to detect roads in high-resolution aerial images,” ECCV, pp. 210–223, 2010. <https://link.springer.com/chapter/10.1007/978-3-642-15567-3_16>`_

3. `C. Unsalan and B. Sirmacek, “Road network detection using probabilistic and graph theoretical methods,” TGRS, vol. 50, no. 11, pp. 4441–4453, 2012. <https://ieeexplore.ieee.org/document/6185661/>`_

4. `G. Cheng, Y. Wang, Y. Gong, F. Zhu, and C. Pan, “Urban road extraction via graph cuts based probability propagation,” in ICIP, 2015, pp. 5072–5076. <https://ieeexplore.ieee.org/document/7026027/>`_

5. `S. Saito, T. Yamashita, and Y. Aoki, “Multiple object extraction from aerial imagery with convolutional neural networks,” J. ELECTRON IMAGING, vol. 2016, no. 10, pp. 1–9, 2016. <https://www.ingentaconnect.com/content/ist/jist/2016/00000060/00000001/art00003>`_

6. `R. Alshehhi and P. R. Marpu, “Hierarchical graph-based segmentation for extracting road networks from high-resolution satellite images,” P&RS, vol. 126, pp. 245–260, 2017. <https://www.sciencedirect.com/science/article/pii/S0924271616302015>`_

References on **road centerline detection** (only partially summarized below):

1. `B. Liu, H. Wu, Y. Wang, and W. Liu, “Main road extraction from ZY-3 grayscale imagery based on directional mathematical morphology and VGI prior knowledge in urban areas,” PLOS ONE, vol. 10, no. 9, p.
e0138071, 2015. <http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0138071>`_

2. `C. Sujatha and D. Selvathi, “Connected component-based technique
for automatic extraction of road centerline in high resolution satellite
images,” J. Image Video Process., vol. 2015, no. 1, p. 8, 2015. <https://link.springer.com/article/10.1186/s13640-015-0062-9>`_

`Learning to Detect Roads in High-Resolution Aerial Images <https://link.springer.com/chapter/10.1007/978-3-642-15567-3_16>`_
============================================================================================================================================
:Authors: V Mnih, G E Hinton
:Journal: Daniilidis K., Maragos P., Paragios N. (eds) Computer Vision – ECCV 2010. ECCV 2010. Lecture Notes in Computer Science, vol 6316. Springer, Berlin, Heidelberg
:Date: 2010
:Hardware: consumer GPU
:Data: `Massachusetts Roads Dataset (Mass. Roads) <http://www.cs.toronto.edu/~vmnih/data/>`_
          
          high-resolution areal images (area ~500 km²)

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
:Measure:



**Error analysis**

Most of the errors are due to the labeling process:

* Lack of information about the width.
* Small roads are not labeled in the vector-formatted road map.

Suggestion for performance improvement:

View the labels as noisy versions of underlying true labels. This allows the NN to override labels that are incorrect (commonly used in the field of object recognition)


**Update to the paper:**

a) `V. Mnih and G. Hinton, ‘‘Learning to label aerial images from noisy data,’’ Proc. 29th Annual Int’l Conf. on Machine Learning (ICML 2012) <https://www.cs.toronto.edu/~vmnih/docs/noisy_maps.pdf>`_


  Incorporate two different noise models occuring in label images:

   1. omission noise:
     occurs when an object appears in an aerial imagery but not in the corresponding label image
   
   2. registration noise
     inaccurate location of the object in a label image
   
  Proposal:

  Asymmetric Bernoulli distribution and transnational noise distribution


 
b) Mnih `PhD thesis <https://www.cs.toronto.edu/~vmnih/docs/Mnih_Volodymyr_PhD_Thesis.pdf>`_ (2013)


`DeepSat – A Learning framework for Satellite Imagery <http://bit.csc.lsu.edu/~saikat/publications/sigproc-sp.pdf>`_
====================================================================================================================

:Authors: S Basu *et al.*
:Journal: Daniilidis K., Maragos P., Paragios N. (eds) Computer Vision – ECCV 2010. ECCV 2010. Lecture Notes in Computer Science, vol 6316. Springer, Berlin, Heidelberg
:Date: 09.2015
:Hardware: consumer GPU
:Data: SAT-4 & SAT-6 (new satellite datasets, four bands (red, green, blue, NIR),  U.S.)
:GitHub: `link <https://github.com/mpapadomanolaki/Training-on-DeepSat>`_
:Measure:

**Approach - NO road detection**

Classification framework consisting of

1. Pre-processing (input data -> normalized data):
 
  * feature extraction
  * normalization 
  
 2. Classification: **Deep Belief Network** (input:  normalized data from 1., unsupervised appraoch)
    Accuracy:  97.95 % (SAT-4) / 93.9% (SAT-6)
    
    
    
 



`Road Extraction from Very High Resolution Remote Sensing Optical Images based onTexture Analysis and Beamlet Transform <https://ieeexplore.ieee.org/document/7159022/>`_
====================================================================================================================

:Authors: M O Sghaier, R Lepage
:Journal: IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing ( Volume: 9, Issue: 5, May 2016 ) 
:Date: 07.2015
:Data:   ``_
:GitHub: ``_
:Measure:

**Approach (simultaneous building and road detection)**


**Method**



**Training**



**Result**



    
    
    
 



`Multiple Object Extraction from Aerial Imagery with Convolutional Neural Networks <https://www.ingentaconnect.com/content/ist/jist/2016/00000060/00000001/art00003>`_
====================================================================================================================

:Authors: S Shunta, Y Takayoshi, A Yoshimitsu
:Journal: Society for Imaging Science and Technology
:Date: 01.2016
:Data:   `Massachusetts Buildings Dataset (Mass. Buildings) and Massachusetts Roads Dataset (Mass. Roads) <http://www.cs.toronto.edu/~vmnih/data/>`_
:GitHub: `link <https://github.com/mitmul/ssai>`_
:Measure: 90.47% (recall at breakeven, road detection)

**Approach (simultaneous building and road detection)**

CNNs trained on publicly available aerial imagery dataset according to `Mnih PhD Thesis <https://www.cs.toronto.edu/~vmnih/docs/Mnih_Volodymyr_PhD_Thesis.pdf>`_.

No need to

* design image features manually
* individual training of multiple classifiers for each terrestrial object 
* consider how to fuse multiple decisions 

**Method**

similar to `Learning to Detect Roads in High-Resolution Aerial Images <https://link.springer.com/chapter/10.1007/978-3-642-15567-3_16>`_ but with multi-class output


**Training**

Mini-batch stochastic gradient decent with momentum

Learning rate is reduced during learning by multiplication with a fixed reducing rate every x iterations.

Regularization with L2 weight decay.  

Hyperparamters: 

1. Mini-batch size
2. Learning rate
3. Learning-rate reducing rate
4. Weight of the L2 term


**Result**

Increase of the road detection accuracy.

    
    
    
 



`Satellite Imagery Classification Based on Deep Convolutional Network <https://waset.org/publications/10004722/satellite-imagery-classification-based-on-deep-convolution-network>`_
====================================================================================================================

:Authors: Z Ma, Z Wang, C Liu, X Liu
:Hardware: 
:Data: 
:GitHub: 
:Measure:

**Approach (simultaneous building and road detection)**


**Method**



**Training**



**Result**

    
    
    
 



`Fully Convolutional Networks for Dense Semantic Labelling of High-Resolution Aerial Imagery <https://ieeexplore.ieee.org/document/7159022/>`_
====================================================================================================================

:Authors: J Sherrah
:Journal: arXiv
:Date: 06.2016
:Hardware: 
:Data: 
:GitHub: 
:Measure:

    
    
**Approach**
Fully convolutional network for semantic labeling (no patch-based approach)

Usually FCN have low resolution output (lower than input) due to *down-sampling*.  

The presented novel approach maintain the full resolution. 


**Method**



**Training**



**Result**

    
 



`MRF-based Segmentation and Unsupervised Classification for Building and Road Detection in Peri-urban Areas of High-resolution Satellite Images <https://www.sciencedirect.com/science/article/pii/S0924271616304816>`_
====================================================================================================================

:Authors: I Grinias, C Panagiotakis, G Tziritas
:Journal: ISPRS Journal of Photogrammetry and Remote Sensing
:Date: 12.2016
:Hardware: 
:Data: 
:GitHub: 
:Measure: 

   
**Approach (simultaneous building and road detection)**


**Method**



**Training**



**Result**
 
    
    
 



`Creating Roadmaps in Aerial Images with Generative Adversarial Networks and Smoothing-based Optimization <http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w30/Costea_Creating_Roadmaps_in_ICCV_2017_paper.pdf>`_
====================================================================================================================

:Authors: D Costea, A Marcu, E Slusanschi, M Leordeanu
:Journal: IEEE Xplore
:Date: 10.2017
:Hardware: GPU (Tesla K40)
:Data: `European Road Dataset <https://pdfs.semanticscholar.org/191b/eb87f84326d2cc9c427efe2a5abee8f67574.pdf>`_
:Measure: 84.05 % (F-measure road detection)



 **Task** 
 
 1. Translate RGB images into road maps
 2. Translate the predictions into intersection locations
 
**Approach (road detection and road map graph)**

1. Novel dual-hop generative adversarial network (DH-GAN):  segments images at the level of pixels

2. Smoothing based optimization (SBO): Transform pixelwise segmentation into a road map graph

**Method**

Two conditional GANs, each consisting of one 

a) segmentation Generator G (each an adapted version of `U-nets <https://arxiv.org/abs/1505.04597>`_)
b) discriminator D (variant `PatchGAN <https://arxiv.org/abs/1611.07004>`_)

The 

:1. cGAN: predicts pixelwise roat maps
          learns a pixelwise segmentation generator G
          D detects G's misleading road outputs
          
:2. cGAN: outputs intersection locations (has access to original RGB and 1. cGAN's output)
          learns intersection generator G
          D detects G's misleading intersection outputs

Generator Architecture:

* fully convolutional encoder-decoder network
* 9 down-sampling modules
* output: 512x512 pixels, 64 filters
* bottleneck layer: 1x1x512 (-> loss of high-frequency information, solving by applying `skip connections <https://arxiv.org/abs/1611.07004>`_)
* decoder mirrors encoder, but fractionally-strided convolutions

Discriminator Architecture:

* fully convolutional network 
* 5 downsampling modules
* **increase of number of parameters results in very small performance**

**Training**

* Optimize negative log-likelihood 
* mini-batch stochastic gradient decent
* Adam solver, LR = 2e-4, momentum = 0.5
* 200 epochs (60th was the best, afterwards overfitting)

**Remarks**

* Store vertices uses 70 times less space 



`Road Extraction by Deep Residual U-Net <https://ieeexplore.ieee.org/document/8309343/>`_
====================================================================================================================

:Authors: Z Zhang, Q Liu, Y Wang
:Journal: IEEE Geoscience and Remote Sensing Letters
:Date: 03.2018
:Hardware: GPU (NVIDIA Titan 1080)
:Data: `Massachusetts Roads Dataset (Mass. Roads) <http://www.cs.toronto.edu/~vmnih/data/>`_
:GitHub: `link1 <https://github.com/DuFanXin/deep_residual_unet>`_, `link2 <https://github.com/handong1587/handong1587.github.io/blob/master/_posts/deep_learning/2015-10-09-segmentation.md>`_
:Measure: 91.87% (breakeven point)



**Approach (simultaneous building and road detection)**

Deep residual U-Net:  combines advantages of residual learning and U-Net architecture.  

Difference to U-Net:

1. Residual units instead of plain neural units (basic block)
2. No cropping operation required.




**Method**

The U-Net (basically encoder - bridge - decoder structure):

* to get a finer result, it is important to use low level details while retaining high-level semantic information
* training is very hard also due to limitations on training samples 
          
          -> solve this by employing per-trained networks, then fine-tune them on target data
          
          OR: employing extensive data augmentation (as done here)
          
 The `Residual unit <https://arxiv.org/abs/1512.03385>`_:
 
 * consists of a series of stacked residual units


**Training**

* no data augmentation during training 
* 224x244 image size
* 30000 samples
* convergence after 50 epochs
* Mini-batches (size 8)
* LR 10e3, reduced by factor of 0.1 every 20 epochs

**Remarks**
Achieves promising results although parameters of the residual U-Net are only 1/4 of the one required for an U-Net
 7.8M  versus  30.6M).
