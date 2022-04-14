# A Reproduction of FDA: Fourier Domain Adaptation for Semantic Segmentation
by _Junhan Wen_ and _Liangchen Sui (l.sui@student.tudelft.nl)_

This is a Pytorch reproduction of [2020 CVPR paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.pdf) based on [open source github code](https://github.com/YanchaoYang/FDA).

# 1. Introduction
Semantic segmentation can be treated as a pixel-level classification task. For each pixel in an image, a category is predicted to which this pixel belongs. With high resolution and small granularity, the semantic information is helpful in many downstream tasks. However, a supervised learning of semantic segmantation requires much more pixel-level annotations, which makes it challenging and time-consuming to acquire a good dataset. Therefore, a solution using domain adaptation is put forward to adress the problem. It suggests to train a network using massive annotated data in a source domain, which is usually a simulator where data can be easily sampled, and few unannotated data in a target domain, which is the exact domain our model should be tested. Domain adaptation aims to reduce the shift between distributions of the source and the target, which enables our model to learn the essence of target domain based mainly on the information of source domain. Before this paper, popular methods tackle this issue by training discriminators to maximize the differene between two domains with adversarial learning, which is computationally heavy becasue of the complexity of learning high-level representations. The contribution of this paper is to introduce low-level statistics as prior variability before training by simply transplanting the amplitude-frequency characteristics of target fomain into source domain. The success of this paper provides great insights to representation learning.

# 2. Paper Overview

## 2.1. Domain Adaptation using Fast Fourier Transformation(FFT)
Domain adaptation via *style transfer* made easy using FFT. FDA needs **no deep networks** for style transfer, and involves **no adversarial training**. Below is the diagram of the proposed Fourier Domain Adaptation method:

1. Apply FFT to source and target images.
2. Replace the low frequency part of the source amplitude with that from the target.
3. Apply inverse FFT to the modified source spectrum.

<img src="https://github.com/YanchaoYang/FDA/blob/master/demo_images/FDA.png" width = "600" alt="" align=center />

## 2.2. Traning Loss for Domain Adaptaion
The loss function is contributed by two parts. The first part is a segmentation loss on the source images adapted to target domain, using cross-entrooy with labels of original source images and predictions on adapted source images. The second part is a self-entropy on unannotated target images with predictions on target images as labels of themselves, which regularizes the model decision boundary to be more determined.

![img](https://github.com/CarpeDiem1012/FDA-project/blob/master/demo_images/1.png)

# 3. Reproduction Results
We , which is trained . The results 

# 4. Ablation Study with [Dataset ACDC](https://acdc.vision.ee.ethz.ch/other/ACDC_the_Adverse_Conditions_Dataset_with_Correspondences_for_semantic_driving_scene_understanding-Sakaridis+Dai+Van_Gool-ICCV_21.pdf)

The original paper builds upon the hypothesis that . Our ablation study further explores the effects of swapping amplitude in frequency domain as a low-level statistic style transfer by evaluating our model without training in ACDC dataset. Below is the evaluation process:

1. Apply FFT to Cityscapes data and ACDC data.
2. Replace the low frequency part of the amplitude of ACDC data with that from the Cityscapes data.
3. Apply inverse FFT to the generate new ACDC images with Cityscapes style.
4. Evaluate directly on new ACDC images with Cityscapes style with model trained on GTA5 and Cityscapes.

<img src="https://github.com/CarpeDiem1012/FDA-project/blob/master/demo_acdc/acdc.png" width = "600" alt="original ACDC image" align=center />
<img src="https://github.com/CarpeDiem1012/FDA-project/blob/master/demo_acdc/acdc_in_tar.png" width = "600" alt="ACDC image with Cityscapes style" align=center />
<img src="https://github.com/CarpeDiem1012/FDA-project/blob/master/demo_acdc/source.png" width = "600" alt="Cityscapes image" align=center />
<img src="https://github.com/CarpeDiem1012/FDA-project/blob/master/demo_acdc/source_in_tar.png" width = "600" alt="GTA5 image with Cityscapes style" align=center />

The above command should output:
===> mIoU19: 50.45
===> mIoU16: 54.23
===> mIoU13: 59.78
   
   
**Reference**

[1] 
[2]
[3]
