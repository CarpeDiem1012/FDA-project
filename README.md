# A Reproduction of FDA: Fourier Domain Adaptation for Semantic Segmentation
by _Junhan Wen (Junhan.Wen@tudelft.nl)_ and _Liangchen Sui (l.sui@student.tudelft.nl)_

This is a Pytorch reproduction of [2020 CVPR paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.pdf) based on [open source github code](https://github.com/YanchaoYang/FDA).

# 1. Introduction
Semantic segmentation can be treated as a pixel-level classification task. For each pixel in an image, a category is predicted to which this pixel belongs. With high resolution and small granularity, the semantic information is helpful in many downstream tasks. However, a supervised learning of semantic segmantation requires much more pixel-level annotations, which makes it challenging and time-consuming to acquire a good dataset. Therefore, a solution using domain adaptation is put forward to adress the problem. It suggests to train a network using massive annotated data in a source domain, which is usually a simulator where data can be easily sampled, and few unannotated data in a target domain, which is the exact domain our model should be tested. Domain adaptation aims to reduce the shift between distributions of the source and the target, which enables our model to learn the essence of target domain based mainly on the information of source domain. Before this paper, popular methods tackle this issue by training discriminators to maximize the difference between two domains with adversarial learning, which is computationally heavy becasue of the complexity of learning high-level representations. The contribution of this paper is to introduce low-level statistics as prior variability before training by simply transplanting the low-frequency part of amplitude-frequency characteristics of target fomain into source domain. The success of this paper draws great attention back to the importance of low-level statistics.

# 2. Paper Overview

## 2.1. Domain Adaptation using Fast Fourier Transformation(FFT)
Domain adaptation via *style transfer* made easy using FFT. FDA needs **no deep networks** for style transfer, and involves **no adversarial training**. Below is the diagram of the proposed Fourier Domain Adaptation method:

1. Apply FFT to source and target images.
2. Replace the low frequency part of the source amplitude with that from the target.
3. Apply inverse FFT to the modified source spectrum.

<div align=center>
<img src="https://github.com/YanchaoYang/FDA/blob/master/demo_images/FDA.png" width = "800" alt="" align=center />
</div>

## 2.2. Traning Loss for Domain Adaptaion
The loss function is contributed by two parts. The first part is a segmentation loss on the source images adapted to target domain, using cross-entropy with labels of original source images and predictions on adapted source images. The second part is a self-entropy on unannotated target images with predictions on target images as labels of themselves, which regularizes the model decision boundary to be more determined.

<div align=center>
<img src="https://github.com/CarpeDiem1012/FDA-project/blob/master/demo_images/1.png" alt="loss">
</div>

# 3. Reproduction Results
The results reproduced is based on the first 3 rows of Table 1 in the paper. It is trained from scratch in GTA5 and Cityscapes datasets under 3 different Beat (0.01, 0.05, 0.09) without self-supervised loops. The main is to find out the effect of the only hyperparameter Beta during FDA. Due to limited computational resources, we reduce number of steps from 150000 to 32000, where the training loss are already asymptoticly converging. More detailed configurations are listed in [training and eval configutation](https://github.com/CarpeDiem1012/FDA-project/blob/master/training_cmd.txt).

<div align=center>
<img src="https://github.com/CarpeDiem1012/FDA-project/blob/master/loss_log/loss_log_01.png" alt="training loss" width="300" /><img src="https://github.com/CarpeDiem1012/FDA-project/blob/master/loss_log/loss_log_05.png" alt="training loss" width="300" /><img src="https://github.com/CarpeDiem1012/FDA-project/blob/master/loss_log/loss_log_09.png" alt="training loss" width="300" />
</div>

<div align=center>
   training loss (Beta = 0.01, 0.05, 0.09)
</div>

***

<img src="https://github.com/CarpeDiem1012/FDA-project/blob/master/demo_images/sheet1.png" alt="reproduction sheet" align=center />

<div align=center>
   reproduction results
</div>

***

<img src="https://github.com/CarpeDiem1012/FDA-project/blob/master/demo_images/sheet2.png" alt="original sheet" align=center />

<div align=center>
   original results
</div>

***

<div align=center>
<img src="https://github.com/CarpeDiem1012/FDA-project/blob/master/demo_images/frankfurt_000000_001016_gtFine_color.png" height="150"/><img src="https://github.com/CarpeDiem1012/FDA-project/blob/master/demo_images/frankfurt_000000_001016_leftImg8bit_color.png" height="150"/><img src="https://github.com/CarpeDiem1012/FDA-project/blob/master/demo_images/frankfurt_000000_001016_leftImg8bit_color%20(1).png" height="150"/>
</div>
<div align=center>
   segmentation (ground truth, reproduction, original)
</div>

***

Our reproduction results shows 25% average drop on mIoU compared to the original results, which is probably caused by lack of training iterations. Despite an imperfect reproduction, a similar observation can be that changing Beta in a reasonable range has little impact on the model performance, which is aligned with the conclusion of original papar.

# 4. Ablation Study with [Dataset ACDC](https://acdc.vision.ee.ethz.ch/other/ACDC_the_Adverse_Conditions_Dataset_with_Correspondences_for_semantic_driving_scene_understanding-Sakaridis+Dai+Van_Gool-ICCV_21.pdf)

The original paper builds upon the hypothesis that, low-frequency amplitude swapping introduces a prior variability from the target domain, towards which our categorial classifier should be robust. Our ablation study further explores the effects of this swapping operation as a low-level statistic style transfer by evaluating our model without training in ACDC dataset. Below is the evaluation process:

1. Apply FFT to Cityscapes data and ACDC data.
2. Replace the low frequency part of the amplitude of ACDC data with that from the Cityscapes data.
3. Apply inverse FFT to the generate new ACDC images with Cityscapes style.
4. Evaluate directly on new ACDC images with Cityscapes style using model trained on GTA5 and Cityscapes.

<img src="https://github.com/CarpeDiem1012/FDA-project/blob/master/demo_acdc/exp.png" alt="hybrid" align=center />

This evaluation process is training-free in order to test the . If
1. our model trained on GTA5 and Cityscapes has gained generalizations in the domain of Cityscapes, 
2. and the Fourier swapping operation as style transfer approach can really reduce the domain shift, drawing adapted ACDC images close to the domain of Cityscapes,

then a generally nice performance should also appear on adapted images from ACDC to Cityscapes by prvious model without training.

***

<img src="https://github.com/CarpeDiem1012/FDA-project/blob/master/demo_acdc/sheet3.png" alt="ablation" align=center />

<div align=center>
   ablation results
</div>

***

<div align=center>
<img src="https://github.com/CarpeDiem1012/FDA-project/blob/master/demo_images/GOPR0351_frame_000165_gt_labelColor.png" height="150"/><img src="https://github.com/CarpeDiem1012/FDA-project/blob/master/demo_images/GOPR0351_frame_000165_rgb_anon_color.png" height="150"/>
</div>
<div align=center>
   segmentation (ground truth, ablation)
</div>

***

However, the performance turns out to be horrible, which indicates our previous hypothesis is apprently not complete. Accordingly, we can have the following speculations:
1. It is still believed that the model trained from GTA5 to Cityscapes possess great generalization ability to the Cityscapes domain. This can be proved by the outstanding mIoU performance of the model tested on Cityscapes, which the model has never touched before.
2. The training loss has revealed that, in addtion to the adapted source domain, a few unannoted samples in target domain also contribute to the network by regularizing through self-entropy, which proves to be necessary for a adaptation to target domain in certain way. This might explains well why the swapping operation alone fails to work during our ablation evaluation.
3. Due to the necessity of regularization on target domain, training a new network would be a proper way for applications on the target domain.

_**Reference**_

[1] Yang, Yanchao, and Stefano Soatto. "Fda: Fourier domain adaptation for semantic segmentation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.

[2] Achille, Alessandro, Matteo Rovere, and Stefano Soatto. "Critical learning periods in deep networks." International Conference on Learning Representations. 2018.
