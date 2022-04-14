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
The loss function is contributed by two parts.

$$
C(\phi ^ {w} D^ {s\rightarrow t}, D^ {t} )= c_ {ce} ( \phi ^ {w} D^ {s\rightarrow t} )+  \lambda _ {ent} c_ {ent} ( \phi ^ {w} D^ {t} ) 
$$

# 3. Reproduction Results


# 4. Ablation Study with [Dataset ACDC](https://acdc.vision.ee.ethz.ch/other/ACDC_the_Adverse_Conditions_Dataset_with_Correspondences_for_semantic_driving_scene_understanding-Sakaridis+Dai+Van_Gool-ICCV_21.pdf)

The original paper builds upon the hypothesis that . Our ablation study further explores the effects of swapping amplitude in frequency domain as a low-level statistic style transfer by evaluating our model without training in ACDC dataset. Firstly, 

# Usage

1. FDA Demo
   
   > python3 FDA_demo.py
   
   An example of FDA for domain adaptation. (source: GTA5, target: CityScapes, with beta 0.01)
   
   ![Image of Source](https://github.com/YanchaoYang/FDA/blob/master/demo_images/example.png)


2. Sim2Real Adaptation Using FDA (single beta)

   > python3 train.py --snapshot-dir='../checkpoints/FDA' --init-weights='../checkpoints/FDA/init_weight/DeepLab_init.pth' 
                      --LB=0.01 --entW=0.005 --ita=2.0 --switch2entropy=0

   *Important*: use the original images for FDA, then do mean subtraction, normalization, etc. Otherwise, will be numerical artifacts.

   DeepLab initialization can be downloaded through this [link.](https://drive.google.com/file/d/1dk_4JJZBj4OZ1mkfJ-iLLWPIulQqvHQd/view?usp=sharing)

   LB: beta in the paper, controls the size of the low frequency window to be replaced.

   entW: weight on the entropy term.
   
   ita: coefficient for the robust norm on entropy.
   
   switch2entropy: entropy minimization kicks in after this many steps.


3. Evaluation of the Segmentation Networks Adapted with Multi-band Transfer (multiple betas)

   > python3 evaluation_multi.py --model='DeepLab' --save='../results' 
                                 --restore-opt1="../checkpoints/FDA/gta2city_deeplab/gta2city_LB_0_01" 
                                 --restore-opt2="../checkpoints/FDA/gta2city_deeplab/gta2city_LB_0_05" 
                                 --restore-opt3="../checkpoints/FDA/gta2city_deeplab/gta2city_LB_0_09"

   Pretrained models on the GTA5 -> CityScapes task using DeepLab backbone can be downloaded [here.](https://drive.google.com/file/d/1HueawBlg6RFaKNt2wAX__1vmmupKqHmS/view?usp=sharing)
   
   The above command should output:
       ===> mIoU19: 50.45
       ===> mIoU16: 54.23
       ===> mIoU13: 59.78
       

4. Get Pseudo Labels for Self-supervised Training

   > python3 getSudoLabel_multi.py --model='DeepLab' --data-list-target='./dataset/cityscapes_list/train.txt' --set='train' 
                                   --restore-opt1="../checkpoints/FDA/gta2city_deeplab/gta2city_LB_0_01" 
                                   --restore-opt2="../checkpoints/FDA/gta2city_deeplab/gta2city_LB_0_05" 
                                   --restore-opt3="../checkpoints/FDA/gta2city_deeplab/gta2city_LB_0_09"


5. Self-supervised Training with Pseudo Labels

   > python3 SStrain.py --model='DeepLab' --snapshot-dir='../checkpoints/FDA' --init-weights='../checkpoints/FDA/init_weight/DeepLab_init.pth' 
                        --label-folder='cs_pseudo_label' --LB=0.01 --entW=0.005 --ita=2.0

6. Other Models

   VGG initializations can be downloaded through this [link.](https://drive.google.com/file/d/1pgHtwBKUcbAyItnU4hgMb96UfY1PGiCv/view?usp=sharing)
   
   Pretrained models on the Synthia -> CityScapes task using DeepLab backbone [link.](https://drive.google.com/file/d/1FRI_KIWnubyknChhTOAVl6ZsPxzvEXce/view?usp=sharing)
   
   Pretrained models on the GTA5 -> CityScapes task using VGG backbone [link.](https://drive.google.com/file/d/15Az8DFaLw1kTgt82KX9rI6S85n7iesdc/view?usp=sharing)
   
   Pretrained models on the Synthia -> CityScapes task using VGG backbone [link.](https://drive.google.com/file/d/1SC7sxKtic_7ClFmAZDlrBqRaL0pvKYZ8/view?usp=sharing)
   
   
**Reference**

[1] 
[2]
[3]
