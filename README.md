# Dual Stream Fusion U-Net Transformers for 3D Medical Image Segmentation

This repository is the ```official open-source``` of [Dual Stream Fusion U-Net Transformers for 3D Medical Image Segmentation](https://ieeexplore.ieee.org/abstract/document/10488278)
by Seungkyun Hong*, Sunghyun Ahn*, Youngwan Jo and Sanghyun Park. ```(*equally contributed)```

## 📣 News
* **[2024/08/20]** network codes are released.
* **[2024/04/11]** Our DS-UNETR has been accepted by IEEE BigComp 2024!

## Abstract
we propose a Dual Stream fusion U-NEt TRansformers (DS-UNETR) comprising a Dual Stream Attention Encoder (DS-AE) and Bidirectional All Scale Fusion (Bi-ASF) module. We designed the DS-AE that extracts both spatial and channel features in parallel streams to better understand the relation between channels. When transferring the extracted features from the DS-AE to the decoder, we used the Bi-ASF module to fuse all scale features. We achieved an average **Dice similarity coefficient (Dice score) improvement of 0.97 % and a 95 % Hausdorff distance (HD95), indicating an improvement of 7.43%** compared to that for a state-of-the-art model on the Synapse dataset. We also demonstrated the efficiency of our model by **reducing the space and time complexity with a decrease of 80.73 % in parameters and 78.86 % in FLoating point OPerationS (FLOPS)**. Our proposed model, DS-UNETR, shows superior performance and efficiency in terms of segmentation accuracy and model complexity (both space and time) compared to existing state-of-the-art models on the 3D medical image segmentation benchmark dataset. The approach of our proposed model can be effectively applied in various medical big data analysis applications.

<img width="600" alt="fig-generator" src="https://github.com/user-attachments/assets/6b9f1fec-2f5f-450c-a8a4-f67add3cd5f0">


## Architecture overview of DS-UNETR
The overview of the DS-UNETR framework. In DS-AE, the outputs of the Swin block and C-MSA block in each stage of each stream are fused by the Fusion block. In Bi-ASF, the Fnf is performed on the features received from DS-AE, followed
 by depth-wise separable convolution.

 <img width="1000" alt="fig-generator" src="https://github.com/user-attachments/assets/ce2e1990-caaa-4dc8-a39c-947a42e01d8d">

 ## Results
 Comparison on the abdominal multi-organ segmentation (Synapse) dataset. Abbreviations are: Spl: spleen, RKid: right kidney, LKid: left kidney, Gal: gallbladder, Liv: liver, Sto: stomach, Aor: aorta, Pan: pancreas. Best results are bolded. Best seconds are underlined.
![image](https://github.com/user-attachments/assets/a36dcd34-1f45-4feb-9865-a0e26babc5ed)


## Citation
If you use our work, please consider citing:  
```Shell
@inproceedings{hong2024dual,
  title={Dual Stream Fusion U-Net Transformers for 3D Medical Image Segmentation},
  author={Hong, Seungkyun and Ahn, Sunghyun and Jo, Youngwan and Park, Sanghyun},
  booktitle={2024 IEEE International Conference on Big Data and Smart Computing (BigComp)},
  pages={301--308},
  year={2024},
  organization={IEEE}
}

```

## Contact
Should you have any question, please create an issue on this repository or contact me at skd@yonsei.ac.kr.
