# CAPTURE-GAN
Personal pytorch implementation of the 2024 MICCAI paper: CAPTURE-GAN: Conditional Attribute Preservation through Unveiling Realistic GAN for artifact removal in dual-energy CT imaging
We authors will upload all source code after acceptance.

## Abstract
Dual-energy CT (DECT) is gaining attention as an effective medical imaging modality for detecting bone marrow edema. However, imaging is complicated by the lower contrast offered by DECT compared to MRI and the inherent presence of artifacts in the image formation process, necessitating expertise in DECT. Despite advancements in AI-based solutions for image enhancement, achieving an artifact-free outcome in DECT remains diffcult due to the impracticality of obtaining paired ground-truth and artifact-containing images for supervised learning. Recently, unsupervised techniques demonstrate high performance in image translation tasks. However, these methods face challenges in DECT due to the similarity between artifact and pathological patterns and could have a detrimental impact on image interpretation. In this study, we developed CAPTURE-GAN, which leverages a pre-trained classifier to preserve edema characteristics while removing DECT artifacts. Additionally, we introduced a mask indicating local regions pertaining to artifacts in order to prevent the output of the model from being over-smoothed or losing the bones' structural outline. Our approach fully utilizes automatically generated masks within the overall framework to only selectively modify the necessary local regions more cleanly and precisely than existing networks while preserving intricate bone patterns. Particularly, the performance of the classifier on artifact-removed images has been shown to surpass corresponding images before artifact removal.

## Notice
This repository is currently a **work in progress**. I will continue to improve this repository in the future. If you have any questions, please feel free to open an issue so we can make this repo better.




