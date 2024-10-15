# Awesome Video Anomaly Detection
![GitHub License](https://img.shields.io/github/license/Junxi-Chen/Awesome-Video-Anomaly-Detection)
![Awesome](https://awesome.re/badge.svg)

Video anomaly detection (VAD) aims to identify anomalous frames within given videos, which servers a vital function in critical areas, e.g., public security, media content monitoring and industrial manufacture. This repository collects latest research papers, code, datasets, utilities and related resources for VAD.

If you find this repository helpful, feel free to star or share it 😆! If you spot any errors, notice omissions or have any suggestions, please reach out via GitHub [issues](https://github.com/Junxi-Chen/Awesome-Video-Anomaly-Detection/issues), [pull requests](https://github.com/Junxi-Chen/Awesome-Video-Anomaly-Detection/pulls) or [email]((mailto:chenjunxi22@mails.ucas.ac.cn)).



## Contents
- [Recent Updates](#recent-updates)
- [New Setting Papers](#new-setting-papers)
- [Weakly-supervised VAD Papers](#weakly-supervised-vad-papers)
  - [Prompt Involved Papers](#prompt-involved-papers)
- [Semi-supervised VAD Papers](#semi-supervised-vad-papers)
- [Fully-supervised VAD Papers](#fully-supervised-vad-papers)
- [Surveys](#surveys)
- [Datasets](#datasets)
- [Utilities](#utilities)
- [Related Repositories](#related-repositories)

## Recent Updates
Last Update: October 15, 2024
- ACM MM 24'
- ECCV 24'
- CVPR 24'

## New Setting Papers
**Follow the Rules: Reasoning for Video Anomaly Detection with Large Language Models** \
*Yuchen Yang, Kwonjoon Lee, Behzad Dariush, Yinzhi Cao, Shao-Yuan Lo* \
![LLM](https://img.shields.io/badge/LLM-FFA500)\
ECCV 24' [[paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10568.pdf)][[code](https://github.com/Yuchen413/AnomalyRuler)]

**Open-Vocabulary Video Anomaly Detection** \
*Peng Wu, Xuerong Zhou, Guansong Pang, Yujia Sun, Jing Liu, Peng Wang, Yanning Zhang* \
![CLIP-V](https://img.shields.io/badge/CLIP--V-6d4aff)
![CLIP-T](https://img.shields.io/badge/CLIP--T-C3B9FA)
![LLM](https://img.shields.io/badge/LLM-FFA500)\
CVPR 24' [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Open-Vocabulary_Video_Anomaly_Detection_CVPR_2024_paper.pdf)][[supp](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Wu_Open-Vocabulary_Video_Anomaly_CVPR_2024_supplemental.pdf)]

**Harnessing Large Language Models for Training-free Video Anomaly Detection** \
*Luca Zanella, Willi Menapace, Massimiliano Mancini, Yiming Wang, Elisa Ricci* \
![LLM](https://img.shields.io/badge/LLM-FFA500)\
CVPR 24' [[paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10568.pdf)][[code](https://github.com/lucazanella/lavad)][[supp](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Zanella_Harnessing_Large_Language_CVPR_2024_supplemental.pdf)]

**Uncovering What, Why and How:  A Comprehensive Benchmark for Causation Understanding of Video Anomaly** \
*Hang Du, Sicheng Zhang, Binzhu Xie, Guoshun Nan, Jiayang Zhang, Junrui Xu, Hangyu Liu, Sicong Leng, Jiangming Liu, Hehe Fan, Dajiu Huang, Jing Feng, Linli Chen, Can Zhang, Xuhuan Li, Hao Zhang, Jianhang Chen, Qimei Cui, Xiaofeng Tao* \
![LLM](https://img.shields.io/badge/LLM-FFA500)\
CVPR 24' [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Du_Uncovering_What_Why_and_How_A_Comprehensive_Benchmark_for_Causation_CVPR_2024_paper.pdf)][[code & dataset](https://github.com/fesvhtr/CUVA)][[supp](https://openaccess.thecvf.com/content/CVPR2024/html/Du_Uncovering_What_Why_and_How_A_Comprehensive_Benchmark_for_Causation_CVPR_2024_paper.html)]

**TDSD: Text-Driven Scene-Decoupled Weakly Supervised Video Anomaly Detection** \
*Shengyang Sun, Jiashen Hua, Junyi Feng, Dongxu Wei, Baisheng Lai, Xiaojin Gong* \
![I3D](https://img.shields.io/badge/I3D-35BF5C)
![CLIP-V](https://img.shields.io/badge/CLIP--V-6d4aff)
![CLIP-T](https://img.shields.io/badge/CLIP--T-C3B9FA)\
ACM MM 24' [[paper](https://openreview.net/pdf?id=TAVtkpjS9P)][[code](https://github.com/shengyangsun/TDSD)][[OpenReview](https://openreview.net/forum?id=TAVtkpjS9P&noteId=TAVtkpjS9P)]

## Weakly-supervised VAD Papers
**Cross-Domain Learning for Video Anomaly Detection with Limited Supervision** \
*Yashika Jain, Ali Dabouei, Min Xu* \
![I3D](https://img.shields.io/badge/I3D-35BF5C)
![CLIP-V](https://img.shields.io/badge/CLIP--V-6d4aff)\
ECCV 24' [[paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04459.pdf)]

**Cross-Modal Fusion and Attention Mechanism for Weakly Supervised Video Anomaly Detection** \
*Ayush Ghadiya, Purbayan Kar, Vishal Chudasama, Pankaj Wasnik* \
![I3D](https://img.shields.io/badge/I3D-35BF5C)
![with-Audio](https://img.shields.io/badge/with--Audio-00B2FF)\
CVPR 24' Workshop [[paper](https://openaccess.thecvf.com/content/CVPR2024W/MULA/papers/Ghadiya_Cross-Modal_Fusion_and_Attention_Mechanism_for_Weakly_Supervised_Video_Anomaly_CVPRW_2024_paper.pdf)] 

**Weakly Supervised Video Anomaly Detection and Localization with Spatio-Temporal Prompts** \
*Peng Wu, Xuerong Zhou, Guansong Pang, Zhiwei Yang, Qingsen Yan, Peng Wang, Yanning Zhang* \
![CLIP-V](https://img.shields.io/badge/CLIP--V-6d4aff)
![CLIP-T](https://img.shields.io/badge/CLIP--T-C3B9FA)\
ACM MM 24' [[paper](https://arxiv.org/pdf/2408.05905)][[OpenReview](https://openreview.net/forum?id=2es1ojI14x&referrer=%5Bthe%20profile%20of%20Peng%20Wu%5D(%2Fprofile%3Fid%3D~Peng_Wu2))]

**Exploiting Completeness and Uncertainty of Pseudo Labels  for Weakly Supervised Video Anomaly Detection** \
*Chen Zhang, Guorong Li, Yuankai Qi, Shuhui Wang, Laiyun Qing, Qingming Huang, Ming-Hsuan Yang* \
![I3D](https://img.shields.io/badge/I3D-35BF5C)\
CVPR 23' [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Exploiting_Completeness_and_Uncertainty_of_Pseudo_Labels_for_Weakly_Supervised_CVPR_2023_paper.pdf)][[code](https://github.com/ArielZc/CU-Net)][[supp](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Zhang_Exploiting_Completeness_and_CVPR_2023_supplemental.pdf)]

**Look Around for Anomalies: Weakly-supervised Anomaly Detection via Context-Motion Relational Learning** \
*MyeongAh Cho, Minjung Kim, Sangwon Hwang, Chaewon Park, Kyungjae Lee, Sangyoun Lee* \
![I3D](https://img.shields.io/badge/I3D-35BF5C)\
CVPR 23' [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Cho_Look_Around_for_Anomalies_Weakly-Supervised_Anomaly_Detection_via_Context-Motion_Relational_CVPR_2023_paper.pdf)][[supp](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Cho_Look_Around_for_CVPR_2023_supplemental.pdf)]


### Prompt Involved Papers
**Vadclip: Adapting vision-language models for weakly supervised video anomaly detection** \
*Peng Wu, Xuerong Zhou, Guansong Pang, Lingru Zhou, Qingsen Yan, Peng Wang, Yanning Zhang* \
![CLIP-V](https://img.shields.io/badge/CLIP--V-6d4aff)
![CLIP-T](https://img.shields.io/badge/CLIP--T-C3B9FA)\
AAAI 24' [[paper](https://ojs.aaai.org/index.php/AAAI/article/download/28423/28826)][[code](https://github.com/nwpu-zxr/VadCLIP)] 

**Prompt-Enhanced Multiple Instance Learning for Weakly Supervised Video Anomaly Detection** 
 \
*Junxi Chen, Liang Li , Li Su , Zheng-Jun Zha, Qingming Huang* \
![I3D](https://img.shields.io/badge/I3D-35BF5C)
![CLIP-T](https://img.shields.io/badge/CLIP--T-C3B9FA)
![with-Audio](https://img.shields.io/badge/with--Audio-00B2FF)\
CVPR 24' [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Prompt-Enhanced_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_Detection_CVPR_2024_paper.pdf)][[code](https://github.com/Junxi-Chen/PE-MIL)][[supp](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Chen_Prompt-Enhanced_Multiple_Instance_CVPR_2024_supplemental.pdf)]

**Text Prompt with Normality Guidance for Weakly Supervised Video Anomaly Detection** \
*Zhiwei Yang, Jing Liu , Peng Wu* \
![CLIP-V](https://img.shields.io/badge/CLIP--V-6d4aff)
![CLIP-T](https://img.shields.io/badge/CLIP--T-C3B9FA)\
CVPR 24' [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Text_Prompt_with_Normality_Guidance_for_Weakly_Supervised_Video_Anomaly_CVPR_2024_paper.pdf)][[supp](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Yang_Text_Prompt_with_CVPR_2024_supplemental.pdf)]


## Semi-supervised VAD Papers

**Learning Anomalies with Normality Prior for Unsupervised Video Anomaly Detection** \
*Haoyue Shi, Le Wang, Sanping Zhou, Gang Hua, Wei Tang* \
![ResNext](https://img.shields.io/badge/ResNext-05CC47)\
ECCV 24' [[paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00941.pdf)] 

**Interleaving One-Class and Weakly-Supervised Models with Adaptive Thresholding for Unsupervised Video Anomaly Detection** \
*Yongwei Nie, Hao Huang, Chengjiang Long, Qing Zhang, Pradipta Maji, Hongmin Cai* \
![I3D](https://img.shields.io/badge/I3D-35BF5C)\
ECCV 24' [[paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04450.pdf)][[code](https://github.com/benedictstar/Joint-VAD)]

**Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors** \
*Nicolae-C&#259;t&#259;lin Ristea, Florinel-Alin Croitoru, Radu Tudor Ionescu, Marius Popescu, Fahad Shahbaz Khan, Mubarak Shah* \
CVPR 24' [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Ristea_Self-Distilled_Masked_Auto-Encoders_are_Efficient_Video_Anomaly_Detectors_CVPR_2024_paper.pdf)][[code](https://github.com/ristea/aed-mae/tree/main)][[supp](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Ristea_Self-Distilled_Masked_Auto-Encoders_CVPR_2024_supplemental.pdf)]

**Multi-Scale Video Anomaly Detection by Multi-Grained Spatio-Temporal  Representation Learning** \
![I3D](https://img.shields.io/badge/I3D-35BF5C)\
*Menghao Zhang, Jingyu Wang, Qi Qi, Haifeng Sun, Zirui Zhuang, Pengfei Ren, Ruilong Ma, Jianxin Liao* \
CVPR 24' [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Multi-Scale_Video_Anomaly_Detection_by_Multi-Grained_Spatio-Temporal_Representation_Learning_CVPR_2024_paper.pdf)][[supp](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Zhang_Multi-Scale_Video_Anomaly_CVPR_2024_supplemental.pdf)]

**MULDE: Multiscale Log-Density Estimation via Denoising Score Matching for  Video Anomaly Detection** \
*Jakub Micorek Horst Possegger Dominik Narnhofer Horst Bischof Mateusz Kozi&#769;nski* \
CVPR 24' [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Micorek_MULDE_Multiscale_Log-Density_Estimation_via_Denoising_Score_Matching_for_Video_CVPR_2024_paper.pdf)][[code](https://github.com/jakubmicorek/MULDE-Multiscale-Log-Density-Estimation-via-Denoising-Score-Matching-for-Video-Anomaly-Detection)][[supp](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Micorek_MULDE_Multiscale_Log-Density_CVPR_2024_supplemental.pdf)]

**Collaborative Learning of Anomalies with Privacy (CLAP) for Unsupervised Video Anomaly Detection: A New Baseline** \
*Anas Al-lahham, Muhammad Zaigham Zaheer, Nubrek Tastan, Karthik Nandakumar* \
CVPR 24' [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Al-lahham_Collaborative_Learning_of_Anomalies_with_Privacy_CLAP_for_Unsupervised_Video_CVPR_2024_paper.pdf)][[code](https://github.com/AnasEmad11/CLAP)][[supp](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Al-lahham_Collaborative_Learning_of_CVPR_2024_supplemental.pdf)]

**A Multilevel Guidance-Exploration Network and Behavior-Scene Matching Method for Human Behavior Anomaly Detection** \
*Guoqing Yang, Zhiming Luo, Jianzhe Gao, Yingxin Lai, Kun Yang, Yifan He, Shaozi Li*\
![SwinTrans](https://img.shields.io/badge/SwinTrans-0DBD8B)\
ACM MM 24' [[paper](https://openreview.net/pdf?id=g7zkmttvJp)][[code](https://github.com/molu-ggg/GENet)][[OpenReview](https://openreview.net/forum?id=g7zkmttvJp)]

**Video Anomaly Detection via Progressive Learning of Multiple Proxy Tasks** \
*Menghao Zhang, Jingyu Wang, Qi Qi, Pengfei Ren, Haifeng Sun, Zirui Zhuang, Huazheng Wang, Lei Zhang, Jianxin Liao* \
ACM MM 24' [[paper](https://openreview.net/pdf?id=WsNFULCsyj)][[OpenReview](https://openreview.net/forum?id=WsNFULCsyj&referrer=%5Bthe%20profile%20of%20Lei%20Zhang%5D(%2Fprofile%3Fid%3D~Lei_Zhang67))]

## Fully-supervised VAD Papers
**Exploring Background-bias for Anomaly Detection in Surveillance Videos** \
*Kun Liu, Huadong Ma* \
ACM MM 19' [[paper](https://dl.acm.org/doi/abs/10.1145/3343031.3350998)][[annotation](https://github.com/xuzero/UCFCrime_BoundingBox_Annotation)]

**ANOMALY LOCALITY IN VIDEO SURVEILLANCE** \
*Federico Landi, Cees G.M.Snoek, Rita Cucchiara* \
arXiv 19' [[paper](https://arxiv.org/pdf/1901.10364)][[project](https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=30)][[annotation](https://drive.google.com/drive/folders/1Hu2oke7acBqcKKyUpv5NBmQ2fDiYn87c)]

## Surveys
**Weakly Supervised Anomaly Detection: A Survey** \
*Minqi Jiang, Chaochuan Hou, Ao Zheng, Xiyang Hu, Songqiao Han, Hailiang Huang, Xiangnan He , Philip S. Yu, Yue Zhao* \
arXiv 23' [[paper](https://arxiv.org/pdf/2302.04549)][[repo](https://github.com/yzhao062/wsad)]

**Video Anomaly Detection in 10 Years: A Survey and Outlook** \
SAJID 
*Moshira Abdalla, Sajid Javed, Muaz Al Radi, Anwaar Ulhaq, Naoufel Werghi* \
arXiv 24' [[paper](https://arxiv.org/abs/2405.19387)]

## Datasets
| Dataset    | Download Links | Features | Frame-level Annotation | Publication |
|:-----------|:------------|:------------|:------------|:------------|
| [`ShanghaiTech Campus`](https://svip-lab.github.io/dataset/campus_dataset.html)      | [`BaiduYun`]()        |[`I3D RGB`](https://drive.google.com/file/d/1kIv502RxQnMer-8HB7zrU_GU7CNPNNDv/view?usp=drive_link) | - |CVPR 18'       |
| [`UCF-Crime`](https://www.crcv.ucf.edu/projects/real-world/)      | [`Dropbox`](https://www.dropbox.com/scl/fo/2aczdnx37hxvcfdo4rq4q/AOjRokSTaiKxXmgUyqdcI6k?rlkey=5bg7mxxbq46t7aujfch46dlvz&e=1&dl=0) |  [`I3D RGB`](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/pengwu_stu_xidian_edu_cn/EvYcZ5rQZClGs_no2g-B0jcB4ynsonVQIreHIojNnUmPyA?e=xNrGxc)     |  [`Link`](https://github.com/xuzero/UCFCrime_BoundingBox_Annotation) | CVPR 18'       |
| [`XD-Violence`](https://roc-ng.github.io/XD-Violence/) | [`OneDrive`](https://roc-ng.github.io/XD-Violence/) | [`I3D RGB & VGGish`](https://roc-ng.github.io/XD-Violence/)| - | ECCV 20'|

## Utilities
[Video & Audio Feature Extraction] [`video_features`](https://github.com/v-iashin/video_features): it allows you to extract features from video clips, supporting a variety of modalities and extractors, i.e., S3D, R(2+1)d RGB,  I3D-Net RGB + Flow, VGGish, CLIP.

## Related Repositories
[awesome-video-anomaly-detection](https://github.com/fjchange/awesome-video-anomaly-detection): an awesome collection of papers and codes for video anomaly detection, updated to CVPR 22'.

[WSAD](https://github.com/yzhao062/wsad): a comprehensive collection and categorization of weakly supervised anomaly detection papers.

[awesome anomaly detection](https://github.com/hoya012/awesome-anomaly-detection): a curated list of awesome anomaly detection resources, including time-series anomaly detection, video-level anomaly detection, image-level anomaly detection.