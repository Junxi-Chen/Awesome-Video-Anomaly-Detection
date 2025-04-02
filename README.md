# Awesome Video Anomaly Detection
![GitHub License](https://img.shields.io/github/license/Junxi-Chen/Awesome-Video-Anomaly-Detection)
![Awesome](https://awesome.re/badge.svg)

Video anomaly detection (VAD) aims to identify anomalous frames within given videos, which servers a vital function in critical areas, e.g., public security, media content monitoring and industrial manufacture. This repository collects latest research papers, code, datasets, utilities and related resources for VAD.

If you find this repository helpful, feel free to star🌟 or share it😀! If you spot any errors, notice omissions or have any suggestions, please reach out via GitHub [issues](https://github.com/Junxi-Chen/Awesome-Video-Anomaly-Detection/issues), [pull requests](https://github.com/Junxi-Chen/Awesome-Video-Anomaly-Detection/pulls) or [email]((mailto:chenjunxi22@mails.ucas.ac.cn)).


## Contents
- [Awesome Video Anomaly Detection](#awesome-video-anomaly-detection)
  - [Contents](#contents)
  - [Recent Updates](#recent-updates)
  - [New Setting Papers](#new-setting-papers)
  - [Weakly-supervised VAD Papers](#weakly-supervised-vad-papers)
    - [Prompt Involved Papers](#prompt-involved-papers)
  - [Semi-supervised VAD Papers](#semi-supervised-vad-papers)
  - [Fully-supervised VAD Papers](#fully-supervised-vad-papers)
  - [Surveys](#surveys)
  - [Benchmarks](#benchmarks)
  - [Datasets](#datasets)
    - [Links](#links)
    - [Statistics](#statistics)
  - [Utilities](#utilities)
  - [Related Repositories](#related-repositories)


## Recent Updates
Last Update: April, 2025
- ICLR 25'
- ACM MM 24'
- ECCV 24'


## New Setting Papers
**Toward Video Anomaly Retrieval From Video Anomaly Detection: New Benchmarks and Model**\
*Peng Wu, Jing Liu, Xiangteng He, Yuxin Peng, Peng Wang, Yanning Zhang* \
![I3D](https://img.shields.io/badge/I3D-35BF5C)
![with-Audio](https://img.shields.io/badge/with--Audio-00B2FF)\
TIP 24' [[paper](https://ieeexplore.ieee.org/document/10471334/authors#authors)][[dataset](https://github.com/Roc-Ng/VAR)]

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

**Cross-Modal Fusion and Attention Mechanism for Weakly Supervised Video Anomaly Detection** <a id='HLGAtt'></a> \
*Ayush Ghadiya, Purbayan Kar, Vishal Chudasama, Pankaj Wasnik* \
![I3D](https://img.shields.io/badge/I3D-35BF5C)
![with-Audio](https://img.shields.io/badge/with--Audio-00B2FF)\
CVPR 24' Workshop [[paper](https://openaccess.thecvf.com/content/CVPR2024W/MULA/papers/Ghadiya_Cross-Modal_Fusion_and_Attention_Mechanism_for_Weakly_Supervised_Video_Anomaly_CVPRW_2024_paper.pdf)] 

**Weakly Supervised Video Anomaly Detection and Localization with Spatio-Temporal Prompts** <a id='STPrompt'></a> \
*Peng Wu, Xuerong Zhou, Guansong Pang, Zhiwei Yang, Qingsen Yan, Peng Wang, Yanning Zhang* \
![CLIP-V](https://img.shields.io/badge/CLIP--V-6d4aff)
![CLIP-T](https://img.shields.io/badge/CLIP--T-C3B9FA)\
ACM MM 24' [[paper](https://arxiv.org/pdf/2408.05905)][[OpenReview](https://openreview.net/forum?id=2es1ojI14x&referrer=%5Bthe%20profile%20of%20Peng%20Wu%5D(%2Fprofile%3Fid%3D~Peng_Wu2))]

**Exploiting Completeness and Uncertainty of Pseudo Labels  for Weakly Supervised Video Anomaly Detection** <a id='Zhang-CVPR23'></a>\
*Chen Zhang, Guorong Li, Yuankai Qi, Shuhui Wang, Laiyun Qing, Qingming Huang, Ming-Hsuan Yang* \
![I3D](https://img.shields.io/badge/I3D-35BF5C)\
CVPR 23' [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Exploiting_Completeness_and_Uncertainty_of_Pseudo_Labels_for_Weakly_Supervised_CVPR_2023_paper.pdf)][[code](https://github.com/ArielZc/CU-Net)][[supp](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Zhang_Exploiting_Completeness_and_CVPR_2023_supplemental.pdf)]

**Look Around for Anomalies: Weakly-supervised Anomaly Detection via Context-Motion Relational Learning** <a id='CoMo'></a> \
*MyeongAh Cho, Minjung Kim, Sangwon Hwang, Chaewon Park, Kyungjae Lee, Sangyoun Lee* \
![I3D](https://img.shields.io/badge/I3D-35BF5C)\
CVPR 23' [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Cho_Look_Around_for_Anomalies_Weakly-Supervised_Anomaly_Detection_via_Context-Motion_Relational_CVPR_2023_paper.pdf)][[supp](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Cho_Look_Around_for_CVPR_2023_supplemental.pdf)]

**Graph Convolutional Label Noise Cleaner:
Train a Plug-and-play Action Classifier for Anomaly Detection**\
*Jia-Xing Zhong, Nannan Li, Weijie Kong, Shan Liu, Thomas H. Li, Ge L* <a id='Plug-and-play'></a>\
CVPR 19'[[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhong_Graph_Convolutional_Label_Noise_Cleaner_Train_a_Plug-And-Play_Action_Classifier_CVPR_2019_paper.pdf)

### Prompt Involved Papers
**Vadclip: Adapting vision-language models for weakly supervised video anomaly detection** <a id='Vadclip'></a> \
*Peng Wu, Xuerong Zhou, Guansong Pang, Lingru Zhou, Qingsen Yan, Peng Wang, Yanning Zhang* \
![CLIP-V](https://img.shields.io/badge/CLIP--V-6d4aff)
![CLIP-T](https://img.shields.io/badge/CLIP--T-C3B9FA)\
AAAI 24' [[paper](https://ojs.aaai.org/index.php/AAAI/article/download/28423/28826)][[code](https://github.com/nwpu-zxr/VadCLIP)] 

**Prompt-Enhanced Multiple Instance Learning for Weakly Supervised Video Anomaly Detection** <a id='PE-MIL'></a> \
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

**Local Patterns Generalize Better for Novel Anomalies** \
*Yalong Jiang* \
ICLR 25' [[paper](https://openreview.net/pdf?id=4ua4wyAQLm)][[code](https://github.com/AllenYLJiang/Local-Patterns-Generalize-Better/)][[OpenReview](https://openreview.net/forum?id=4ua4wyAQLm)]

**Learning Anomalies with Normality Prior for Unsupervised Video Anomaly Detection** <a id="LANP"></a> \
*Haoyue Shi, Le Wang, Sanping Zhou, Gang Hua, Wei Tang* \
![ResNext](https://img.shields.io/badge/ResNext-05CC47)\
ECCV 24' [[paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00941.pdf)] 

**Interleaving One-Class and Weakly-Supervised Models with Adaptive Thresholding for Unsupervised Video Anomaly Detection** \
*Yongwei Nie, Hao Huang, Chengjiang Long, Qing Zhang, Pradipta Maji, Hongmin Cai* \
![I3D](https://img.shields.io/badge/I3D-35BF5C)\
ECCV 24' [[paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04450.pdf)][[code](https://github.com/benedictstar/Joint-VAD)]

**Scene-Dependent Prediction in Latent Space for Video Anomaly Detection and Anticipation** <a id="SSAE"></a> \
*Congqi Cao, Hanwen Zhang, Yue Lu, Peng Wang, Yanning Zhang* \
T-PAMI 24'[[paper](https://ieeexplore.ieee.org/abstract/document/10681297)][[project](https://campusvaa.github.io)][[code](https://github.com/zugexiaodui/campus_vad_code)][[dataset](https://drive.google.com/drive/folders/1_EztmkNpTPyVb4lM0m4rLTXgXo_LzgF1?usp=share_link)]

**DoTA: Unsupervised Detection of Traffic Anomaly in Driving Videos** \
*Yu Yao, Xizi Wang, Mingze Xu, Zelin Pu, Yuchen Wang, Ella Atkins, Senior Member, David J. Crandall* \
T-PAMI 23' [[paper](https://ieeexplore.ieee.org/document/9712446)][[code](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly)][[dataset](https://drive.google.com/drive/folders/1_WzhwZC2NIpzZIpX7YCvapq66rtBc67n)]

**Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors** <a id="AED-MAE"></a> \
*Nicolae-C&#259;t&#259;lin Ristea, Florinel-Alin Croitoru, Radu Tudor Ionescu, Marius Popescu, Fahad Shahbaz Khan, Mubarak Shah* \
CVPR 24' [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Ristea_Self-Distilled_Masked_Auto-Encoders_are_Efficient_Video_Anomaly_Detectors_CVPR_2024_paper.pdf)][[code](https://github.com/ristea/aed-mae/tree/main)][[supp](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Ristea_Self-Distilled_Masked_Auto-Encoders_CVPR_2024_supplemental.pdf)]

**Multi-Scale Video Anomaly Detection by Multi-Grained Spatio-Temporal  Representation Learning** <a id="Zhang-CVPR24"></a> \
*Menghao Zhang, Jingyu Wang, Qi Qi, Haifeng Sun, Zirui Zhuang, Pengfei Ren, Ruilong Ma, Jianxin Liao* \
![I3D](https://img.shields.io/badge/I3D-35BF5C) \
CVPR 24' [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Multi-Scale_Video_Anomaly_Detection_by_Multi-Grained_Spatio-Temporal_Representation_Learning_CVPR_2024_paper.pdf)][[supp](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Zhang_Multi-Scale_Video_Anomaly_CVPR_2024_supplemental.pdf)]

**MULDE: Multiscale Log-Density Estimation via Denoising Score Matching for  Video Anomaly Detection** <a id="MULDE"></a> \
*Jakub Micorek Horst Possegger Dominik Narnhofer Horst Bischof Mateusz Kozi&#769;nski* \
![CLIP-V](https://img.shields.io/badge/CLIP--V-6d4aff) ![Hiera-L](https://img.shields.io/badge/Hiera--L-25D366) \
CVPR 24' [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Micorek_MULDE_Multiscale_Log-Density_Estimation_via_Denoising_Score_Matching_for_Video_CVPR_2024_paper.pdf)][[code](https://github.com/jakubmicorek/MULDE-Multiscale-Log-Density-Estimation-via-Denoising-Score-Matching-for-Video-Anomaly-Detection)][[supp](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Micorek_MULDE_Multiscale_Log-Density_CVPR_2024_supplemental.pdf)]

**Collaborative Learning of Anomalies with Privacy (CLAP) for Unsupervised Video Anomaly Detection: A New Baseline** \
*Anas Al-lahham, Muhammad Zaigham Zaheer, Nubrek Tastan, Karthik Nandakumar* \
CVPR 24' [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Al-lahham_Collaborative_Learning_of_Anomalies_with_Privacy_CLAP_for_Unsupervised_Video_CVPR_2024_paper.pdf)][[code](https://github.com/AnasEmad11/CLAP)][[supp](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Al-lahham_Collaborative_Learning_of_CVPR_2024_supplemental.pdf)]

**A Multilevel Guidance-Exploration Network and Behavior-Scene Matching Method for Human Behavior Anomaly Detection** <a id="MGENet"></a>  \
*Guoqing Yang, Zhiming Luo, Jianzhe Gao, Yingxin Lai, Kun Yang, Yifan He, Shaozi Li*\
![SwinTrans](https://img.shields.io/badge/SwinTrans-0DBD8B)\
ACM MM 24' [[paper](https://openreview.net/pdf?id=g7zkmttvJp)][[code](https://github.com/molu-ggg/GENet)][[OpenReview](https://openreview.net/forum?id=g7zkmttvJp)]

**Video Anomaly Detection via Progressive Learning of Multiple Proxy Tasks** <a id="Zhang-MM24"></a> \
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

**A Survey on Graph Neural Networks for Time Series: Forecasting, Classification, Imputation, and Anomaly Detection** \
*Ming Jin, Huan Yee Koh, Qingsong Wen, Daniele Zambon, Cesare Alippi, Geoffrey I. Webb, Irwin King, Shirui Pan* \
T-PAMI 24' [[paper](https://github.com/KimMeen/Awesome-GNN4TS)][[repo](https://github.com/KimMeen/Awesome-GNN4TS)]

**Graph-Time Convolutional Neural Networks: Architecture and Theoretical Analysis** \
*Mohammad Sabbaqi and Elvin Isufi* \
T-PAMI 23' [[paper](https://ieeexplore.ieee.org/abstract/document/10239277)]


## Benchmarks
<table>
  <tr>
    <th></th>
     <th>Method</th>
     <th>Publication</th>
     <th>Visual Features</th>
     <th>STC (AUC)</th>
     <th>UCF (AUC)</th>
     <th>XDV (AP)</th>
     <th>Ave (AUC)</th>
     <th>Cor (AUC)</th>
     <th>UBnormal (AUC)</th>
     <th>Ped2 (AUC)</th>
     <th>Campus (AUC)</th>
     <th>NWPU (AUC)</th>
     <th>NPDI (AUC)</th>
     <th>TAD (AUC)</th>
     <th>Audio Features</th>
     <th>Text Prompt</th>
     <th>LLM Involved</th>
     <th>Data Augumentation</th>
  </tr>
  <tr>
    <td rowspan="8">Semi-supervised</td>
    <td rowspan="2"><a href="#LANP">LANP</a></td>
    <td rowspan="2">ECCV 24'</td>
    <td>I3D</td>
    <td>88.32</td>
    <td>80.02</td>
    <td rowspan="2"> - </td>
    <td rowspan="2"> - </td>
    <td rowspan="2"> - </td>
    <td rowspan="2"> - </td>
    <td rowspan="2"> - </td>
    <td rowspan="2"> - </td>
    <td rowspan="2"> - </td>
    <td rowspan="2"> - </td>
    <td rowspan="2"> - </td>
    <td rowspan="2"> - </td>
    <td rowspan="2"> - </td>
    <td rowspan="2"> - </td>
    <td rowspan="2"> - </td>
  </tr>
  <tr>
    <td>ResNext</td>
    <td>86.46</td>
    <td>76.64</td>
  </tr>
  <tr>
    <td><a href="#SSAE">SSAE</a></td>
    <td>T-PAMI 24'</td>
    <td> - </td>
    <td>80.5</td>
    <td> - </td>
    <td> - </td>
    <td>90.2</td>
    <td>75.8</td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td>75.6</td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
  </tr>
  <tr>
    <td><a href="#AED-MAE">AED-MAE</a></td>
    <td>CVPR 24'</td>
    <td> - </td>
    <td>79.1</td>
    <td> - </td>
    <td> - </td>
    <td>91.3</td>
    <td> - </td>
    <td>58.5</td>
    <td>95.4</td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
  </tr>
  <tr>
    <td><a href="#Zhang-CVPR24">Zhang et al.</a></td>
    <td>CVPR 24'</td>
    <td>I3D</td>
    <td> 87.5 </td>
    <td> 80.6 </td>
    <td> - </td>
    <td> 94.3 </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> 70.1 </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
  </tr>
  <tr>
    <td><a href='#MULDE'>MULDE</a></td>
    <td>CVPR 24'</td>
    <td>CLIP+Hiera</td>
    <td>81.3</td>
    <td>78.5</td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td>72.8</td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
  </tr>
  <tr>
    <td><a href='#MGENet'>MGEnet</a></td>
    <td>MM 24'</td>
    <td>Video Swin</td>
    <td>86.9</td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> 74.3 </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
  </tr>
  <tr>
    <td><a href='#Zhang-MM24'>Zhang et al.</a></td>
    <td>MM 24'</td>
    <td> - </td>
    <td>88.6</td>
    <td>83.2</td>
    <td> - </td>
    <td> 94.5 </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
  </tr>
  <tr>
    <th colspan="19"></th>
  </tr>
  <tr>
    <td rowspan="8">Weakly-supervised</td>
    <td><a href='#HLGAtt'>HLGAtt</a></td>
    <td>CVPR 24' Workshop</td>
    <td>I3D</td>
    <td> - </td>
    <td> - </td>
    <td> 86.34 </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> 99.45 </td>
    <td> - </td>
    <td> VGGish </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
  </tr>
  <tr>
    <td><a href='#STPrompt'>STPrompt</a></td>
    <td>MM 24'</td>
    <td>CLIP</td>
    <td> 97.81 </td>
    <td> 88.08 </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> 63.98 </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> &#10004; </td>
    <td> - </td>
    <td> - </td>
  </tr>
  <tr>
    <td><a href='#Zhang-CVPR23'>Zhang et al.</a></td>
    <td>CVPR 23'</td>
    <td>I3D</td>
    <td> - </td>
    <td> 86.22 </td>
    <td> 81.43 </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> 91.66 </td>
    <td> VGGish </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
  </tr>
  <tr>
    <td><a href='#CoMo'>CoMo</a></td>
    <td>CVPR 23'</td>
    <td>I3D</td>
    <td>97.6</td>
    <td>86.1</td>
    <td>81.3</td>
    <td>89.8</td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
  </tr>
  <tr>
    <td><a href='#Vadclip'>Vadclip</a></td>
    <td>AAAI 24'</td>
    <td>CLIP</td>
    <td> - </td>
    <td> 88.02 </td>
    <td> 84.51 </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> &#10004; </td>
    <td> - </td>
    <td> - </td>
  </tr>
  <tr>
    <td><a href='#PE-MIL'>PE-MIL</a></td>
    <td>CVPR 24'</td>
    <td>I3D</td>
    <td> 98.35 </td>
    <td> 86.83 </td>
    <td> 88.21 </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> VGGish </td>
    <td> &#10004; </td>
    <td> - </td>
    <td> - </td>
  </tr>
</table>


## Datasets
### Links
| Dataset | Download Links | Features | Frame-level Annotation | Publication |
|:-----------|:------------|:------------|:------------|:------------|
| [`ShanghaiTech Campus`](https://svip-lab.github.io/dataset/campus_dataset.html)      | [`BaiduYun`](https://pan.baidu.com/s/1W3_tkiiNHKd_4uQ2tgBq8g?pwd=3mh5)        |[`I3D RGB`](https://drive.google.com/file/d/1kIv502RxQnMer-8HB7zrU_GU7CNPNNDv/view?usp=drive_link) | - |CVPR 18'       |
| [`UCF-Crime`](https://www.crcv.ucf.edu/projects/real-world/)      | [`Dropbox`](https://www.dropbox.com/scl/fo/2aczdnx37hxvcfdo4rq4q/AOjRokSTaiKxXmgUyqdcI6k?rlkey=5bg7mxxbq46t7aujfch46dlvz&e=1&dl=0) |  [`I3D RGB`](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/pengwu_stu_xidian_edu_cn/EvYcZ5rQZClGs_no2g-B0jcB4ynsonVQIreHIojNnUmPyA?e=xNrGxc)     |  [`Link`](https://github.com/xuzero/UCFCrime_BoundingBox_Annotation) | CVPR 18'       |
| [`XD-Violence`](https://roc-ng.github.io/XD-Violence/) | [`OneDrive`](https://roc-ng.github.io/XD-Violence/) | [`I3D RGB & VGGish`](https://roc-ng.github.io/XD-Violence/)| - | ECCV 20'|

### Statistics
<table border="1">
    <tr>
        <th>Dataset</th>
        <th>Year</th>
        <th>Modality</th>
        <th>#Videos</td>
        <th>Supervision</th>
        <th>#Training Abnormal Videos</th>
        <th>#Training Normal Videos</th>
        <th>#Test Abnormal Videos</th>
        <th>#Test Normal Videos</th>
        <th>#Anomaly Types</th>
    </tr>
    <tr>
        <td rowspan="2">ShanghaiTech</td>
        <td rowspan="2">2018</td>
        <td rowspan="2">Visual</td>
        <td rowspan="2">437</td>
        <td>Semi</td>
        <td>-</td>
        <td>330</td>
        <td>107</td>
        <td>-</td>
        <td rowspan="2">13</td>
    </tr>
    <tr>
        <td>Weakly*</td>
        <td>63</td>
        <td>175</td>
        <td>44</td>
        <td>155</td>
    </tr>
    <tr>
        <td rowspan="2">UCF-Crime</td>
        <td rowspan="2">2018</td>
        <td rowspan="2">Visual</td>
        <td rowspan="2">1900</td>
        <td>Semi<sup>†</sup></td>
        <td>-</td>
        <td>800</td>
        <td>140</td>
        <td>150</td>
        <td rowspan="2">13</td>
    </tr>
    <tr>
        <td>Weakly</td>
        <td>810</td>
        <td>800</td>
        <td>140</td>
        <td>150</td>
    </tr>
    <tr>
        <td>XD-Violence</td>
        <td>2020</td>
        <td>Visual & Audio</td>
        <td>4754</td>
        <td>Weakly</td>
        <td>1905</td>
        <td>2049</td>
        <td>500</td>
        <td>300</td>
        <td>7</td>
    </tr>
    <tr>
        <td>ECVA</td>
        <td>2024</td>
        <td>Visual & Audio & Text</td>
        <td>2240</td>
        <td>Semi</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>100</td>
    </tr>
    <tr>
        <td rowspan="2">MSAD</td>
        <td rowspan="2">2024</td>
        <td rowspan="2">Visual</td>
        <td rowspan="2">720</td>
        <td>Semi</td>
        <td>-</td>
        <td>360</td>
        <td>240</td>
        <td>120</td>
        <td rowspan="2">55</td>
    </tr>
    <tr>
        <td>Weakly</td>
        <td>120</td>
        <td>360</td>
        <td>120</td>
        <td>120</td>
    </tr>
</table>
<div style="font-size: smaller; margin-top: 1em;">
    * : ShanghaiTech was initially proposed as a semi-supervised VAD dataset, and <a href='#Plug-and-play'>Zhong etal</a>. later introduced its weakly supervised split.<br>
    †: Derived from <a href='#MULDE'>MULDE</a>.
</div>

## Utilities
[Video & Audio Feature Extraction] [`video_features`](https://github.com/v-iashin/video_features): it allows you to extract features from video clips, supporting a variety of modalities and extractors, i.e., S3D, R(2+1)d RGB,  I3D-Net RGB + Flow, VGGish, CLIP.


## Related Repositories
[awesome-video-anomaly-detection](https://github.com/fjchange/awesome-video-anomaly-detection): an awesome collection of papers and codes for video anomaly detection, updated to CVPR 22'.

[WSAD](https://github.com/yzhao062/wsad): a comprehensive collection and categorization of weakly supervised anomaly detection papers.

[awesome anomaly detection](https://github.com/hoya012/awesome-anomaly-detection): a curated list of awesome anomaly detection resources, including time-series anomaly detection, video-level anomaly detection, image-level anomaly detection.