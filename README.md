# [NExT-QA](https://arxiv.org/pdf/2105.08276.pdf) <img src="images/logo.png" height="64" width="128">

We reproduce some SOTA VideoQA methods to provide benchmark results for our NExT-QA dataset accepted to CVPR2021. 

NExT-QA is a VideoQA benchmark targeting the explanation of video contents. It challenges QA models to reason about the causal and temporal actions and understand the rich object interactions in daily activities. We set up both multi-choice and open-ended QA tasks on the dataset. <strong>This repo. provides resources for open-ended QA</strong>; multi-choice QA is found in [NExT-QA](https://github.com/doc-doc/NExT-QA). For more details, please refer to our [dataset](https://doc-doc.github.io/docs/nextqa.html) page.

## Todo
1. [ ] <s>Open online evaluation server</s> and release [test data](https://drive.google.com/file/d/1bXBFN61PaTSHTnJqz3R79mpIgEQPFGIU/view?usp=sharing).
2. [ ] <s>Release spatial feature</s>.
3. [ ] Release RoI feature.
## Environment

Anaconda 4.8.4, python 3.6.8, pytorch 1.6 and cuda 10.2. For other libs, please refer to the file requirements.txt.

## Install
Please create an env for this project using anaconda (should install [anaconda](https://docs.anaconda.com/anaconda/install/linux/) first)
```
>conda create -n videoqa python==3.6.8
>conda activate videoqa
>git clone https://github.com/doc-doc/NExT-OE.git
>pip install -r requirements.txt
```
## Data Preparation
Please download the pre-computed features and QA annotations from [here](https://drive.google.com/drive/folders/14jSt4sGFQaZxBu4AGL2Svj34fUhcK2u0?usp=sharing). There are 3 zip files: 
- ```['vid_feat.zip']```: Appearance and motion feature for video representation (same as multi-choice QA).
- ```['nextqa.zip']```: Annotations of QAs and GloVe Embeddings (open-ended version). 
- ```['models.zip']```: HGA model (open-ended version). 

After downloading the data, please create a folder ```['data/feats']``` at the same directory as ```['NExT-OE']```, then unzip the video features into it. You will have directories like ```['data/feats/vid_feat/', and 'NExT-OE/']``` in your workspace. Please unzip the files in ```['nextqa.zip']``` into ```['NExT-OE/dataset/nextqa']``` and ```['models.zip']``` into ```['NExT-OE/models/']```. 


## Usage
Once the data is ready, you can easily run the code. First, to test the environment and code, we provide the prediction and model of the SOTA approach (i.e., HGA) on NExT-QA. 
You can get the results reported in the paper by running: 
```
>python eval_oe.py
```
The command above will load the prediction file under ['results/'] and evaluate it. 
You can also obtain the prediction by running: 
```
>./main.sh 0 val #Test the model with GPU id 0
```
The command above will load the model under ['models/'] and generate the prediction file.
If you want to train the model, please run
```
>./main.sh 0 train # Train the model with GPU id 0
```
It will train the model and save to ['models']. (*The results may be slightly different depending on the environments*)
## Results on Val
| Methods                  | Text Rep. | WUPS_C | WUPS_T | WUPS_D | WUPS | 
| -------------------------| --------: | ----: | ----: | ----: | ---:| 
| BlindQA                  |   GloVe   | 12.14 | 14.85 | 40.41 | 18.88 | 
| [STVQA](https://github.com/doc-doc/NExT-OE/blob/main/networks/VQAModel/STVQA.py) ([CVPR17](https://openaccess.thecvf.com/content_cvpr_2017/papers/Jang_TGIF-QA_Toward_Spatio-Temporal_CVPR_2017_paper.pdf))  |   GloVe   | 12.52 | 14.57 | 45.64 | 20.08 | 
| [UATT](https://github.com/doc-doc/NExT-OE/blob/main/networks/VQAModel/UATT.py) ([TIP17](https://ieeexplore.ieee.org/document/8017608)) | GloVe | 13.62 | **16.23** | 43.41 | 20.65 |
| [HME](https://github.com/doc-doc/NExT-OE/blob/main/networks/VQAModel/HME.py) ([CVPR19](https://openaccess.thecvf.com/content_CVPR_2019/papers/Fan_Heterogeneous_Memory_Enhanced_Multimodal_Attention_Model_for_Video_Question_Answering_CVPR_2019_paper.pdf))   |   GloVe   | 12.83 | 14.76 | 45.13 | 20.18 | 
| [HCRN](https://github.com/thaolmk54/hcrn-videoqa) ([CVPR20](https://openaccess.thecvf.com/content_CVPR_2020/papers/Le_Hierarchical_Conditional_Relation_Networks_for_Video_Question_Answering_CVPR_2020_paper.pdf))   |   GloVe   | 12.53 | 15.37 | 45.29 | 20.25 | 
| [HGA](https://github.com/doc-doc/NExT-OE/blob/main/networks/VQAModel/HGA.py) ([AAAI20](https://ojs.aaai.org//index.php/AAAI/article/view/6767))    |   GloVe   | **14.76** | 14.90 | **46.60** | **21.48** |

Please refer to our paper for results on the test set.
## Multi-choice QA *vs.* Open-ended QA
![vis mc_oe](./images/res-mc-oe.png)

## Some Latest Results
| Methods                  | Publication | Highlight | Val (WUPS@All)   | Test (WUPS@All) | 
| -------------------------| --------:   |--------:  | ----:            | ----:| 
|[HGA](https://ojs.aaai.org/index.php/AAAI/article/view/6767) | AAAI'20 | Heterogenous Graph | 21.5 | 25.2 |
| [Flamingo(0-shot)](https://arxiv.org/pdf/2204.14198.pdf) | arXiv by DeepMind            |   VL foundation model   | -     | 26.7|
| [Flamingo(32-shot)](https://arxiv.org/pdf/2204.14198.pdf)   | arXiv by DeepMind            |  VL foundation model   | -     | 33.5|

## Citation
```
@InProceedings{xiao2021next,
    author    = {Xiao, Junbin and Shang, Xindi and Yao, Angela and Chua, Tat-Seng},
    title     = {NExT-QA: Next Phase of Question-Answering to Explaining Temporal Actions},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {9777-9786}
}
```
## Acknowledgement
Our reproduction of the methods is based on the respective official repositories, we thank the authors to release their code. If you use the related part, please cite the corresponding paper commented in the code.
