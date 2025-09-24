# Towards Understanding Continual Factual Knowledge Acquisition of Language Models: From Theory to Algorithm

Continual Factual Knowledge Acquisition (cFKA) is essential for enabling Language Models (LMs) to integrate new facts without erasing prior knowledge. While Continual Pre-Training~(CPT) has become the standard paradigm for injecting factual knowledge, the mechanisms underlying how models acquire and retain facts over time remain unclear. In this work, we present a theoretical framework that characterizes the training dynamics of cFKA using a simplified Transformer with linear attention, offering a unified explanation for the behavior of CPT methods. Our analysis reveals that regularization-based methods merely adjust the convergence rate of parameters without altering the inherent forgetting tendency, whereas data replay methods shift convergence dynamics and stabilize pretrained knowledge, even at low replay ratios. Building on these insights, we propose a novel generative data replay approach, Select Tokens via attentiOn Contribution~(STOC), which identifies influential factual snippets to guide replay generation. Extensive experiments on both synthetic and real-world datasets validate our theoretical findings and demonstrate that STOC effectively enhances continual factual knowledge acquisition by mitigating catastrophic forgetting and improving retention.



## Get Started

#### Create virtual environment
```
conda create -n cfka python=3.11.11
conda activate cfka
pip install -r requirements.txt
```
#### Download Pretrained Models and Datasets if Necessary
After setting id_token, run
```
python download_model.py --model_name Qwen/Qwen2.5-0.5B --folder_path qwen2.5-0.5b
python download_model.py --model_name EleutherAI/pythia-160m --folder_path pythia-160m
```
The real-world benchmark of Knowedit: https://huggingface.co/datasets/zjunlp/KnowEdit

#### Run Experiments of Data Augmentation
```
# cd 1aug/5aug/Paug
cd 1aug
# bash -v run_pythia.sh
bash -v run.sh
```
#### Run Experiment of Continual Pretraining
```
cd 5aug
bash -v run_pythia.sh
bash -v run_qwen.sh
```

#### Run Experiments of Real Datasets
```
cd Real # make sure the dataset is downloaded
bash -v run_pythia_wiki_bio.sh
bash -v run_qwen_zsre.sh
```
