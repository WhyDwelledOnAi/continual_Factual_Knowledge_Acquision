# cFKA

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
