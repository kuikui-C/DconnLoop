# DconnLoop
A deep learning approach using multi-source data integration to predict chromatin loops.

## Installation
```bash
git clone https://github.com/kuikui-C/DconnLoop.git
cd DconnLoop
conda create -n DconnLoop python==3.8 tensorflow-gpu==2.6.0 scikit-learn imbalanced-learn scipy numpy=1.19.5 pandas h5py cooler tqdm hic-straw pyBigWig statsmodels hdbscan joblib=0.14.1 pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install matplotlib
conda activate DconnLoop
``````
## Usage
