# DconnLoop
A deep learning approach using multi-source data integration to predict chromatin loops.

## Installation
```bash
git clone https://github.com/kuikui-C/DconnLoop.git
cd DconnLoop
conda create -n DconnLoop python==3.8 tensorflow-gpu==2.6.0 scikit-learn imbalanced-learn scipy numpy=1.19.5 pandas h5py cooler tqdm hic-straw pyBigWig statsmodels hdbscan joblib=0.14.1 pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install matplotlib
conda install hicexplorer
conda activate DconnLoop
``````

## Usage
The input data used can be downloaded in the supplementary materials of the paper. The input contact maps use the cool file format, which, if needed, can be converted and normalized using the HiCExplorer's hicConvertFormat command.
### HiC to cool
```bash
hicConvertFormat -m ./ENCFF097SKJ.hic --inputFormat hic --outputFormat cool -o ./ENCFF097SKJ.cool --resolutions 10000
hicConvertFormat -m ./ENCFF097SKJ_10000.cool --inputFormat cool --outputFormat cool -o ./ENCFF097SKJ_KR.cool --correction_name KR
``````

### Generate positive and negative samples
```bash
python PosNeg_Samp_Gen.py -p ./input/gm12878/Rao2014-GM12878-MboI-allreps-filtered.10kb.cool -b ./training-sets/gm12878_ctcf_h3k27ac.bedpe -a ./input/gm12878/ENCFF816ZFB.bigWig -c ./input/gm12878/ENCFF797LSC.bigWig -o ./PosNeg_samp/
``````

### training
```bash
python leave_one_train.py -d ./PosNeg_samp/ -g 1,2,3 -b 256 -lr 0.001 -e 30 -w 0.0005 -c ./model/
``````

### Testing
```bash
python leave_one_test.py -d ./PosNeg_samp/ -g 1,2,3  -c ./model/ -f ./model/chr15-record_test.txt
``````

### Score
```bash
python score_chromosome.py -p ./input/gm12878/Rao2014-GM12878-MboI-allreps-filtered.10kb.cool -a ./input/gm12878/ENCFF816ZFB.bigWig -c ./input/gm12878/ENCFF797LSC.bigWig -q 0.1 -n 12 -o ./scores/ -m ./model/chr15_model_best.pth
``````
