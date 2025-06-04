# MRGCDDI  

**Multi-Relation Graph Contrastive Learning without Data Augmentation for Drug-Drug Interaction Events Prediction**  
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FJBHI.2024.3483812-blue)](https://doi.org/10.1109/JBHI.2024.3483812)

Official implementation of the MRGCDDI model for drug-drug interaction prediction.

## Installation Guide  

### Prerequisites  
- Python 3.8+  
- [RDKit](https://www.rdkit.org/docs/Install.html) (Recommended to install via `conda`)  
- Other dependencies: `numpy`, `pandas`, `tqdm`, `joblib`  

### Quick Installation  
```bash  
git clone https://github.com/Nokeli/MRGCDDI.git  
cd MRGCDDI  
pip install -r requirements.txt

## run code
We provide a sample based on Deng's dataset.

1.In order to learn the desired representation, the path of the dataset in the 'drugfeature_fromMG.py' file needs to be changed to your own path. If you want to run it on your own dataset, please ensure that the data in the "trimnet" fold is the same as that in the fold.
```
python drugfeature_fromMG.py
```

2.Training/validating/testing for 5 times and get the average scores of multiple metrics.
```
python 5timesrun.py
```

3. You can see the final results of 5 runs in 'test.txt'
