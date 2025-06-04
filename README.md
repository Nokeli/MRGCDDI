# MRGCDDI
The code of MRGCDDI: Multi-Relation Graph Contrastive Learning without Data Augmentation for Drug-Drug Interaction Events Prediction（DOI: 10.1109/JBHI.2024.3483812）
## 安装指南

### 前提条件
- Python 3.8+
- [RDKit](https://www.rdkit.org/docs/Install.html) （推荐通过 `conda` 安装）
- 其他依赖：`numpy`, `pandas`, `tqdm`, `joblib`

### 快速安装
```bash
git clone https://github.com/Nokeli/MRGCDDI.git
cd MOLGAECL
pip install -r requirements.txt

## 运行代码
For how to use MRCGNN, we present an example based on the Deng's dataset.

1. To learn drug structural features from drug molecular graphs, you need to change the path in 'drugfeature_fromMG.py' first. If you want to use MRCGNN on your own dataset, please ensure the data in the 'trimnet' fold and the data in the 'codes for MRCGNN' fold are the same.)
```
python drugfeature_fromMG.py
```
2.Training/validating/testing for 5 times and get the average scores of multiple metrics.
```
python 5timesrun.py
```
3. You can see the final results of 5 runs in 'test.txt'
