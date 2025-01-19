**Official PyTorch code for the SPL2025 paper "Blind Light Field Image Quality Assessment via Frequency Domain Analysis and Auxiliary Learning". Please refer to the [paper](https://ieeexplore.ieee.org/document/10844526) for details.**

![image](https://github.com/oldblackfish/FABLFQA/blob/main/fig/framework.png)

**Note: First, we convert the dataset into H5 files using MATLAB. Then, we train and test the model in Python.**

### Generate Dataset in MATLAB
Take the NBU-LF1.0 dataset for instance, convert the dataset into h5 files, and then put them into './Datasets/NBU_FABLFQA_5x64x64/':
```
 ./FABLFQA/Datasets/Generateh5_for_NBU_Dataset.m
```
    
### Train
Train the model using the following command:
```
python Train.py  --trainset_dir ./Datasets/NBU_FABLFQA_5x64x64/
```

### Test Overall Performance
Test the overall performance using the following command:
```
python Test.py
```

### Test Individual Distortion Type Performance
Test the individual distortion type performance using the following command:
```
 python Test_Dist.py
```
### Acknowledgement
This project is based on [DeeBLiF](https://github.com/ZhengyuZhang96/DeeBLiF). Thanks for the awesome work.

### Citation
Please cite the following paper if you use this repository in your reseach.
```
@ARTICLE{10844526,
  author={Zhou, Rui and Jiang, Gangyi and Zhu, Linwei and Cui, Yueli and Luo, Ting},
  journal={IEEE Signal Processing Letters}, 
  title={Blind Light Field Image Quality Assessment via Frequency Domain Analysis and Auxiliary Learning}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Measurement;Feature extraction;Discrete cosine transforms;Distortion;Light fields;Three-dimensional displays;Spatial resolution;Visualization;Frequency conversion;Frequency-domain analysis;Light field;blind image quality assessment;frequency domain;auxiliary learning;deep learning network},
  doi={10.1109/LSP.2025.3531209}}
```
### Contact
For any questions, feel free to contact: 2211100079@nbu.edu.cn or blackfish5254@gmail.com
