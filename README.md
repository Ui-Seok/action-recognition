# Action-Recognition

### Use model

* LateTemporalModeling3DCNN - R(2+1)D + BERT

### Environment

* OS: Window10
* python version: 3.7.13

### Installation

1. conda create -n LateTemporalModeling3DCNN python=3.7
2. git clone https://github.com/Ui-Seok/latetemporalmodeling3DCNN.git
3. cd latetemporalmodeling3DCNN
4. pip install -r requirements.txt

### How to use

1. Download pre-trained model

   [LateTemporalModelong3DCNN](https://github.com/artest08/LateTemporalModeling3DCNN) Go to this site, download the pre-trained model and save it.
2. Make validation datasets

   Prepare HMDB51 datasets and put the downloaded data file(*.avi) in "datasets/HMDB51". Then run the code shown below.

   ```python
   cd scripts/eval
   python make_datasets.py
   ```
3. Test of the dataset

   After completing the previous process,image file will exist in the format 'img_%05d.jpg' in "datasets/hmdb51_frames". If you check that the file is created and run the code below, you can see the test result.

   ```python
   python spatial_demo_bert.py --arch=rgb_r2plus1d_64f_34_bert10 --split=1
   ```
4. Visualization of the dataset

   If you want to visualize the result, just run the code below.

   ```python
   python recognition.py --arch=rgb_r2plus1d_64f_34_bert10 --split=1
   ```

## Citaton

```python
@inproceedings{kalfaoglu2020late,
title={Late temporal modeling in 3d cnn architectures with bert for action recognition},
author={Kalfaoglu, M Esat and Kalkan, Sinan and Alatan, A Aydin},
booktitle={European Conference on Computer Vision},
pages={731--747},
year={2020},
organization={Springer}
}
```
