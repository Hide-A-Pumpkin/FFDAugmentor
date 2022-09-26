# Implementation of FFD Augmentor: Towards Few-Shot Oracle Character Recognition from Scratch

Xinyi Zhao, Siyuan Liu, Yikai Wang, Yanwei Fu

(This paper is the final project of Neural Network and Deep Learning (DATA130011.01), School of Data Science, Fudan University.)

## Abstract

Recognizing oracle characters, the earliest hieroglyph discovered in China, is recently addressed with more and more attention. Due to the difficulty of collecting labeled data, recognizing oracle characters is naturally a Few-Shot Learning (FSL) problem, which aims to tackle the learning problem with only one or a few training data. Most current FSL methods assume a disjoint but related big dataset can be utilized such that one can transfer the related knowledge to the few-shot case. However, unlike common phonetic words like English letters, oracle bone inscriptions are composed of radicals representing graphic symbols. Furthermore, as time goes, the graphic symbols to represent specific objects were significantly changed. Hence we can hardly find plenty of prior knowledge to learn without negative transfer. Another perspective to solve this problem is to use data augmentation algorithms to directly enlarge the size of training data to help the training of deep models. But popular augment strategies, such as dividing the characters to stroke sequences breaks the orthographic units of Chinese characters and destroys the semantic information.  Thus simply adding noise to strokes perform weakly in enhancing the learning capacity.

To solve these issues, we in this paper propose a new data augmentation algorithm for oracle characters such that (1) it will introduce informative diversity for the training data while alleviate the loss of semantics; (2) with this data augmentation algorithm, we can train the few-shot model from scratch without pre-training and still get a powerful recognition model with superior performance to models pre-trained with a large dataset. Specifically, our data augmentation algorithm includes a B-spline free form deformation method to randomly distort the strokes of characters but maintain the overall structures. We generate 20 - 40 augmented images for each training data and use this augmented training set to train a deep neural network model in a standard pipeline. Extensive experiments on several benchmark datasets demonstrate the effectiveness of our augmentor.

## Environment

python == 3.8
torch == 1.10.0+cu111

## Preparing the Dataset

You can get the dataset on 

[Oracle_FS](https://github.com/avalonstrel/SketchBERT) 

HWOBC

Sketch

## Generate Augmented Data

* To automatically generate the data augmentation file, put the raw data in the ``oralce_fs`` folder or modify the ``path`` variable on line 198 to point the path to the original training set.

* line192-195 are arguments:

    block_num: num of FFD control points(patches)

    offset: max offset value

    num: num of augmented sample

    shot: k-shot(k=1,2,3…)

* After successfully running the code, a folder corresponding to the hyperparameters ``path`` will be generated under the path.

    ```shell
    python FFD_augmentor.py
    ```

    

## Start Training and Testing

```shell
# oracle-fs
python main_new.py --dataset-path ../oracle_fs --dataset oracle --n-shots 1 --mixup --model wideresnet --feature-maps 16 --skip-epochs 90 --epochs 100 --rotations --preprocessing "PEME"
```



**Training arguments**

-   `dataset`: choices=['oracleAug', 'HWOBCAug','sketch']

-   `model`: choices=['resnet12', 'resnet18', 'resnet20', 'wideresnet']

-   `dataset-path`: path of the datasets folder which contains folders of all the datasets.

-   `cosine` : if mentionned, cosine scheduler will be used during training.

-   `save-model`: path where to save the best model based on validation data.

-   `skip-epochs`: number of epochs to skip before evaluating few-shot performance. Used to speed-up training.

-   `n-shots` : how many shots per few-shot run, can be int or list of ints.

    

## Performance

Best accuracy on Oracle-FS

| k-shot | EASY  | FFD Augmentor+EASY |
| ------ | ----- | ------------------ |
| 1      | 55.17 | **78.90**          |
| 3      | 79.45 | **93.36**          |
| 5      | 90.34 | **96.47**          |

Accuracy on 1-shot HWOBC

| model      | EASY  | FFD Augmentor+EASY |
| ---------- | ----- | ------------------ |
| Resnet12   | 65.24 | **98.53**          |
| Resnet18   | 41.85 | **98.97**          |
| Resnet20   | 52.32 | **98.60**          |
| Wideresnet | 62.16 | **99.52**          |