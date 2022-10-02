# 代码运行操作


## 环境配置
python == 3.8
torch == 1.10.0+cu111


## 下载数据集
将oracle_fs.zip和oracle_source.zip放在同一文件夹下进行解压

## 训练代码

```shell
# EASY for wideresnet n-shot=1
CUDA_LAUNCH_BLOCKING=1 python -u main_new.py --dataset-path <your-dataset-path>  --dataset oracle --n-shots 1 --mixup --model wideresnet --feature-maps 16 --skip-epochs 90 --epochs 100 --rotations --preprocessing "PEME"
```
