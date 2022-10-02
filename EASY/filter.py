from torchvision.datasets.folder import ImageFolder
import pathlib

dataset = ImageFolder("../oracle_fs/img/oracle_200_1_shot/train/", loader=lambda path: pathlib.Path(path).name)

for path, category in dataset:
    print(path, category)