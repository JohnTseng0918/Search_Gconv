import os
import torch.utils.data
import torchvision.transforms
from PIL import Image

class ImageNet(torch.utils.data.Dataset):
    def __init__(self, data_path, split):
        #splits = ["train", "val"]
        self._split = split
        self._data_path = data_path
        self._construct_image_database()

    def __getitem__(self, index):
        img_path = self._img_database[index]["img_path"]
        img = Image.open(img_path).convert('RGB')

        if self._split == "train":
            img = torchvision.transforms.RandomResizedCrop(224)(img)
            img = torchvision.transforms.RandomHorizontalFlip()(img)
            img = torchvision.transforms.ToTensor()(img)
            img = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
        else:
            img = torchvision.transforms.Resize(256)(img)
            img = torchvision.transforms.CenterCrop(224)(img)
            img = torchvision.transforms.ToTensor()(img)
            img = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
                        
        return img, self._img_database[index]["class"]

    def __len__(self):
        return len(self._img_database)

    def _construct_image_database(self):
        split_path = os.path.join(self._data_path, self._split)
        dir_name_list = os.listdir(split_path)
        self._class_dict = {dirname: i for i, dirname in enumerate(dir_name_list)}
        self._img_database = []
        for dirname in dir_name_list:
            dir_path = os.path.join(split_path, dirname)
            for filename in os.listdir(dir_path):
                img_path = os.path.join(dir_path, filename)
                self._img_database.append({"img_path":img_path, "class": self._class_dict[dirname]})

