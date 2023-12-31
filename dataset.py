from PIL import Image
from torch.utils.data import Dataset


class V90kList(Dataset):
    def __init__(
        self,
        data_dir="./",
        list_path="v90k_list",
        transform=None,
        cnt=10000,
        split="train",
    ):
        with open(data_dir + split + "/" + list_path, "r") as f:
            self.samples = f.read().split("\n")

        self.data_dir = data_dir
        self.transform = transform
        self.samples = self.samples[:cnt]
        self.split = split

    def __getitem__(self, index):
        img = Image.open(
            self.data_dir + self.split + "/" + self.samples[index]
        ).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)
