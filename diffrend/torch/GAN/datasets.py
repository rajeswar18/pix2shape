"""Datset loader module."""
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from diffrend.torch.GAN.shapenet import ShapeNetDataset
from diffrend.torch.GAN.objects_folder_multi import ObjectsFolderMultiObjectDataset


class Dataset_load():
    """Load a dataset."""

    def __init__(self, opt):
        """Constructor."""
        self.opt = opt
        self.dataset = None
        self.dataset_loader = None

    def initialize_dataset(self):
        """Initialize."""
        if self.opt.dataset in ['imagenet', 'folder', 'lfw']:
            # folder dataset
            self.dataset = dset.ImageFolder(
                root=self.opt.dataroot,
                transform=transforms.Compose([
                    transforms.Scale(self.opt.imageSize),
                    transforms.CenterCrop(self.opt.imageSize),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]))
        elif self.opt.dataset == 'lsun':
            self.dataset = dset.LSUN(
                db_path=self.opt.dataroot, classes=['bedroom_train'],
                transform=transforms.Compose([
                    transforms.Scale(self.opt.imageSize),
                    transforms.CenterCrop(self.opt.imageSize),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]))
        elif self.opt.dataset == 'cifar10':
            self.dataset = dset.CIFAR10(
                root=self.opt.dataroot, download=True,
                transform=transforms.Compose([
                    transforms.Scale(self.opt.imageSize),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]))
        elif self.opt.dataset == 'fake':
            self.dataset = dset.FakeData(
                image_size=(3, self.opt.imageSize, self.opt.imageSize),
                transform=transforms.ToTensor())
        elif self.opt.dataset == 'shapenet':
            self.dataset = ShapeNetDataset(self.opt, transform=None)
        elif self.opt.dataset == 'objects_folder_multi':
            self.dataset = ObjectsFolderMultiObjectDataset(self.opt, transform=None)

        assert self.dataset

    def initialize_dataset_loader(self, batchSize=None):
        """Create the datset loader."""
        if batchSize is None:
            batchSize = self.opt.batchSize

        self.dataset_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batchSize, shuffle=True,
            num_workers=int(self.opt.workers))

    def get_dataset(self):
        """Get the dataset."""
        if self.dataset is None:
            raise ValueError("Error: Init the dataset first")
        return self.dataset

    def get_dataset_loader(self):
        """Get the dataset loader."""
        if self.dataset_loader is None:
            raise ValueError("Error: Init the loader first")
        return self.dataset_loader
