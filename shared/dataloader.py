import os.path
import numpy as np
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import *
from torch.utils import data
from PIL import Image
import urllib.request
import zipfile
from torch.utils.data import random_split
import random
# from data_processor import DataProcessor
# from unused_function.lisa import LISA
import ssl; ssl._create_default_https_context = ssl._create_unverified_context

# CIFAR mirror fallback: cs.toronto.edu has been 503-ing since 2026-05. Try
# the canonical host first; on failure fall through to a byte-identical
# Internet Archive snapshot. torchvision's MD5 check runs either way, so the
# fallback can only succeed if the file is the genuine canonical tarball.
from torchvision.datasets.utils import download_and_extract_archive as _dl_extract
def _install_cifar_fallback(cls, urls):
    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        for i, url in enumerate(urls):
            try:
                _dl_extract(url, self.root, filename=self.filename, md5=self.tgz_md5)
                return
            except Exception as e:
                if i == len(urls) - 1:
                    raise
                print(f"[dataloader] {cls.__name__} download from {url} failed ({e}); trying mirror")
    cls.download = download
_install_cifar_fallback(CIFAR10, [
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "https://web.archive.org/web/20241225200100id_/https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
])
_install_cifar_fallback(CIFAR100, [
    "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
    "https://web.archive.org/web/20241225200100id_/https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
])

NLP_DATASETS = {'sst2', 'mrpc', 'rte', 'cola'}

root_map = {
    'cifar10': '.data/',
    'cifar100': '.data/',
    'subcifar': '.data/',
    'gtsrb': '.data/',
    'subgtsrb': '.data/',
    'svhn': '.data/',
    'flower': '.data/',
    'pubfig': '../dataset/pubfig',
    'eurosat': '.data/eurosat',
    'imagenet': '../imagenet/',
    'imagenet64': '../imagenet64/',
    'lisa': '.data/',
    'mnist': '.data/',
    'mnistm': '.data/',
    'food': '.data/',
    'pet': '.data/',
    'resisc': '.data/NWPU-RESISC45/',
    'voc': '.data/',
    'lingspam': '.data/spam/lingspam',
    'sst2':     '.data/glue',
    'mrpc':     '.data/glue',
    'rte':      '.data/glue',
    'cola':     '.data/glue',
}
mean_map = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.4914, 0.4822, 0.4465),
    'subcifar': (0.4914, 0.4822, 0.4465),
    'mnist': (0.5, 0.5, 0.5),
    'mnistm': (0.5, 0.5, 0.5),
    'imagenet': (0.485, 0.456, 0.406),
    'imagenet64': (0.485, 0.456, 0.406),
    'flower': (0.485, 0.456, 0.406),
    'caltech101': (0.485, 0.456, 0.406),
    'stl10': (0.485, 0.456, 0.406),
    'iris': (0.485, 0.456, 0.406),
    'fmnist': (0.2860, 0.2860, 0.2860),
    'svhn': (0.5, 0.5, 0.5),
    'gtsrb': (0.3403, 0.3121, 0.3214),  # (0.5, 0.5, 0.5), #
    'subgtsrb': (0.5, 0.5, 0.5),  # (0.3403, 0.3121, 0.3214),#
    'pubfig': (129.1863 / 255.0, 104.7624 / 255.0, 93.5940 / 255.0),
    'lisa': (0.3403, 0.3121, 0.3214),  #(0.4563, 0.4076, 0.3895),
    'eurosat': (0.3442, 0.3802, 0.4077),
     'food': (0.5, 0.5, 0.5),
    'pet': (0.5, 0.5, 0.5),
    'resisc': (0.5, 0.5, 0.5),
    'voc': (0.5, 0.5, 0.5),
    'lingspam': (0.5, 0.5, 0.5),
    'unknown': (0.5, 0.5, 0.5),
}
std_map = {
    'cifar10': (0.2023, 0.1994, 0.201),
    'cifar100': (0.2023, 0.1994, 0.201),
    'subcifar': (0.2023, 0.1994, 0.201),
    'mnist': (0.5, 0.5, 0.5),
    'mnistm': (0.5, 0.5, 0.5),
    'imagenet': (0.229, 0.224, 0.225),
    'imagenet64': (0.229, 0.224, 0.225),
    'flower': (0.229, 0.224, 0.225),
    'caltech101': (0.229, 0.224, 0.225),
    'stl10': (0.229, 0.224, 0.225),
    'iris': (0.229, 0.224, 0.225),
    'fmnist': (0.3530, 0.3530, 0.3530),
    'svhn': (0.5, 0.5, 0.5),
    'gtsrb': (0.2724, 0.2608, 0.2669),  # (0.5, 0.5, 0.5), #
    'subgtsrb': (0.5, 0.5, 0.5),  # (0.2724, 0.2608, 0.2669),#
    'pubfig': (1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0),  # (1.0, 1.0, 1.0), #
    'lisa': (0.2724, 0.2608, 0.2669),  #(0.2298, 0.2144, 0.2259),
    'eurosat': (0.2036, 0.1366, 0.1148),
     'food': (0.5, 0.5, 0.5),
     'pet': (0.5, 0.5, 0.5),
    'resisc': (0.5, 0.5, 0.5),
     'voc': (0.5, 0.5, 0.5),
    'lingspam': (0.5, 0.5, 0.5),
    'unknown': (0.5, 0.5, 0.5),
}
nlp_columns = {
    'sst2': ('sentence',  None),
    'mrpc': ('sentence1', 'sentence2'),
    'rte':  ('sentence1', 'sentence2'),
    'cola': ('sentence',  None),
}

num_class_map = {
    'cifar10': 10,
    'cifar100': 100,
    'subcifar': 2,
    'subgtsrb': 2,
    'svhn': 10,
    'pubfig': 83,
    'gtsrb': 43,
    'eurosat': 10,
    'flower': 102,
    'imagenet': 1000,
    'imagenet64': 1000,
    'mnist': 10,
    'mnistm': 10,
    'food': 101,
    'pet': 37,
    'resisc': 45,
    'voc':      21,
    'lingspam':  2,
    'sst2':      2,
    'mrpc':      2,
    'rte':       2,
    'cola':      2,
}
image_size_map = {
    'cifar10': 64,
    'cifar100': 224,
    'gtsrb': 64,
    'svhn': 64,
}


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
def set_seed(seed):
    if seed is None: return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class GTSRBDataset(Dataset):
    def __init__(self, root_dir, transform=None, download=False):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        if download:
            self.download()
        class_path = os.path.join(root_dir, 'GTSRB/Final_Training/Images')
        for class_id in os.listdir(class_path):
            class_dir = os.path.join(class_path, class_id)
            if os.path.isdir(class_dir):
                for image_name in os.listdir(class_dir):
                    if image_name.endswith('.ppm'):
                        self.images.append(os.path.join(class_dir, image_name))
                        self.labels.append(int(class_id))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        label = self.labels[idx]

        ppm_image = Image.open(img_path)
        rgb_image = ppm_image.convert('RGB')

        if self.transform:
            rgb_image = self.transform(rgb_image)

        return rgb_image, label

    def download(self):
        dataset_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
        zip_filename = os.path.join(self.root_dir, "GTSRB_Final_Training_Images.zip")

        # Download the dataset
        if os.path.exists(zip_filename):
            try:
                with zipfile.ZipFile(zip_filename, 'r') as _zf:
                    _zf.namelist()
                print("zip file is here, skip download")
            except zipfile.BadZipFile:
                print("zip file is corrupt, re-downloading...")
                os.remove(zip_filename)
                urllib.request.urlretrieve(dataset_url, zip_filename)
        else:
            urllib.request.urlretrieve(dataset_url, zip_filename)

        # Extract the dataset
        if os.path.exists(os.path.join(self.root_dir, "GTSRB")):
            print("GTSRB directory is here, skip unzip")
        else:
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(self.root_dir)
        print("GTSRB dataset is ready.")

class ImageNet64(data.Dataset):
    """
    ImageNet (downsampled) dataset.
    """

    def __init__(self, root, split='train', transform=None):
        super().__init__()
        self.transform = transform
        self.data = []
        self.labels = []
        if split == 'train':
            for i in range(1, 10):
                file_path = os.path.join(root, 'train_data_batch_{}'.format(i))
                dct = np.load(file_path, allow_pickle=True)
                self.data += list(dct['data'])
                self.labels += dct['labels']
        elif split == 'val':
            file_path = os.path.join(root, 'train_data_batch_10')
            dct = np.load(file_path, allow_pickle=True)
            self.data += list(dct['data'])
            self.labels += dct['labels']
        elif split == 'test':
            file_path = os.path.join(root, 'val_data')
            dct = np.load(file_path, allow_pickle=True)
            self.data += list(dct['data'])
            self.labels += dct['labels']
        else:
            raise NotImplementedError(
                '"split" must be "train" or "val" or "test".')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        img = img.reshape(3, 64, 64)  # [1, 12288] -> [3, 64, 64]
        img = img.transpose(1, 2, 0)
        label = self.labels[index] - 1  # [1, 1000]  -> [0, 999]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

class FMDataset(data.Dataset):
    """Feature Map dataset."""
    def __init__(self, root, split='train', transform=None, seeds = []):
        super().__init__()
        self.transform = transform
        self.data = []
        self.labels = []
        assert len(seeds) >= 1
        for i in seeds:
            file_path = os.path.join(root, f'{split}_data_batch_{str(i)}.npy')
            dct = np.load(file_path, allow_pickle=True)
            for idx, x in np.ndenumerate(dct):
                dict = x
            self.data += list(dict['data'])
            self.labels += list(dict['labels'])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]
        # img = img.reshape(3, 64, 64)  # [1, 12288] -> [3, 64, 64]
        # img = img.transpose(1, 2, 0)
        # label = self.labels[index] - 1  # [1, 1000]  -> [0, 999]
        # img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

def ImageNet64Loader(root, batch_size=256, num_workers=0, split='train', transform=None, shuffle=None):
    dataset = ImageNet64(root, split, transform)
    if shuffle is None: shuffle = (split == 'train')
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )

def ImageNetLoader(root, batch_size=256, num_workers=0, split='train', transform=None, shuffle=None):
    dataset = ImageNet(root, split, transform=transform)
    if shuffle is None: shuffle = (split == 'train')
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )

def GTSRBLoader(root, batch_size=256, num_workers=2, split='train', transform=None, shuffle=None):


    g = torch.Generator()
    g.manual_seed(0)

    set_seed(0)
    dataset = GTSRBDataset(root, transform, download=True)
    ppm_samples_count = len(dataset)
    train_size = 33200
    train_dataset, test_dataset = random_split(dataset, [train_size, ppm_samples_count - train_size])
    # dataset = GTSRB(root, split, transform, download=True)
    if shuffle is None: shuffle = (split == 'train')
    if split == 'train':
        dataset = train_dataset
    else:
        dataset = test_dataset
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
        worker_init_fn=seed_worker, # deterministic running
        generator=g,
    )

def CIFAR10Loader(root, batch_size=256, num_workers=0, split='train', transform=None, shuffle=None):
    if shuffle is None: shuffle = (split == 'train')
    split = True if split == 'train' else False
    dataset = CIFAR10(root, split, transform, download=True)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )

def EuroSatLoader(root, batch_size=256, num_workers=0, split='train', transform=None, shuffle=None):
    root = os.path.join(root, split)
    dataset = ImageFolder(root, transform)
    if shuffle is None: shuffle = (split == 'train')
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )

def SVHNLoader(root, batch_size=256, num_workers=0, split='train', transform=None, shuffle=None):
    dataset = SVHN(root, split, transform, download=True)
    if shuffle is None: shuffle = (split == 'train')
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )

def FlowerLoader(root, batch_size=256, num_workers=2, split='train', transform=None, shuffle=None):
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    if shuffle is None: shuffle = (split == 'train')
    split_reverse = 'train' if split == 'test' else 'test'
    dataset = Flowers102(root, split_reverse, transform, download=True)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )

def CIFAR100Loader(root, batch_size=256, num_workers=0, split='train', transform=None, shuffle=None):
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    if shuffle is None: shuffle = (split == 'train')
    split = True if split == 'train' else False
    print(os.path.join(os.getcwd(), root))
    dataset = CIFAR100(root, split, transform, download=True)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )

def RESISCLoader(root, batch_size=256, num_workers=0, split='train', transform=None, shuffle=None):
    dataset = ImageFolder(root=root, transform=transform)
    set_seed(0)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    if shuffle is None: shuffle = (split == 'trainval')

    return DataLoader(
        dataset=train_dataset if split =='train' else test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )

def PetLoader(root, batch_size=256, num_workers=0, split='train', transform=None, shuffle=None):
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    if split == 'train':
        split = 'trainval'
    else:
        split = 'test'
    dataset = OxfordIIITPet(root='data', split=split, transform=transform, download=True)
    if shuffle is None: shuffle = (split == 'trainval')
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )



def NLPLoader(
    dataset_name: str,
    tokenizer,
    max_length: int = 128,
    train_batch_size: int = 32,
    test_batch_size: int = 32,
    seed: int = 42,
):
    """
    Returns (train_loader, test_loader) for GLUE classification tasks.
    Each batch is a dict: {input_ids, attention_mask, labels} (all torch.Tensor).

    Supported: 'sst2' (sentiment, 2-cls), 'mrpc' (paraphrase, 2-cls), 'rte' (entailment, 2-cls).
    Requires: `datasets` and `transformers` from HuggingFace.
      #    │ Task │ Train │                                   Why                                   │
  ├────────┼──────┼───────┼─────────────────────────────────────────────────────────────────────────┤
  │ Easy   │ CoLA │ ~8.5K │ Single sentence, binary (grammatical/not) — simple input, clear label   │
  ├────────┼──────┼───────┼─────────────────────────────────────────────────────────────────────────┤
  │ Medium │ MRPC │ ~3.7K │ Sentence pair, paraphrase — requires semantic comparison                │
  ├────────┼──────┼───────┼─────────────────────────────────────────────────────────────────────────┤
  │ Hard   │ RTE  │ ~2.5K │ Sentence pair, entailment — requires reasoning, known to be challenging │

    """
    from datasets import load_dataset

    col1, col2 = nlp_columns[dataset_name]

    raw = load_dataset('glue', dataset_name, cache_dir=root_map[dataset_name])

    def _tokenize(examples):
        if col2 is not None:
            return tokenizer(
                examples[col1], examples[col2],
                truncation=True, max_length=max_length, padding='max_length',
            )
        return tokenizer(
            examples[col1],
            truncation=True, max_length=max_length, padding='max_length',
        )

    tokenized = raw.map(_tokenize, batched=True)

    # Keep only model inputs + label; rename label → labels
    keep_cols = {'input_ids', 'attention_mask', 'label'}
    for split in tokenized:
        drop = [c for c in tokenized[split].column_names if c not in keep_cols]
        tokenized[split] = tokenized[split].remove_columns(drop)
    tokenized = tokenized.rename_column('label', 'labels')
    tokenized.set_format('torch')

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        tokenized['train'],
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        generator=g,
    )
    val_split = 'validation' if 'validation' in tokenized else 'test'
    test_loader = DataLoader(
        tokenized[val_split],
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
    )
    print(
        f"NLPLoader [{dataset_name}]  "
        f"train: {train_batch_size}×{len(train_loader)}  "
        f"test: {test_batch_size}×{len(test_loader)}"
    )
    return train_loader, test_loader


loader_map = {
    'cifar10': CIFAR10Loader,
    'cifar100': CIFAR100Loader,
    'gtsrb': GTSRBLoader,
    'svhn': SVHNLoader,
    'eurosat': EuroSatLoader,
    'flower': FlowerLoader,
    'imagenet': ImageNetLoader,
    'imagenet64': ImageNet64Loader,
    'pet': PetLoader,
    'resisc': RESISCLoader,
}

class SingleLoader:
    def __init__(self, **kwargs):
        self.task = kwargs['task']
        self.num_class = num_class_map[self.task]
        self.train_batch_size = kwargs['train_batch_size']
        self.test_batch_size = kwargs['test_batch_size']
        self.mean = mean_map[self.task]
        self.std = mean_map[self.task]
        self.image_size = kwargs['image_size']
        self.device = kwargs['device']
        self.transform = {}
        if self.task not in ['cifar100']:
            self.init_transform()
        else:
            self.init_transform_pure()
        self.data_split = kwargs['data_split'] if 'data_split' in kwargs.keys() else ['train', 'test']
        self.init_loader()

    def init_loader(self):
        if 'test' in self.data_split:
            self.test_loader = loader_map[self.task](root_map[self.task], batch_size=self.test_batch_size, split='test',
                                                     transform=self.transform['test'])
        elif 'val' in self.data_split:
            self.test_loader = loader_map[self.task](root_map[self.task], batch_size=self.test_batch_size, split='val',
                                                     transform=self.transform['test'])
        if 'train' in self.data_split:
            self.train_loader = loader_map[self.task](root_map[self.task], batch_size=self.train_batch_size, split='train',
                                                  transform=self.transform['train'])
        print(f"build data loader for: {self.task}\n"
              f"test loader: {self.test_batch_size} * {len(self.test_loader)}\n"
              f"train loader: {self.train_batch_size} * {len(self.train_loader)}")

    def init_transform_pure(self):
        size = (self.image_size, self.image_size)
        normalize = transforms.Normalize(self.mean, self.std)
        self.transform['train'] = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size),
            normalize,
        ])
        self.transform['test'] = self.transform['train']


    def init_transform(self):
        size = (self.image_size, self.image_size)
        normalize = transforms.Normalize(mean_map[self.task], std_map[self.task])

        self.transform['test'] = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size, antialias=None),
            normalize,
        ])
        transform_list = []
        transform_list.append(transforms.Compose([]))
        transform_list.append(transforms.RandomResizedCrop(self.image_size))
        if self.task in ['gtsrb', 'svhn', 'lisa', 'mnist']:
            transform_list.append(transforms.RandomRotation(10))
        elif self.task in ['pubfig', 'flower', 'cifar100']:
            transform_list.append(transforms.RandomHorizontalFlip(1.))
            transform_list.append(transforms.RandomRotation(30))
        elif self.task in ['eurosat', 'cifar10', 'food', 'pet', 'resisc']:
            transform_list.append(transforms.RandomHorizontalFlip(1.))
            transform_list.append(transforms.RandomVerticalFlip(1.))
            transform_list.append(transforms.RandomRotation(30))

        self.random_choice = transforms.RandomChoice(transform_list)
        self.transform['train'] = transforms.Compose([
            self.random_choice,
            transforms.ToTensor(),
            transforms.Resize(size, antialias=None),
            normalize,
        ])

        if self.task == 'lisa':
            self.transform['test'] = transforms.Compose([
            transforms.Resize(size),
            normalize,
            ])
            self.transform['train'] = transforms.Compose([
                self.random_choice,
                transforms.Resize(size),
                normalize,
            ])
            self.transform['train_determ'] = transforms.Compose([
                transforms.Resize(size),
                normalize,
            ])
        elif self.task == 'mnist':
            self.transform['test'] = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                transforms.Resize(size),
                normalize,
            ])
            self.transform['train'] = transforms.Compose([
                self.random_choice,
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                transforms.Resize(size),
                normalize,
            ])
        print(f'image preprocessing finished')



