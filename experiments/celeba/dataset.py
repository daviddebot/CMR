import os.path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA
from pathlib import Path
import torchvision.models as models
import torch.nn.functional as F

import lightning.pytorch as pl
import pytorch_lightning as pl2

pl2.seed_everything(42)


class CelebADataset(CelebA):
    def __init__(self, root, split='train', transform=None, download=False, class_attributes=None):
        super(CelebADataset, self).__init__(root, split=split, target_type="attr", transform=transform, download=download)

        # Set the class attributes
        if class_attributes is None:
            # Default to 'Attractive' if no class_attributes provided
            self.class_idx = [self.attr_names.index('Attractive')]
        else:
            # Use the provided class attributes
            self.class_idx = [self.attr_names.index(attr) for attr in class_attributes]

        self.attr_names = [string for string in self.attr_names if string]

        # Determine concept and task attribute names based on class attributes
        self.concept_attr_names = [attr for i, attr in enumerate(self.attr_names) if i not in self.class_idx]
        self.task_attr_names = [self.attr_names[i] for i in self.class_idx]

    def __getitem__(self, index):
        image, attributes = super(CelebADataset, self).__getitem__(index)

        # Extract the target (y) based on the class index
        y = torch.stack([attributes[i] for i in self.class_idx])

        # Extract concept attributes, excluding the class attributes
        concept_attributes = torch.stack([attributes[i] for i in range(len(attributes)) if i not in self.class_idx])

        return image, concept_attributes, y


class ResNetEmbedding(torch.nn.Module):
    def __init__(self, original_model):
        super(ResNetEmbedding, self).__init__()
        self.features = torch.nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


def extract_and_save_embeddings(dataset_root, class_attributes=None, split='test', download=False, batch_size=32):
    # Load pre-trained ResNet model and modify it
    resnet_model = models.resnet18(pretrained=True)
    model = ResNetEmbedding(resnet_model)
    model.eval()

    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load CelebA dataset
    celeba_dataset = CelebADataset(root=dataset_root, split=split, transform=transform, download=download,
                                   class_attributes=class_attributes)
    data_loader = DataLoader(celeba_dataset, batch_size=batch_size, shuffle=False)

    # Extract embeddings
    embeddings, c, y = [], [], []
    with torch.no_grad():
        for batch_idx, (images, concepts, tasks) in enumerate(data_loader):
            print(f"Processing batch {batch_idx + 1}/{len(data_loader)}...")
            emb = model(images)
            embeddings.append(emb)
            c.append(concepts)
            y.append(tasks)

    # Concatenate and save embeddings
    embeddings = torch.cat(embeddings, dim=0)
    c = torch.cat(c, dim=0)
    y = torch.cat(y, dim=0)
    torch.save(embeddings, os.path.join(dataset_root, 'celeba', f'{split}_embeddings.pt'))
    torch.save(c, os.path.join(dataset_root, 'celeba', f'{split}_concepts.pt'))
    torch.save(y, os.path.join(dataset_root, 'celeba', f'{split}_tasks.pt'))
    torch.save(celeba_dataset.concept_attr_names, os.path.join(dataset_root, 'celeba', f'{split}_concept_names.pt'))
    torch.save(celeba_dataset.task_attr_names, os.path.join(dataset_root, 'celeba', f'{split}_task_names.pt'))


def load_celeba_data(dataset_root, split):
    embeddings_path = os.path.join(dataset_root, 'celeba', f'{split}_embeddings.pt')
    concepts_path = os.path.join(dataset_root, 'celeba', f'{split}_concepts.pt')
    tasks_path = os.path.join(dataset_root, 'celeba', f'{split}_tasks.pt')
    concept_names_path = os.path.join(dataset_root, 'celeba', f'{split}_concept_names.pt')
    task_names_path = os.path.join(dataset_root, 'celeba', f'{split}_task_names.pt')

    embeddings = torch.load(embeddings_path)
    concepts = torch.load(concepts_path)
    tasks = torch.load(tasks_path)
    concept_names = torch.load(concept_names_path)
    task_names = torch.load(task_names_path)

    concepts = concepts.float()
    if len(tasks.shape) == 1:
        tasks = tasks.unsqueeze(1)
    tasks = tasks.float()
    return embeddings, concepts, tasks, concept_names, task_names


if __name__ == '__main__':
    dataset_root = Path('../data')
    class_attributes = ['Attractive', 'Male', 'Young']
    extract_and_save_embeddings(dataset_root, class_attributes, split='test', download=False, batch_size=512)
