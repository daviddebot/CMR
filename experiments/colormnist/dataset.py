import os.path
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from tqdm import tqdm
import random


# Helper function to convert grayscale image to red or green
def colorize(image, color):
    colored_image = torch.zeros(3, 28, 28)  # Create an image with 3 channels (RGB)
    if color == 'red':
        colored_image[0] = image  # Red channel
    elif color == 'green':
        colored_image[1] = image  # Green channel
    return colored_image


# Creating the custom dataset
class CustomMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, mnist_dataset, random=True):
        self.mnist_dataset = mnist_dataset
        self.random = random

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        image, digit = self.mnist_dataset[idx]

        # Colorize the image
        if self.random:
            color = 'red' if random.random() < 0.5 else 'green'
        else:
            color = 'red' if digit <= 5 else 'green'
        colored_image = colorize(image.squeeze(), color)  # Remove channel dimension of the grayscale image

        # Create the concept label
        concept_label = np.zeros(12)  # 10 digits + 2 colors
        concept_label[digit] = 1
        concept_label[10] = 1 if color == 'red' else 0
        concept_label[11] = 1 if color == 'green' else 0

        # Create the target label
        target_label = 1 if digit % 2 == 0 else 0
        target_label = [target_label, 1 - target_label]

        return colored_image, torch.tensor(concept_label, dtype=torch.float32), torch.tensor(target_label, dtype=torch.float32)


def load_preprocessed_data(base_dir='./experiments/colored_mnist'):
    train_features = torch.from_numpy(np.load(os.path.join(base_dir, 'train_features.npy')))
    train_concepts = torch.from_numpy(np.load(os.path.join(base_dir, 'train_concepts.npy')))
    train_tasks = torch.from_numpy(np.load(os.path.join(base_dir, 'train_tasks.npy'))).unsqueeze(1)
    test_features = torch.from_numpy(np.load(os.path.join(base_dir, 'test_features.npy')))
    test_concepts = torch.from_numpy(np.load(os.path.join(base_dir, 'test_concepts.npy')))
    test_tasks = torch.from_numpy(np.load(os.path.join(base_dir, 'test_tasks.npy'))).unsqueeze(1)
    return train_features, train_concepts, train_tasks, test_features, test_concepts, test_tasks


def main():
    # Step 1: Prepare the MNIST dataset

    # Load MNIST data
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    # Create custom datasets
    custom_train_dataset = CustomMNISTDataset(mnist_train)
    custom_test_dataset = CustomMNISTDataset(mnist_test)

    # DataLoaders
    train_loader = DataLoader(custom_train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(custom_test_dataset, batch_size=64, shuffle=False)

    # Step 2: Prepare ResNet18 model for feature extraction
    model = models.resnet18(pretrained=True)
    model.eval()  # Set the model to evaluation mode

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Step 3: Extract features
    def extract_features(data_loader):
        features = []
        concept_labels = []
        task_labels = []

        with torch.no_grad():
            for imgs, concepts, tasks in tqdm(data_loader):
                imgs = imgs.to(device)
                out = model(imgs)
                features.append(out.cpu().numpy())
                concept_labels.append(concepts.numpy())
                task_labels.append(tasks.numpy())

        return np.concatenate(features), np.concatenate(concept_labels), np.concatenate(task_labels)

    train_features, train_concepts, train_tasks = extract_features(train_loader)
    test_features, test_concepts, test_tasks = extract_features(test_loader)

    # Step 4: Save the embeddings and labels
    np.save('train_features.npy', train_features)
    np.save('train_concepts.npy', train_concepts)
    np.save('train_tasks.npy', train_tasks)

    np.save('test_features.npy', test_features)
    np.save('test_concepts.npy', test_concepts)
    np.save('test_tasks.npy', test_tasks)


if __name__ == "__main__":
    main()
