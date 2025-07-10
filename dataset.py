# dataset.py

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ChartDataset(Dataset):
    def __init__(self, chart_dir: str, label_dir: str, transform=None):
        """
        Initializes the ChartDataset.

        Parameters:
            chart_dir (str): Directory containing chart images.
            label_dir (str): Directory containing pattern label text files.
            transform (callable, optional): An optional transform to be applied to an image.
        """
        self.chart_dir = chart_dir
        self.label_dir = label_dir
        self.transform = transform

        self.image_files = sorted([f for f in os.listdir(chart_dir) if f.endswith('.png')])
        self.labels = {}

        # All labels are loaded into memory; suitable for datasets of thousands
        for filename in os.listdir(label_dir):
            if filename.endswith('.txt'):
                base_name = filename.replace('.txt', '')
                with open(os.path.join(label_dir, filename), 'r') as f:
                    self.labels[base_name] = f.read().strip()

        # Pairs of (image_file_path, label_text) are created for direct indexed access
        self.data_pairs = []
        for img_filename in self.image_files:
            base_name = img_filename.replace('.png', '')
            if base_name in self.labels:
                self.data_pairs.append((os.path.join(chart_dir, img_filename), self.labels[base_name]))
            else:
                print(f"Warning: No label found for image {img_filename}. Skipping.")

        # A vocabulary for pattern labels is defined, including special tokens
        # This vocabulary should encompass all possible pattern labels returned by the detection logic
        self.unique_labels = sorted(list(set(self.labels.values())))
        self.vocab = {
            '<pad>': 0,
            '<unk>': 1,
            '<sos>': 2,
            '<eos>': 3,
        }
        for i, label in enumerate(self.unique_labels):
            self.vocab[label] = i + 4

        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path, label_text = self.data_pairs[idx]

        # The image is loaded and converted to RGB to ensure 3 channels.
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Label text is tokenized. For simple, single-word labels, they are treated as single tokens
        # Special <sos> and <eos> tokens are added for the decoder
        label_tokens = [self.vocab['<sos>']] + [self.vocab.get(label_text, self.vocab['<unk>'])] + [self.vocab['<eos>']]
        label_tensor = torch.tensor(label_tokens, dtype=torch.long)

        return image, label_tensor

# Example usage block for self-testing
if __name__ == "__main__":
    # Image transformations are defined for consistency with model input requirements
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)), # Resizing to 224x224 pixels is required by ViT-B-16
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet-specific normalization is applied
    ])

    CHART_DIR = 'data/charts'
    LABEL_DIR = 'data/labels'

    chart_dataset = ChartDataset(CHART_DIR, LABEL_DIR, transform=data_transforms)

    print(f"Number of samples in dataset: {len(chart_dataset)}")
    print(f"Unique patterns detected: {chart_dataset.unique_labels}")
    print(f"Vocabulary: {chart_dataset.vocab}")

    # A sample from the dataset is accessed for verification
    if len(chart_dataset) > 0:
        img_tensor, label_tensor = chart_dataset[0]
        print(f"\nShape of image tensor: {img_tensor.shape}") # Expected shape is [C, H, W], e.g., [3, 224, 224]
        print(f"Label tensor: {label_tensor}")
        # The label tensor is converted back to text for verification purposes
        decoded_label = [chart_dataset.idx_to_token[idx.item()] for idx in label_tensor]
        print(f"Decoded label: {decoded_label}")
    else:
        print("No samples in the dataset to test.")