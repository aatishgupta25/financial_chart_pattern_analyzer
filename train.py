import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import ChartDataset
from model import ChartSenseVLM
from tqdm import tqdm
import os


CHART_DIR = 'data/charts'
LABEL_DIR = 'data/labels'

# A small batch size is selected due to memory constraints; adjustments may be needed for Out-Of-Memory errors.
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
SAVE_MODEL_PATH = 'chart_sense_vlm.pt'
TEST_SPLIT_RATIO = 0.2 # 20% of the data is allocated for testing/validation.


# The appropriate computational device (MPS, CUDA, or CPU) is determined and printed.
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device (Apple Silicon GPU).")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device (NVIDIA GPU).")
else:
    device = torch.device("cpu")
    print("Using CPU device.")


# Image transformations are applied, consistent with the requirements for the Vision Transformer model.
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # Images are resized to 224x224 pixels, as expected by ViT-B-16.
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet-specific normalization is applied.
])

def train_model():
    """
    Orchestrates the training and evaluation process for the ChartSenseVLM.
    """
    
    full_dataset = ChartDataset(CHART_DIR, LABEL_DIR, transform=data_transforms)
    
    if len(full_dataset) == 0:
        print("Error: Dataset is empty. It is ensured that data generation was successful.")
        return

    # The dataset is split into training and testing sets based on the defined ratio.
    test_size = int(TEST_SPLIT_RATIO * len(full_dataset))
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # DataLoaders are created for efficient batch processing during training and evaluation.
    # Number of workers is set to half of CPU cores to prevent contention.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2 or 1)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2 or 1)

    # The vocabulary and index-to-token mapping are retrieved from the dataset.
    vocab = full_dataset.vocab
    idx_to_token = full_dataset.idx_to_token

    
    # The ChartSenseVLM is instantiated and moved to the selected device.
    model = ChartSenseVLM(vocab=vocab).to(device)

    # Encoder layers are frozen, which is a common practice for fine-tuning pre-trained models on small datasets.
    # This prevents extensive updates to the large ViT weights, focusing training on the decoder.
    for param in model.encoder.parameters():
        param.requires_grad = False

    # CrossEntropyLoss is used for sequence prediction, with padding tokens ignored during loss calculation.
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    # The Adam optimizer is configured to update only parameters that require gradients.
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    
    print("\n--- Starting Training ---")
    best_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train() # The model is set to training mode.
        running_loss = 0.0
        # tqdm is utilized to display a progress bar for the training epoch.
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)", leave=False)

        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() # Gradients are zeroed before backpropagation.

            # The model performs a forward pass to predict logits. The decoder expects input tokens
            # (labels without the <eos> token) and predicts the subsequent tokens.
            logits = model(images, target_sequences=labels)

            # Logits and labels are reshaped to fit CrossEntropyLoss requirements.
            # The target labels for loss calculation exclude the <sos> token.
            target_labels_for_loss = labels[:, 1:]
            
            loss = criterion(logits.reshape(-1, logits.shape[-1]), target_labels_for_loss.reshape(-1))

            loss.backward() # Backpropagation is performed.
            optimizer.step() # Model weights are updated.

            running_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)

        
        model.eval() # The model is set to evaluation mode.
        val_running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        val_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Val)", leave=False)
        with torch.no_grad(): # Gradient calculations are disabled during evaluation for efficiency.
            for images, labels in val_bar:
                images = images.to(device)
                labels = labels.to(device)

                # For validation loss, predicted logits are compared with ground truth.
                # The model generates tokens up to MAX_INFERENCE_LEN.
                MAX_INFERENCE_LEN = labels.size(1)

                # Logits from inference are obtained.
                logits = model(images, max_len=MAX_INFERENCE_LEN)

                # Targets for CrossEntropyLoss exclude the <sos> token.
                target_labels_for_loss = labels[:, 1:]

                # Logits are sliced to match the length of target_labels_for_loss, ensuring
                # that predictions for relevant parts of the sequence are compared.
                val_logits_for_loss = logits[:, :target_labels_for_loss.size(1), :]
                
                val_loss = criterion(val_logits_for_loss.reshape(-1, val_logits_for_loss.shape[-1]), target_labels_for_loss.reshape(-1))
                val_running_loss += val_loss.item()

                # Accuracy is calculated based on simple token comparison.
                predicted_tokens = logits.argmax(-1)
                
                # Predicted tokens are sliced to match the length of the target labels for accurate comparison.
                predicted_tokens_for_accuracy = predicted_tokens[:, :target_labels_for_loss.size(1)]
                
                mask = (target_labels_for_loss != vocab['<pad>']) # A mask is applied for valid tokens.
                correct_predictions += (predicted_tokens_for_accuracy == target_labels_for_loss)[mask].sum().item()
                total_predictions += mask.sum().item()

                val_bar.set_postfix(val_loss=f"{val_loss.item():.4f}")

        avg_val_loss = val_running_loss / len(test_loader)
        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Accuracy = {accuracy:.2f}%")

        
        # The model's state dictionary is saved if the current validation loss is an improvement over the best recorded loss.
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"Model saved to {SAVE_MODEL_PATH} with validation loss: {best_loss:.4f}")

    print("\n--- Training Complete ---")

# The training process is initiated when the script is executed directly.
if __name__ == "__main__":
    train_model()