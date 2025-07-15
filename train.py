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

BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
SAVE_MODEL_PATH = 'chart_sense_vlm.pt'
TEST_SPLIT_RATIO = 0.2 

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device (Apple Silicon GPU).")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device (NVIDIA GPU).")
else:
    device = torch.device("cpu")
    print("Using CPU device.")

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def train_model():
    """
    Orchestrates the training and evaluation process for the ChartSenseVLM.
    """
    
    full_dataset = ChartDataset(CHART_DIR, LABEL_DIR, transform=data_transforms)
    
    if len(full_dataset) == 0:
        print("Error: Dataset is empty. It is ensured that data generation was successful.")
        return

    test_size = int(TEST_SPLIT_RATIO * len(full_dataset))
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2 or 1)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2 or 1)

    vocab = full_dataset.vocab
    idx_to_token = full_dataset.idx_to_token

    model = ChartSenseVLM(vocab=vocab).to(device)

    for param in model.encoder.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    
    print("\n--- Starting Training ---")
    best_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train() 
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)", leave=False)

        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images, target_sequences=labels)

            target_labels_for_loss = labels[:, 1:]
            
            loss = criterion(logits.reshape(-1, logits.shape[-1]), target_labels_for_loss.reshape(-1))

            loss.backward() 
            optimizer.step() 

            running_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)

        
        model.eval() 
        val_running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        val_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Val)", leave=False)
        with torch.no_grad():
            for images, labels in val_bar:
                images = images.to(device)
                labels = labels.to(device)

                MAX_INFERENCE_LEN = labels.size(1)

                logits = model(images, max_len=MAX_INFERENCE_LEN)

                target_labels_for_loss = labels[:, 1:]

                val_logits_for_loss = logits[:, :target_labels_for_loss.size(1), :]
                
                val_loss = criterion(val_logits_for_loss.reshape(-1, val_logits_for_loss.shape[-1]), target_labels_for_loss.reshape(-1))
                val_running_loss += val_loss.item()

                predicted_tokens = logits.argmax(-1)
                
                predicted_tokens_for_accuracy = predicted_tokens[:, :target_labels_for_loss.size(1)]
                
                mask = (target_labels_for_loss != vocab['<pad>']) 
                correct_predictions += (predicted_tokens_for_accuracy == target_labels_for_loss)[mask].sum().item()
                total_predictions += mask.sum().item()

                val_bar.set_postfix(val_loss=f"{val_loss.item():.4f}")

        avg_val_loss = val_running_loss / len(test_loader)
        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Accuracy = {accuracy:.2f}%")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"Model saved to {SAVE_MODEL_PATH} with validation loss: {best_loss:.4f}")

    print("\n--- Training Complete ---")

if __name__ == "__main__":
    train_model()