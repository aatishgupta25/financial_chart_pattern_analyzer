# model.py 

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict

class ImageEncoder(nn.Module):
    """
    Encodes an image using a pre-trained Vision Transformer (ViT).
    """
    def __init__(self):
        super().__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_head = self.vit.heads
        self.vit.heads = nn.Identity()
        features = self.vit(x)
        self.vit.heads = original_head
        return features


class TextDecoder(nn.Module):
    """
    Decodes a sequence of tokens from image features using a GRU.
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, encoder_output_dim: int, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim + encoder_output_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.encoder_output_dim = encoder_output_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if encoder_output_dim != hidden_size:
            self.initial_hidden_projection = nn.Linear(encoder_output_dim, hidden_size)
        else:
            self.initial_hidden_projection = nn.Identity() 

    def forward(self, features: torch.Tensor, target_sequence: torch.Tensor = None, max_len: int = None, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        batch_size = features.size(0)

        projected_features = self.initial_hidden_projection(features)
        hidden = projected_features.unsqueeze(0).repeat(self.num_layers, 1, 1)


        outputs = []
        if self.training and target_sequence is not None:
            input_tokens = target_sequence[:, :-1]
            embedded = self.embedding(input_tokens)

            expanded_features = features.unsqueeze(1).expand(-1, embedded.size(1), -1)

            gru_input = torch.cat((embedded, expanded_features), dim=2)

            gru_output, _ = self.gru(gru_input, hidden)
            logits = self.fc(gru_output)
            return logits

        else: 
            input_token = torch.tensor([2] * batch_size, dtype=torch.long, device=features.device).unsqueeze(1) 

            for _ in range(max_len):
                embedded = self.embedding(input_token)
                expanded_features = features.unsqueeze(1)
                gru_input = torch.cat((embedded, expanded_features), dim=2)

                gru_output, hidden = self.gru(gru_input, hidden)
                logits = self.fc(gru_output.squeeze(1))

                outputs.append(logits)
                predicted_token = logits.argmax(1)

                input_token = predicted_token.unsqueeze(1)
                if (predicted_token == self.vocab_size - 1).all(): 
                    break
            return torch.stack(outputs, dim=1)


class ChartSenseVLM(nn.Module):
    """
    Combines the ImageEncoder and TextDecoder into a single Vision-Language Model.
    """
    def __init__(self, vocab: Dict[str, int], embed_dim: int = 256, hidden_size: int = 512, decoder_num_layers: int = 1):
        super().__init__()
        self.encoder = ImageEncoder()
        encoder_output_dim = 768
        self.decoder = TextDecoder(
            vocab_size=len(vocab),
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            encoder_output_dim=encoder_output_dim,
            num_layers=decoder_num_layers
        )
        self.vocab = vocab

    def forward(self, images: torch.Tensor, target_sequences: torch.Tensor = None, max_len: int = None, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        features = self.encoder(images)
        logits = self.decoder(features, target_sequences, max_len, teacher_forcing_ratio)
        return logits

if __name__ == "__main__":
    dummy_vocab = {
        '<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3,
        'doji': 4, 'hammer': 5, 'bullish engulfing': 6, 'shooting star': 7
    }
    VOCAB_SIZE = len(dummy_vocab)
    EMBED_DIM = 256
    HIDDEN_SIZE = 512

    model = ChartSenseVLM(dummy_vocab, EMBED_DIM, HIDDEN_SIZE)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    print("Model architecture created:")
    print(model)

    BATCH_SIZE = 4
    dummy_images = torch.randn(BATCH_SIZE, 3, 224, 224).to(device) 
    dummy_target_sequences = torch.tensor([
        [2, 4, 3], # doji
        [2, 5, 3], # hammer
        [2, 6, 3], # bullish engulfing
        [2, 4, 3]  # doji
    ], dtype=torch.long).to(device) 

    model.train()
    logits_train = model(dummy_images, dummy_target_sequences)
    print(f"\nLogits shape (training): {logits_train.shape}")

    model.eval()
    MAX_DECODE_LEN = 5
    with torch.no_grad():
        logits_eval = model(dummy_images, max_len=MAX_DECODE_LEN)
    print(f"Logits shape (inference): {logits_eval.shape}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f} M")