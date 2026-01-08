import timm
import torch
import torch.nn as nn


class AttentionFuse(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        attn = torch.softmax((self.q(x) @ self.k(x).transpose(-2,-1)) * self.scale, dim=-1)
        return (attn @ self.v(x)).mean(dim=1)

class HybridMRIModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # CNN backbone
        self.cnn = timm.create_model("resnet18", pretrained=True, num_classes=0)
        cnn_dim = self.cnn.num_features

        # ViT backbone
        self.vit = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0)
        vit_dim = self.vit.num_features

        # Project to common fusion dimension
        fusion_dim = 256
        self.cnn_proj = nn.Linear(cnn_dim, fusion_dim)
        self.vit_proj = nn.Linear(vit_dim, fusion_dim)

        # RNN
        self.rnn = nn.LSTM(fusion_dim, 128, batch_first=True, bidirectional=True)
        rnn_dim = 256

        # Attention
        self.attn = AttentionFuse(rnn_dim)

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(rnn_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        B = x.size(0)

        cnn_f = self.cnn_proj(self.cnn(x))
        vit_f = self.vit_proj(self.vit(x))

        fused = cnn_f + vit_f
        fused = fused.unsqueeze(1)

        rnn_out,_ = self.rnn(fused)
        attn_out = self.attn(rnn_out)

        return self.fc(attn_out)
