import torch
from torch import nn
import torch.nn.functional as F


class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(torch.nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class ResidualInceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[1,3], dropout=0.05):
        super(ResidualInceptionBlock, self).__init__()

        self.out_channels = out_channels
        num_branches = len(kernel_sizes)
        branch_out_channels = out_channels // num_branches

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, in_channels, kernel_size=1),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(),
                nn.Conv1d(in_channels, branch_out_channels, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(branch_out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for k in kernel_sizes
        ])

        self.residual_adjust = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        concatenated = torch.cat(branch_outputs, dim=1)
        residual = self.residual_adjust(x)
        output = self.relu(concatenated + residual)
        return output

class ResidualMLPBlock(nn.Module):
    """FC → LayerNorm → Mish + skip connection"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            Mish(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.block(x)



class Integration(nn.Module):
    def __init__(self, drug_model, protein_model, drug_out_dim=384, protein_dim=1024,
                 proj_dim=1024, hidden_sizes=[1024, 512, 256, 1], dropout=0.1):
        super().__init__()

        self.drug_model = drug_model
        self.protein_model = protein_model

        # # Drug projection: 차원 불균형 완화 (4096 → 1024)
        # self.drug_resnet = nn.Sequential(
        #     ResidualMLPBlock(drug_out_dim, dropout=dropout),
        #     ResidualMLPBlock(drug_out_dim, dropout=dropout),
        #     ResidualMLPBlock(drug_out_dim, dropout=dropout),
        # )

        # # Protein residual blocks (3층)
        # self.protein_resnet = nn.Sequential(
        #     ResidualMLPBlock(protein_dim, dropout=dropout),
        #     ResidualMLPBlock(protein_dim, dropout=dropout),
        #     ResidualMLPBlock(protein_dim, dropout=dropout),
        # )

        combined_dim = drug_out_dim + protein_dim 

        # AffinityPredictor 스타일: ResidualInceptionBlock × 2
        self.inc1 = ResidualInceptionBlock(combined_dim, combined_dim, dropout=dropout)
        self.inc2 = ResidualInceptionBlock(combined_dim, combined_dim, dropout=dropout)

        # self.inc_mean1 = ResidualInceptionBlock(combined_dim, combined_dim, dropout=dropout)
        # self.inc_mean2 = ResidualInceptionBlock(combined_dim, combined_dim, dropout=dropout)

        # self.inc_max1 = ResidualInceptionBlock(combined_dim, combined_dim, dropout=dropout)
        # self.inc_max2 = ResidualInceptionBlock(combined_dim, combined_dim, dropout=dropout) 

        # 동적 regressor (hidden_sizes 마지막 원소는 1이어야 함)
        layers = []
        input_dim = combined_dim * 2
        for output_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, output_dim))
            if output_dim != 1:
                layers.append(Mish())
            input_dim = output_dim
        self.regressor = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():

            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, drug_emb, fasta_emb, fasta_max_emb):
        # v_drug    = self.drug_resnet(self.drug_model(drug_emb))       # (B, 4096) → (B, proj_dim)
        # v_protein = self.protein_resnet(self.protein_model(fasta_emb))  # (B, protein_dim)
        concat    = torch.cat((fasta_emb,drug_emb), dim=-1)          # (B, combined_dim)
        concat2    = torch.cat((fasta_max_emb,drug_emb), dim=-1)          # (B, combined_dim)  
        # Conv1d 기반 Inception 블록을 위해 (B, D, 1)로 확장
        x = concat.unsqueeze(2)   # (B, combined_dim, 1)
        x2 = concat2.unsqueeze(2)   # (B, combined_dim, 1)
        x = self.inc1(x)
        #x = self.inc2(x)
        x = x.squeeze(2)          # (B, combined_dim)
        #x2 = self.inc1(x2)
        x2 = self.inc2(x2)
        x2 = x2.squeeze(2)          # (B, combined_dim)
        concat3 = torch.cat((x,x2), dim=-1)          # (B, combined_dim)

        return self.regressor(self.dropout(concat3))              # (B, 1)

        