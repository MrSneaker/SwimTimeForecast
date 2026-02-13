import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedLinearUnit(nn.Module):
    """GLU: Contrôle le flux d'information."""
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, input_dim * 2)

    def forward(self, x):
        val, gate = self.fc(x).chunk(2, dim=-1)
        return val * torch.sigmoid(gate)

class GatedResidualNetwork(nn.Module):
    """GRN: Bloc de base non-linéaire avec skip connection."""
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Projection si les dimensions entrée/sortie diffèrent
        self.project = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

    def forward(self, x):
        residual = self.project(x)
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.glu(x)
        return self.norm(residual + x)

class VariableSelectionNetwork(nn.Module):
    """
    C'est ici que réside l'EXPLICABILITÉ.
    Le réseau apprend un poids pour chaque feature à chaque pas de temps.
    """
    def __init__(self, num_features, hidden_dim, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        
        # Un GRN par variable pour la transformation individuelle
        self.single_variable_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_dim, dropout) for _ in range(num_features)
        ])
        
        # GRN pour calculer les poids d'importance
        # On concatène toutes les features transformées pour décider des poids
        self.weight_grn = GatedResidualNetwork(num_features * hidden_dim, num_features, dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [batch, seq_len, num_features]
        batch_size, seq_len, _ = x.shape
        
        # 1. Transformation individuelle de chaque feature
        # On passe chaque feature (dimension 1) dans son propre GRN
        # flat_x : liste de [batch, seq_len, hidden_dim]
        processed_features = []
        for i in range(self.num_features):
            feat = x[..., i:i+1] # Garde la dim: [batch, seq, 1]
            processed_features.append(self.single_variable_grns[i](feat))
            
        # Stack: [batch, seq_len, num_features, hidden_dim]
        processed_stack = torch.stack(processed_features, dim=2)
        
        # 2. Calcul des poids d'importance
        # On aplatit pour le GRN de poids: [batch, seq_len, num_features * hidden_dim]
        flattened = processed_stack.view(batch_size, seq_len, -1)
        weights = self.weight_grn(flattened) # [batch, seq_len, num_features]
        weights = self.softmax(weights)
        
        # 3. Somme pondérée
        # weigths: [batch, seq, num_features, 1]
        weights_expanded = weights.unsqueeze(-1)
        # weighted_sum: [batch, seq, hidden_dim]
        combined = (processed_stack * weights_expanded).sum(dim=2)
        
        return combined, weights

# =============================================================
# Model V3: SwimTFT (Compact)
# =============================================================
class SwimTFT(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2, n_heads=4):
        super().__init__()
        
        # Attention: hidden_dim réduit par défaut (64) pour le GPU 3GB
        self.hidden_dim = hidden_dim
        
        # 1. Variable Selection (Transforme inputs -> hidden_dim tout en sélectionnant)
        self.vsn = VariableSelectionNetwork(input_dim, hidden_dim, dropout)
        
        # 2. Encodage Séquentiel (LSTM)
        # Le TFT officiel utilise un LSTM pour encoder le passé
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 3. Gating post-LSTM
        self.post_lstm_gate = GatedResidualNetwork(hidden_dim, hidden_dim, dropout)
        
        # 4. Multi-Head Attention (Interpretable Multi-Head Attention)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.post_attn_gate = GatedResidualNetwork(hidden_dim, hidden_dim, dropout)
        
        # 5. Output layers (Quantiles)
        self.fc = nn.Linear(hidden_dim, 9)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        
        # A. Feature Selection
        # x_embed: [batch, seq, hidden_dim]
        # feature_weights: [batch, seq, input_dim] <- l'explicabilité
        x_embed, feature_weights = self.vsn(x)
        
        # B. LSTM Encoding (Locality)
        lstm_out, _ = self.lstm(x_embed)
        lstm_out = self.post_lstm_gate(lstm_out + x_embed) # Residual + Gate
        
        # C. Attention (Long term dependency)
        # Self-attention sur toute la séquence
        attn_out, attn_weights = self.attn(lstm_out, lstm_out, lstm_out)
        attn_out = self.post_attn_gate(attn_out + lstm_out) # Residual + Gate
        
        # D. Output
        # On prend le dernier état pour la prédiction
        final_state = attn_out[:, -1, :]
        prediction = self.fc(final_state)
        
        prediction = prediction.view(-1, 3, 3) # Reshape pour les 3 horizons et 3 quantiles
        
        # On retourne aussi les poids si besoin d'analyse
        return prediction, feature_weights, attn_weights