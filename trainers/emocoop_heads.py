import torch
import torch.nn as nn
import torch.nn.functional as F

class AuxiliaryEmotionHead(nn.Module):
    """辅助情绪分类头"""
    def __init__(self, input_dim, num_emotions, dropout_rate=0.3):
        super().__init__()
        self.num_emotions = num_emotions
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.emotion_classifier = nn.Linear(input_dim // 4, num_emotions)
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim // 4,
            num_heads=4,
            dropout=dropout_rate
        )
        self.use_attention = True
    def forward(self, image_features):
        projected_features = self.feature_projection(image_features)
        if self.use_attention:
            seq_features = projected_features.unsqueeze(0)
            attended_features, _ = self.attention(seq_features, seq_features, seq_features)
            attended_features = attended_features.squeeze(0)
        else:
            attended_features = projected_features
        emotion_logits = self.emotion_classifier(attended_features)
        return emotion_logits

class AdaptiveWeightScheduler:
    """自适应权重调整器，根据recall@2性能动态调整辅助损失权重"""
    def __init__(self, 
                 initial_weight=0.3, 
                 min_weight=0.2,  
                 max_weight=0.5,  
                 patience=8,      
                 improvement_threshold=0.01,  
                 decay_factor=0.95,  
                 growth_factor=1.05, 
                 min_epochs_before_adjust=3,  
                 strategy="conservative"):
        self.initial_weight = initial_weight
        self.current_weight = initial_weight
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.patience = patience
        self.improvement_threshold = improvement_threshold
        self.decay_factor = decay_factor
        self.growth_factor = growth_factor
        self.min_epochs_before_adjust = min_epochs_before_adjust
        self.strategy = strategy
        self.best_score = 0.0
        self.patience_counter = 0
        self.history = []
        self.consecutive_improvements = 0
        self.consecutive_declines = 0
        self.epoch_count = 0
        if self.strategy == "conservative":
            self.improvement_threshold = 0.015
            self.patience = 10
            self.decay_factor = 0.97
            self.growth_factor = 1.03
        elif self.strategy == "aggressive":
            self.improvement_threshold = 0.005
            self.patience = 5
            self.decay_factor = 0.9
            self.growth_factor = 1.1
    def update(self, current_score):
        self.epoch_count += 1
        self.history.append({
            'score': current_score,
            'weight': self.current_weight,
            'epoch': self.epoch_count
        })
        if self.epoch_count < self.min_epochs_before_adjust:
            print(f"Epoch {self.epoch_count}: 保持初始权重 {self.current_weight:.4f} (前{self.min_epochs_before_adjust}个epoch不调整)")
            return self.current_weight
        if current_score > self.best_score + self.improvement_threshold:
            old_weight = self.current_weight
            self.current_weight = min(self.current_weight * self.growth_factor, self.max_weight)
            self.best_score = current_score
            self.patience_counter = 0
            self.consecutive_improvements += 1
            self.consecutive_declines = 0
            print(f"Epoch {self.epoch_count}: 性能显著提升! {current_score:.4f} > {self.best_score - self.improvement_threshold:.4f}")
            print(f"  增加辅助损失权重: {old_weight:.4f} -> {self.current_weight:.4f} (连续改善: {self.consecutive_improvements})")
        elif current_score > self.best_score:
            self.best_score = current_score
            self.patience_counter = 0
            self.consecutive_improvements += 1
            self.consecutive_declines = 0
            print(f"Epoch {self.epoch_count}: 性能轻微改善，保持权重: {self.current_weight:.4f} (Score: {current_score:.4f})")
        else:
            self.patience_counter += 1
            self.consecutive_improvements = 0
            self.consecutive_declines += 1
            if self.patience_counter >= self.patience:
                old_weight = self.current_weight
                self.current_weight = max(self.current_weight * self.decay_factor, self.min_weight)
                self.patience_counter = 0
                print(f"Epoch {self.epoch_count}: 性能停滞{self.patience}个epoch!")
                print(f"  减少辅助损失权重: {old_weight:.4f} -> {self.current_weight:.4f} (连续下降: {self.consecutive_declines})")
            elif self.consecutive_declines >= 5:
                old_weight = self.current_weight
                self.current_weight = max(self.current_weight * 0.98, self.min_weight)
                self.consecutive_declines = 0
                print(f"Epoch {self.epoch_count}: 连续性能下降，轻微减少权重: {old_weight:.4f} -> {self.current_weight:.4f}")
            else:
                print(f"Epoch {self.epoch_count}: 性能未改善，耐心等待 ({self.patience_counter}/{self.patience}) (Score: {current_score:.4f})")
        return self.current_weight
    def get_weight(self):
        return self.current_weight
    def reset(self):
        self.current_weight = self.initial_weight
        self.best_score = 0.0
        self.patience_counter = 0
        self.consecutive_improvements = 0
        self.consecutive_declines = 0
        self.history = []
    def get_history(self):
        return self.history
