import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path as osp
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import torchvision.transforms as T
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

from .visual_components import (
    ColorToneAnalyzer,
    ModifiedVisualTransformer,
    VisualPromptLearner,
    VisualClusterInitializer
)
from .text_components import (
    GloVeInitializer,
    TextPromptLearner
)
from .data_augmentation import get_train_transforms, get_val_transforms
from .emocoop_utils import load_clip_to_cpu, optimize_cfg_for_memory, forward_with_debug
from .emocoop_heads import AuxiliaryEmotionHead, AdaptiveWeightScheduler

class CustomCLIP(nn.Module):
    """增强的CustomCLIP - 添加辅助情绪分类头"""
    def __init__(self, cfg, classnames, clip_model, dataloader=None, device='cuda'):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # 使用增强的prompt learner
        self.prompt_learner = TextPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.logit_scale = clip_model.logit_scale

        self.visual_prompt_learner = VisualPromptLearner(
            cfg, clip_model, dataloader, device
        )
        self.image_encoder = ModifiedVisualTransformer(clip_model.visual, self.visual_prompt_learner)
        
        # 文本编码器组件
        self.text_encoder = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.token_embedding = clip_model.token_embedding
        self.dtype = clip_model.dtype
        
        # 辅助情绪分类头
        self.auxiliary_emotion_head = AuxiliaryEmotionHead(
            input_dim=clip_model.visual.output_dim,
            num_emotions=len(classnames),  # 假设类别数等于情绪数
            dropout_rate=0.3
        )
        
        # 损失权重 - 现在可以通过set_aux_loss_weight动态设置
        self.aux_loss_weight = getattr(cfg.TRAINER.EMOCOOP, 'AUX_LOSS_WEIGHT', 0.3)
        
        print(f"Enhanced CustomCLIP initialized:")
        print(f"  - Auxiliary emotion head: {self.auxiliary_emotion_head}")
        print(f"  - Auxiliary loss weight: {self.aux_loss_weight}")
    
    def set_aux_loss_weight(self, weight):
        """动态设置辅助损失权重"""
        self.aux_loss_weight = weight
        print(f"辅助损失权重已更新为: {weight:.4f}")

    def encode_text(self, text):
        """优化显存的文本编码方法"""
        original_shape = text.shape
        
        # 检查输入是否已经是嵌入向量 (3D tensor with embedding dimension)
        if len(original_shape) == 3 and original_shape[-1] > 100:  # 假设嵌入维度 > 100
            # 输入已经是嵌入向量，直接处理
            # print(f"Input is already embeddings with shape {original_shape}")
            return self._encode_embeddings_impl(text)
        
        # 输入是token索引，需要转换为long类型
        text = text.long()
        
        if len(original_shape) == 3:
            batch_size, n_cls, seq_len = original_shape
            text = text.view(-1, seq_len)
        elif len(original_shape) == 2:
            batch_size, seq_len = original_shape
            n_cls = 1
        else:
            raise ValueError(f"Unexpected text shape: {original_shape}")
        
        # 使用gradient checkpointing for text encoding
        if self.training:
            x = torch.utils.checkpoint.checkpoint(
                self._encode_text_impl, text, use_reentrant=False
            )
        else:
            x = self._encode_text_impl(text)
        
        if len(original_shape) == 3:
            x = x.view(batch_size, n_cls, -1)
        
        return x
    
    def _encode_text_impl(self, text):
        """文本编码的具体实现"""
        current_batch_size, seq_len = text.shape
        
        # 添加边界检查
        if seq_len <= 0 or current_batch_size <= 0:
            raise ValueError(f"Invalid text dimensions: batch_size={current_batch_size}, seq_len={seq_len}")
        
        # Token embedding
        x = self.token_embedding(text).type(self.dtype)
        
        # 位置编码处理 - 修复潜在的索引越界问题
        pos_emb = self.positional_embedding.type(self.dtype)
        if pos_emb.shape[0] < seq_len:
            padding_len = seq_len - pos_emb.shape[0]
            padding = torch.zeros(padding_len, pos_emb.shape[1], 
                                device=pos_emb.device, dtype=pos_emb.dtype)
            pos_emb = torch.cat([pos_emb, padding], dim=0)
        
        # 确保不会越界
        pos_emb = pos_emb[:seq_len, :].unsqueeze(0).expand(current_batch_size, -1, -1)
        x = x + pos_emb
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # 动态构建attention mask - 添加设备和数据类型检查
        if seq_len > 0:
            attn_mask = torch.empty(seq_len, seq_len, device=text.device, dtype=self.dtype)
            attn_mask.fill_(float("-inf"))
            attn_mask.triu_(1)
        else:
            raise ValueError("Sequence length must be greater than 0")
        
        # 设置attention mask并进行transformer计算
        original_attn_masks = []
        for block in self.text_encoder.resblocks:
            original_attn_masks.append(getattr(block, 'attn_mask', None))
            block.attn_mask = attn_mask
        
        try:
            x = self.text_encoder(x)
        finally:
            for block, original_mask in zip(self.text_encoder.resblocks, original_attn_masks):
                if original_mask is not None:
                    block.attn_mask = original_mask
                elif hasattr(block, 'attn_mask'):
                    delattr(block, 'attn_mask')
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        # 取 [EOS] token 的特征 - 添加边界检查
        eot_indices = text.argmax(dim=-1)
        
        # 检查索引是否有效
        max_valid_index = x.shape[1] - 1  # sequence length - 1
        if torch.any(eot_indices > max_valid_index) or torch.any(eot_indices < 0):
            print(f"Warning: Invalid EOS indices detected. Max valid: {max_valid_index}, Got: {eot_indices}")
            # 将无效索引钳制到有效范围
            eot_indices = torch.clamp(eot_indices, 0, max_valid_index)
        
        # 确保索引在有效范围内
        batch_indices = torch.arange(x.shape[0], device=x.device)
        if len(batch_indices) != len(eot_indices):
            raise ValueError(f"Batch size mismatch: {len(batch_indices)} vs {len(eot_indices)}")
            
        x = x[batch_indices, eot_indices]
        
        if self.text_projection is not None:
            x = x @ self.text_projection
        
    def _encode_embeddings_impl(self, embeddings):
        """处理已经是嵌入向量的输入"""
        original_shape = embeddings.shape
        # print(f"Processing embeddings with original shape: {original_shape}")
        
        if len(original_shape) == 3:
            batch_size, seq_len, embed_dim = original_shape
            # 重塑为2D以便批量处理：(batch_size, seq_len, embed_dim)
            current_batch_size = batch_size
        else:
            raise ValueError(f"Expected 3D embeddings, got shape: {original_shape}")
        
        # 直接使用输入的嵌入向量
        x = embeddings.type(self.dtype)
        
        # 添加位置编码 - 检查维度匹配
        pos_emb = self.positional_embedding.type(self.dtype)
        # print(f"Position embedding shape: {pos_emb.shape}")
        # print(f"Input embeddings shape: {x.shape}")
        
        if pos_emb.shape[0] < seq_len:
            padding_len = seq_len - pos_emb.shape[0]
            padding = torch.zeros(padding_len, pos_emb.shape[1], 
                                device=pos_emb.device, dtype=pos_emb.dtype)
            pos_emb = torch.cat([pos_emb, padding], dim=0)
        
        # 确保位置编码维度匹配输入嵌入维度
        pos_emb_trimmed = pos_emb[:seq_len, :]  # 形状: (seq_len, pos_embed_dim)
        
        # 如果位置编码的维度与输入嵌入维度不匹配，需要进行调整
        if pos_emb_trimmed.shape[1] != embed_dim:
            print(f"Position embedding dim {pos_emb_trimmed.shape[1]} != input dim {embed_dim}")
            if pos_emb_trimmed.shape[1] > embed_dim:
                # 截断位置编码
                pos_emb_trimmed = pos_emb_trimmed[:, :embed_dim]
            else:
                # 填充位置编码
                padding_dim = embed_dim - pos_emb_trimmed.shape[1]
                padding = torch.zeros(seq_len, padding_dim, 
                                    device=pos_emb.device, dtype=pos_emb.dtype)
                pos_emb_trimmed = torch.cat([pos_emb_trimmed, padding], dim=1)
        
        pos_emb_expanded = pos_emb_trimmed.unsqueeze(0).expand(current_batch_size, -1, -1)
        # print(f"Position embedding expanded shape: {pos_emb_expanded.shape}")
        
        # 只有当维度完全匹配时才添加位置编码
        if pos_emb_expanded.shape == x.shape:
            x = x + pos_emb_expanded
        else:
            print(f"Warning: Skipping position encoding due to shape mismatch")
            print(f"pos_emb shape: {pos_emb_expanded.shape}, x shape: {x.shape}")
        
        x = x.permute(1, 0, 2)  # NLD -> LND (seq_len, batch_size, embed_dim)
        
        # 创建attention mask
        if seq_len > 0:
            attn_mask = torch.empty(seq_len, seq_len, device=embeddings.device, dtype=self.dtype)
            attn_mask.fill_(float("-inf"))
            attn_mask.triu_(1)
        else:
            raise ValueError("Sequence length must be greater than 0")
        
        # 应用transformer
        original_attn_masks = []
        for block in self.text_encoder.resblocks:
            original_attn_masks.append(getattr(block, 'attn_mask', None))
            block.attn_mask = attn_mask
        
        try:
            x = self.text_encoder(x)
        finally:
            for block, original_mask in zip(self.text_encoder.resblocks, original_attn_masks):
                if original_mask is not None:
                    block.attn_mask = original_mask
                elif hasattr(block, 'attn_mask'):
                    delattr(block, 'attn_mask')
        
        x = x.permute(1, 0, 2)  # LND -> NLD (batch_size, seq_len, embed_dim)
        x = self.ln_final(x).type(self.dtype)
        
        # print(f"After transformer, x shape: {x.shape}")
        
        # 对于嵌入向量输入，我们取最后一个token的特征
        # 或者可以取平均池化
        x = x[:, -1, :]  # 取最后一个位置的特征 (batch_size, embed_dim)
        
        # print(f"After selecting last token, x shape: {x.shape}")
        
        if self.text_projection is not None:
            x = x @ self.text_projection
        
        # print(f"After text projection, x shape: {x.shape}")
        
        # 注意：这里不需要恢复到3D形状，因为输出应该是2D的特征向量
        # 输出形状应该是 (batch_size, feature_dim)
        
        return x
    
    def forward(self, image, label=None, return_aux_loss=True):
        with torch.amp.autocast('cuda', enabled=self.training):
            # 图像编码
            image_features = self.image_encoder(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # 主要的CLIP损失
            prompts = self.prompt_learner(image_features)
            
            batch_size = image_features.shape[0]
            logits_list = []
            
            chunk_size = min(batch_size, 8)
            
            for i in range(0, batch_size, chunk_size):
                end_idx = min(i + chunk_size, batch_size)
                chunk_prompts = prompts[i:end_idx]
                chunk_image_features = image_features[i:end_idx]
                
                chunk_logits = []
                for pts_i, imf_i in zip(chunk_prompts, chunk_image_features):
                    if len(pts_i.shape) == 3:
                        pts_i = pts_i.squeeze(0)
                    
                    # 添加形状检查
                    if pts_i.numel() == 0:
                        print(f"Warning: Empty prompt tensor detected, skipping...")
                        continue
                        
                    try:
                        text_features = self.encode_text(pts_i)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        logit_scale = self.logit_scale.exp()
                        l_i = logit_scale * imf_i @ text_features.transpose(-2, -1)
                        chunk_logits.append(l_i)
                    except Exception as e:
                        print(f"Error in text encoding: {e}")
                        print(f"pts_i shape: {pts_i.shape}")
                        # 不要打印tensor内容，会触发CUDA错误
                        print(f"pts_i dtype: {pts_i.dtype}")
                        print(f"pts_i device: {pts_i.device}")
                        raise e
                
                if chunk_logits:  # 只有在有有效logits的情况下才扩展
                    logits_list.extend(chunk_logits)
        
        if not logits_list:
            raise RuntimeError("No valid logits computed - all prompts were empty or invalid")
            
        logits = torch.stack(logits_list)
        main_logits = logits
        
        # 辅助情绪分类
        aux_logits = None
        if return_aux_loss and self.training:
            aux_logits = self.auxiliary_emotion_head(image_features)
        
        # 计算损失
        if self.training and label is not None:
            # 主损失
            if len(label.shape) > 1:
                main_loss = F.binary_cross_entropy_with_logits(main_logits, label.float())
            else:
                main_loss = F.cross_entropy(main_logits, label)
            
            # 辅助损失
            if aux_logits is not None:
                if len(label.shape) > 1:
                    aux_loss = F.binary_cross_entropy_with_logits(aux_logits, label.float())
                else:
                    aux_loss = F.cross_entropy(aux_logits, label)
                
                # 组合损失
                total_loss = main_loss + self.aux_loss_weight * aux_loss
                return total_loss
            else:
                return main_loss
        
        return main_logits

@TRAINER_REGISTRY.register()
class EmoCoOp(TrainerX):
    """
    增强版的情感条件上下文优化 ，用于多标签分类。
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.data_loader = self.build_data_loader()
        
        # 初始化自适应权重调整器
        print("正在初始化自适应权重调整器...")
        self.weight_scheduler = AdaptiveWeightScheduler(
            initial_weight=getattr(cfg.TRAINER.EMOCOOP, 'AUX_LOSS_WEIGHT', 0.3),
            min_weight=getattr(cfg.TRAINER.EMOCOOP, 'MIN_AUX_LOSS_WEIGHT', 0.2),
            max_weight=getattr(cfg.TRAINER.EMOCOOP, 'MAX_AUX_LOSS_WEIGHT', 0.5),
            patience=getattr(cfg.TRAINER.EMOCOOP, 'WEIGHT_PATIENCE', 8),
            improvement_threshold=getattr(cfg.TRAINER.EMOCOOP, 'IMPROVEMENT_THRESHOLD', 0.01),
            decay_factor=getattr(cfg.TRAINER.EMOCOOP, 'DECAY_FACTOR', 0.95),
            growth_factor=getattr(cfg.TRAINER.EMOCOOP, 'GROWTH_FACTOR', 1.05),
            min_epochs_before_adjust=getattr(cfg.TRAINER.EMOCOOP, 'MIN_EPOCHS_BEFORE_ADJUST', 3),
            strategy=getattr(cfg.TRAINER.EMOCOOP, 'WEIGHT_STRATEGY', "conservative")
        )
        
        print(f"自适应权重调整器初始化完成:")
        print(f"  - 初始权重: {self.weight_scheduler.initial_weight}")
        print(f"  - 权重范围: [{self.weight_scheduler.min_weight}, {self.weight_scheduler.max_weight}]")
        print(f"  - 耐心值: {self.weight_scheduler.patience}")
        print(f"  - 改善阈值: {self.weight_scheduler.improvement_threshold}")
        print(f"  - 衰减因子: {self.weight_scheduler.decay_factor}")
        print(f"  - 增长因子: {self.weight_scheduler.growth_factor}")
        print(f"  - 当前权重: {self.weight_scheduler.get_weight():.4f}")
        
    def check_cfg(self, cfg):
        # 兼容EmoCoOp和CoCoOp参数读取
        trainer_cfg = getattr(cfg.TRAINER, 'EMOCOOP', None)
        if trainer_cfg is None:
            trainer_cfg = getattr(cfg.TRAINER, 'COCOOP')
        assert trainer_cfg.PREC in ["fp16", "fp32", "amp"]
        assert trainer_cfg.VOTE_TYPE in ["mean", "max", "weighted"]
    def build_model(self):
        """增强的模型构建方法"""
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.EMOCOOP.PREC == "fp32" or cfg.TRAINER.EMOCOOP.PREC == "amp":
            clip_model.float()

        print("Building Enhanced EMOCOOP CLIP with VPT and auxiliary heads")
        
        # 传递训练数据加载器用于聚类初始化
        train_loader = self.train_loader_x if hasattr(self, 'train_loader_x') else None
        
        self.model = CustomCLIP(
            cfg, classnames, clip_model, 
            dataloader=train_loader, 
            device=self.device
        )

        # 冻结不需要的参数
        names_to_update = ["prompt_learner", "visual_prompt_learner", "auxiliary_emotion_head"]
        for name, param in self.model.named_parameters():
            if not any(update_name in name for update_name in names_to_update):
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        
        # 优化器设置 - 包括辅助分类头的参数
        params_to_optimize = []
        params_to_optimize.extend(list(self.model.prompt_learner.parameters()))
        params_to_optimize.extend(list(self.model.visual_prompt_learner.parameters()))
        params_to_optimize.extend(list(self.model.auxiliary_emotion_head.parameters()))
        
        self.optim = build_optimizer(params_to_optimize, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.register_model("visual_prompt_learner", self.model.visual_prompt_learner, self.optim, self.sched)
        self.register_model("auxiliary_emotion_head", self.model.auxiliary_emotion_head, self.optim, self.sched)

        # 使用AMP scaler
        self.scaler = GradScaler() if cfg.TRAINER.EMOCOOP.PREC == "amp" else None

        # GPU设置
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), using DataParallel")
            self.model = nn.DataParallel(self.model)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            torch.backends.cudnn.benchmark = True
            self.model.to("cuda:0")
            
        self.bce_loss = nn.BCEWithLogitsLoss()
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        # 显示当前权重状态（每100个batch显示一次）
        if hasattr(self, 'weight_scheduler') and (self.batch_idx + 1) % 100 == 0:
            current_weight = self.weight_scheduler.get_weight()
            print(f"Batch {self.batch_idx + 1}: 当前辅助损失权重 = {current_weight:.4f}")
        
        prec = self.cfg.TRAINER.EMOCOOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image, label)  # 直接返回loss
                if isinstance(output, torch.Tensor) and output.dim() == 0:
                    loss = output
                else:
                    if len(label.shape) > 1:
                        loss = self.bce_loss(output, label)
                    else:
                        loss = F.cross_entropy(output, label)
            
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            # 获取当前自适应权重
            current_weight = self.weight_scheduler.get_weight()
            self.model.set_aux_loss_weight(current_weight)
            
            output = self.model(image, label)
            if isinstance(output, torch.Tensor) and output.dim() == 0:
                loss = output
            else:
                if len(label.shape) > 1:
                    loss = self.bce_loss(output, label)
                else:
                    loss = F.cross_entropy(output, label)
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        loss_summary = {"loss": loss.item()}

        # 每50个batch显示一次损失信息
        if (self.batch_idx + 1) % 50 == 0:
            current_epoch = getattr(self, 'epoch', 'N/A')
            print(f"Epoch {current_epoch}, Batch {self.batch_idx + 1}/{self.num_batches}: Loss = {loss.item():.4f}")

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        # 定期清理显存
        if (self.batch_idx + 1) % 10 == 0:
            torch.cuda.empty_cache()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        
        input = input.to(self.device)
        label = label.to(self.device)
        
        return input, label
    
    def parse_batch_test(self, batch):
        return self.parse_batch_train(batch)

    def model_inference(self, input):
        return self.model(input)

    def top_k_recall(self, probs, labels, k=2):
        """计算TopK召回率"""
        total = 0
        correct = 0
        
        for i in range(len(probs)):
            true_labels = torch.nonzero(labels[i], as_tuple=False).squeeze(-1)
            
            if true_labels.dim() == 0:
                true_labels = true_labels.unsqueeze(0)
            
            if true_labels.numel() == 0:
                continue
                
            topk_indices = torch.topk(probs[i], k=min(k, probs.size(-1))).indices
            
            if len(torch.tensor([idx for idx in true_labels if idx in topk_indices])) > 0:
                correct += 1
            total += 1
                
        return correct / total if total > 0 else 0

    def evaluate(self, split_name="val"):
        """多标签分类的评估方法"""
        self.set_model_mode("eval")
        self.evaluator.reset()

        # 添加调试信息
        print(f"开始评估 split_name: {split_name}")
        print(f"是否有权重调整器: {hasattr(self, 'weight_scheduler')}")
        if hasattr(self, 'weight_scheduler'):
            print(f"当前权重: {self.weight_scheduler.get_weight():.4f}")
            print(f"权重调整历史长度: {len(self.weight_scheduler.history)}")

        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.data_loader[split_name]):
                input, target = self.parse_batch_test(batch)
                output = self.model_inference(input)
                
                # 确保形状匹配
                if output.shape != target.shape:
                    if len(target.shape) == 1:
                        target = F.one_hot(target.long(), num_classes=output.size(1)).float()
                    elif target.shape[1] == 1 and output.shape[1] > 1:
                        target = F.one_hot(target.long().squeeze(1), num_classes=output.size(1)).float()
                    elif target.shape[0] == output.shape[1] and target.shape[1] == output.shape[0]:
                        target = target.t()
                
                probs = torch.sigmoid(output)
                preds = (probs >= 0.5).float()
                
                all_labels.append(target.cpu())
                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())
        
        # 合并结果
        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()
        all_probs = torch.cat(all_probs).numpy()

        labels = torch.tensor(all_labels)
        probs = torch.tensor(all_probs)
        
        # 计算评估指标 - 只保留需要的指标
        results = {}
        
        # 只计算需要的指标
        recall_micro = recall_score(all_labels, all_preds, average='micro', zero_division=0)
        f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        
        recall_at_2 = self.top_k_recall(torch.tensor(all_probs), torch.tensor(all_labels), k=2)
        recall_at_3 = self.top_k_recall(torch.tensor(all_probs), torch.tensor(all_labels), k=3)
        
        # 计算 mAP
        ap_per_class = []
        for i in range(all_labels.shape[1]):
            if np.sum(all_labels[:, i]) > 0:
                ap = average_precision_score(all_labels[:, i], all_probs[:, i])
                ap_per_class.append(ap)
        
        map_score = np.mean(ap_per_class) if ap_per_class else 0
        
        results = {
            "mAP": map_score,
            "recall@2": recall_at_2,
            "recall@3": recall_at_3,
            "recall_micro": recall_micro,
            "f1_micro": f1_micro
        }
        
        print(f"* Results on {split_name}:")
        for k, v in results.items():
            print(f"* {k}: {v:.4f}")
        
        # 如果是验证集，显示更详细的信息
        if split_name == "val":
            print(f"\n验证集详细指标 (Epoch {getattr(self, 'epoch', 'N/A')}):")
            print(f"  - 主要指标 (Recall@2): {recall_at_2:.4f}")
            print(f"  - 其他重要指标:")
            print(f"    * mAP: {map_score:.4f}")
            print(f"    * F1-Micro: {f1_micro:.4f}")
            print(f"    * Recall-Micro: {recall_micro:.4f}")
            print(f"    * Recall@3: {recall_at_3:.4f}")
        
        # 更新自适应权重（在验证评估时执行）
        print(f"检查权重更新条件: split_name={split_name}, 是否在列表中: {split_name in ['val', 'test']}")
        if split_name in ["val", "test"] and hasattr(self, 'weight_scheduler'):
            print("开始更新权重...")
            current_recall_at_2 = recall_at_2
            new_weight = self.weight_scheduler.update(current_recall_at_2)
            
            # 更新模型中的权重
            if hasattr(self.model, 'set_aux_loss_weight'):
                self.model.set_aux_loss_weight(new_weight)
                print(f"已更新模型权重: {new_weight:.4f}")
            elif hasattr(self.model, 'module') and hasattr(self.model.module, 'set_aux_loss_weight'):
                self.model.module.set_aux_loss_weight(new_weight)
                print(f"已更新模型权重(DataParallel): {new_weight:.4f}")
            else:
                print("警告: 无法找到set_aux_loss_weight方法")
            
            print(f"* 当前recall@2: {current_recall_at_2:.4f}")
            print(f"* 更新后的辅助损失权重: {new_weight:.4f}")
            print(f"* 权重调整历史: {len(self.weight_scheduler.history)}次调整")
        else:
            print(f"跳过权重更新: split_name={split_name}, 有weight_scheduler={hasattr(self, 'weight_scheduler')}")
        
        return list(results.values())[1], all_probs, all_labels  # 返回 recall@2 作为主要评估指标

    def test(self, split_name="test"):
        """多标签分类的测试评估方法，包括结果保存"""
        self.set_model_mode("test")
        self.evaluator.reset()
        results, probs, labels = self.evaluate(split_name="test")

        all_labels = []
        all_preds = []
        all_probs = []
        file_names = []  # 用于保存每个图像的文件名
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.data_loader[split_name]):
                input, target = self.parse_batch_test(batch)
                output = self.model_inference(input)
                
                # 确保target与output形状匹配（仅用于评估指标计算）
                if output.shape != target.shape:
                    # 如果需要，调整target形状
                    if len(target.shape) == 1:
                        target = F.one_hot(target.long(), num_classes=output.size(1)).float()
                    elif target.shape[1] == 1 and output.shape[1] > 1:
                        target = F.one_hot(target.long().squeeze(1), num_classes=output.size(1)).float()
                    elif target.shape[0] == output.shape[1] and target.shape[1] == output.shape[0]:
                        target = target.t()

                # 使用sigmoid将输出转换为概率
                probs = torch.sigmoid(output)
                # 使用阈值0.5将概率转换为二进制预测
                preds = (probs >= 0.5).float()

                # 保存每个batch的文件名、标签、预测和概率
                file_names.extend(batch["impath"])  # 获取每个图像的文件名
                all_labels.append(target.cpu())
                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())

        # 合并所有批次的结果
        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()
        all_probs = torch.cat(all_probs).numpy()

        # 保存为 CSV 文件，文件名和每个类别的概率
        df = pd.DataFrame(all_probs, columns=[f"class_{i}" for i in range(all_probs.shape[1])])
        df.insert(0, "filename", file_names)  # 将文件名插入到前面
        df.to_csv(f"{split_name}_predictions.csv", index=False)
        print(f"已保存预测结果到 {split_name}_predictions.csv")

        # 计算多标签评估指标 - 只保留需要的指标
        results = {}

        # 只计算需要的指标
        recall_micro = recall_score(all_labels, all_preds, average='micro', zero_division=0)
        f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)

        # 计算top-k recall
        recall_at_2 = self.top_k_recall(torch.tensor(all_probs), torch.tensor(all_labels), k=2)
        recall_at_3 = self.top_k_recall(torch.tensor(all_probs), torch.tensor(all_labels), k=3)        

        # 计算平均精度均值 (mAP)
        ap_per_class = []
        for i in range(all_labels.shape[1]):
            if np.sum(all_labels[:, i]) > 0:  # 只考虑有正样本的类别
                ap = average_precision_score(all_labels[:, i], all_probs[:, i])
                ap_per_class.append(ap)

        map_score = np.mean(ap_per_class) if ap_per_class else 0

        results = {
            "mAP": map_score,
            "recall@2": recall_at_2,
            "recall@3": recall_at_3,
            "recall_micro": recall_micro,
            "f1_micro": f1_micro
        }

        print(f"* Results on {split_name}:")
        for k, v in results.items():
            print(f"* {k}: {v:.4f}")

        # 返回mAP作为主要评估指标
        return results["mAP"]

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def save_weight_history(self, output_dir):
        """保存权重调整历史记录到文件"""
        if hasattr(self, 'weight_scheduler'):
            history = self.weight_scheduler.get_history()
            if history:
                import json
                history_file = os.path.join(output_dir, 'weight_adjustment_history.json')
                with open(history_file, 'w') as f:
                    json.dump(history, f, indent=2)
                print(f"权重调整历史已保存到: {history_file}")
                
                # 同时保存为CSV格式
                import pandas as pd
                df = pd.DataFrame(history)
                csv_file = os.path.join(output_dir, 'weight_adjustment_history.csv')
                df.to_csv(csv_file, index=False)
                print(f"权重调整历史已保存到: {csv_file}")

    def before_train(self):
        """训练开始前的处理"""
        super().before_train()
        
        print(f"\n{'='*80}")
        print(f"开始训练 - 自适应权重调整模式")
        print(f"{'='*80}")
        print(f"训练配置:")
        print(f"  - 总epoch数: {self.cfg.OPTIM.MAX_EPOCH}")
        print(f"  - 学习率: {self.cfg.OPTIM.LR}")
        print(f"  - 批次大小: {self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE}")
        print(f"  - 训练样本数: {len(self.train_loader_x.dataset)}")
        print(f"  - 验证样本数: {len(self.val_loader.dataset) if hasattr(self, 'val_loader') else 'N/A'}")
        
        if hasattr(self, 'weight_scheduler'):
            print(f"\n自适应权重调整器配置:")
            print(f"  - 初始权重: {self.weight_scheduler.initial_weight:.4f}")
            print(f"  - 权重范围: [{self.weight_scheduler.min_weight:.4f}, {self.weight_scheduler.max_weight:.4f}]")
            print(f"  - 耐心值: {self.weight_scheduler.patience}")
            print(f"  - 改善阈值: {self.weight_scheduler.improvement_threshold:.4f}")
            print(f"  - 衰减因子: {self.weight_scheduler.decay_factor:.2f}")
            print(f"  - 增长因子: {self.weight_scheduler.growth_factor:.2f}")
        
        print(f"{'='*80}\n")

    def after_epoch(self):
        """每个epoch结束后的处理"""
        super().after_epoch()
        
        # 在每个epoch结束后进行验证评估
        if hasattr(self, 'data_loader') and 'val' in self.data_loader:
            print(f"\n{'='*60}")
            print(f"Epoch {self.epoch} 结束，开始验证评估...")
            print(f"{'='*60}")
            
            try:
                # 调用evaluate方法进行验证
                val_score = self.evaluate(split_name="val")
                
                # 显示详细的验证结果
                print(f"\nEpoch {self.epoch} 验证结果:")
                if isinstance(val_score, tuple):
                    mAP = val_score[0]
                else:
                    mAP = val_score
                print(f"  - mAP: {mAP:.4f}")
                
                # 更新自适应权重
                if hasattr(self, 'weight_scheduler'):
                    old_weight = self.weight_scheduler.get_weight()
                    self.weight_scheduler.update(mAP)
                    new_weight = self.weight_scheduler.get_weight()
                    
                    if old_weight != new_weight:
                        print(f"  - 权重已调整: {old_weight:.4f} -> {new_weight:.4f}")
                    else:
                        print(f"  - 权重保持不变: {old_weight:.4f}")
                
                # 如果有权重调整器，显示权重信息
                if hasattr(self, 'weight_scheduler'):
                    current_weight = self.weight_scheduler.get_weight()
                    history_length = len(self.weight_scheduler.history)
                    print(f"  - 当前辅助损失权重: {current_weight:.4f}")
                    print(f"  - 权重调整历史: {history_length}次")
                    
                    if history_length > 0:
                        # 显示最近的权重变化
                        recent_history = self.weight_scheduler.history[-3:]  # 最近3次
                        print(f"  - 最近权重变化:")
                        for i, record in enumerate(recent_history):
                            print(f"    {i+1}. Score: {record['score']:.4f}, Weight: {record['weight']:.4f}")
                
                print(f"{'='*60}")
                
            except Exception as e:
                print(f"验证评估时出错: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Epoch {self.epoch} 结束，但未找到验证集")
            if hasattr(self, 'data_loader'):
                print(f"可用的数据集: {list(self.data_loader.keys())}")

    def after_train(self):
        """训练结束后的处理"""
        super().after_train()
        
        print(f"\n{'='*80}")
        print(f"训练结束 - 自适应权重调整模式")
        print(f"{'='*80}")
        
        # 保存权重调整历史
        if hasattr(self, 'weight_scheduler'):
            self.save_weight_history(self.cfg.OUTPUT_DIR)
            
            # 打印最终的权重调整统计
            history = self.weight_scheduler.get_history()
            if history:
                print(f"\n权重调整统计:")
                print(f"  - 总调整次数: {len(history)}")
                print(f"  - 初始权重: {self.weight_scheduler.initial_weight:.4f}")
                print(f"  - 最终权重: {self.weight_scheduler.current_weight:.4f}")
                print(f"  - 最高recall@2: {self.weight_scheduler.best_score:.4f}")
                
                # 计算权重变化范围
                weights = [h['weight'] for h in history]
                print(f"  - 权重变化范围: [{min(weights):.4f}, {max(weights):.4f}]")
                
                # 分析权重变化趋势
                weight_changes = []
                for i in range(1, len(weights)):
                    change = weights[i] - weights[i-1]
                    weight_changes.append(change)
                
                positive_changes = sum(1 for c in weight_changes if c > 0)
                negative_changes = sum(1 for c in weight_changes if c < 0)
                no_changes = sum(1 for c in weight_changes if c == 0)
                
                print(f"\n权重变化分析:")
                print(f"  - 增加次数: {positive_changes}")
                print(f"  - 减少次数: {negative_changes}")
                print(f"  - 保持不变: {no_changes}")
                
                # 显示权重变化历史
                print(f"\n权重变化历史:")
                for i, record in enumerate(history):
                    print(f"  Epoch {i+1}: Score={record['score']:.4f}, Weight={record['weight']:.4f}")
            else:
                print("警告: 没有权重调整历史记录")
        else:
            print("警告: 没有找到权重调整器")
        
        print(f"{'='*80}")
        print(f"训练完成!")
        print(f"{'='*80}")

