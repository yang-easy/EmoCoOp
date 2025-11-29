import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import numpy as np

_tokenizer = _Tokenizer()

def load_clip_to_cpu(backbone_name):
    """Load CLIP model to CPU.
    
    Args:
        backbone_name (str): Name of the CLIP backbone.
    """
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp) for Multi-label Classification.
    """
    def __init__(self, cfg):
        super().__init__(cfg)  # 确保这行存在
        self.data_loader = self.build_data_loader() 
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg.MODEL.BACKBONE.NAME)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # 针对多标签分类设置二元交叉熵损失
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward_backward(self, batch):
        image, target = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = self.bce_loss(output, target)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            
            # 确保target的形状与output匹配
            if output.shape != target.shape:
                # 打印出形状以便调试
                print(f"Shape mismatch: output {output.shape}, target {target.shape}")
                
                if len(target.shape) == 2 and target.shape[1] == 1:
                    # 如果target是[batch_size, 1]，可能需要对其进行变换
                    if hasattr(self.dm.dataset, 'num_classes'):
                        num_classes = self.dm.dataset.num_classes
                        # 转换为one-hot编码
                        new_target = torch.zeros(target.size(0), num_classes, device=target.device)
                        for i in range(target.size(0)):
                            if target[i, 0] < num_classes:  # 安全检查
                                new_target[i, target[i, 0].long()] = 1.0
                        target = new_target
                    else:
                        # 如果不知道类别数，尝试从输出推断
                        target = F.one_hot(target.long().squeeze(1), num_classes=output.size(1)).float()
                
                # 如果target是一维的[batch_size]，转换为与output相同的形状
                elif len(target.shape) == 1:
                    # 我们假设这是多标签分类，每个样本可能有多个标签
                    # 但如果target是索引形式，我们需要转换为one-hot
                    if output.size(1) > 1:  # 确保这是多类问题
                        target = F.one_hot(target.long(), num_classes=output.size(1)).float()
            
            # 最后检查确保形状匹配
            if output.shape != target.shape:
                # 如果仍然不匹配，尝试转置target
                if target.shape[0] == output.shape[1] and target.shape[1] == output.shape[0]:
                    target = target.t()
                # 如果还是不匹配，则调整target使其满足要求
                else:
                    # 确保target是二维的正确形状
                    target = target.view(output.size(0), -1)
                    # 如果列数仍然不匹配，进行填充或裁剪
                    if target.size(1) != output.size(1):
                        if target.size(1) < output.size(1):
                            # 填充
                            padding = torch.zeros(target.size(0), output.size(1) - target.size(1), 
                                                 device=target.device)
                            target = torch.cat([target, padding], dim=1)
                        else:
                            # 裁剪
                            target = target[:, :output.size(1)]
            
            # 确保target是浮点类型
            target = target.float()
            
            # 应用损失函数
            loss = self.bce_loss(output, target)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        # 获取多热编码标签
        multi_hot = batch["label"]
        
        # 确保multi_hot是tensor
        if isinstance(multi_hot, list):
            multi_hot = torch.stack(multi_hot)
        
        # 获取类别数量 - 确保我们可以创建正确维度的标签
        num_classes = self.dm.dataset.num_classes if hasattr(self.dm.dataset, 'num_classes') else None
        if num_classes is None and hasattr(self.model, 'prompt_learner'):
            num_classes = self.model.prompt_learner.n_cls
            
        # 关键修复：确保标签的形状是 [batch_size, num_classes]
        if len(multi_hot.shape) == 1:
            # 如果是一维的，可能是索引标签或者被flatten的多热向量
            if num_classes is not None:
                if torch.max(multi_hot) < num_classes:  # 可能是索引标签
                    new_multi_hot = torch.zeros(multi_hot.size(0), num_classes, 
                                              device=multi_hot.device)
                    for i in range(multi_hot.size(0)):
                        new_multi_hot[i, multi_hot[i].long()] = 1.0
                    multi_hot = new_multi_hot
                else:  # 可能是被flatten的多热向量
                    multi_hot = multi_hot.view(-1, num_classes)
            else:
                # 如果不知道类别数，只能尝试reshape
                batch_size = input.size(0)
                if multi_hot.size(0) % batch_size == 0:
                    multi_hot = multi_hot.view(batch_size, -1)
        
        # 确保是浮点类型，用于BCE损失
        multi_hot = multi_hot.float()
        
        input = input.to(self.device)
        multi_hot = multi_hot.to(self.device)
        return input, multi_hot

    def parse_batch_test(self, batch):
        return self.parse_batch_train(batch)

    def model_inference(self, input):
        return self.model(input)

    
    def top_k_recall(self, probs, labels, k=1):
        total = 0
        correct = 0
        for i in range(len(probs)):
            true_label = labels[i]
            if true_label.dim() > 0 and true_label.numel() > 1:
                true_label = torch.argmax(true_label)  # 处理 one-hot 或多维标签
            else:
                true_label = true_label.item()
            topk = torch.topk(probs[i], k=k).indices.cpu().numpy()
            if true_label in topk:
                correct += 1
            total += 1
        return correct / total


    def evaluate(self, split_name="val"):
        """多标签分类的评估方法"""
        self.set_model_mode("eval")
        self.evaluator.reset()

        all_labels = []
        all_preds = []
        all_probs = []
        
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
                
                all_labels.append(target.cpu())
                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())
        
        # 合并所有批次的结果
        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()
        all_probs = torch.cat(all_probs).numpy()
        
        # 计算多标签评估指标
        results = {}
        
        # 样本平均指标
        precision_samples = precision_score(all_labels, all_preds, average='samples', zero_division=0)
        recall_samples = recall_score(all_labels, all_preds, average='samples', zero_division=0)
        f1_samples = f1_score(all_labels, all_preds, average='samples', zero_division=0)
        
        # 微平均指标
        precision_micro = precision_score(all_labels, all_preds, average='micro', zero_division=0)
        recall_micro = recall_score(all_labels, all_preds, average='micro', zero_division=0)
        f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        
        # 宏平均指标
        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
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
            "precision_samples": precision_samples,
            "recall_samples": recall_samples,
            "f1_samples": f1_samples,
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "f1_micro": f1_micro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "mAP": map_score,
            "recall@2": recall_at_2,
            "recall@3": recall_at_3
        }
        
        print(f"* Results on {split_name}:")
        for k, v in results.items():
            print(f"* {k}: {v:.4f}")
        
        return list(results.values())[10]  # 返回第10个指标作为主要评估指标


    def test(self, split_name="test"):
        """多标签分类的评估方法"""
        self.set_model_mode("test")
        self.evaluator.reset()

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

        # 计算多标签评估指标
        results = {}

        # 样本平均指标
        precision_samples = precision_score(all_labels, all_preds, average='samples', zero_division=0)
        recall_samples = recall_score(all_labels, all_preds, average='samples', zero_division=0)
        f1_samples = f1_score(all_labels, all_preds, average='samples', zero_division=0)

        # 微平均指标
        precision_micro = precision_score(all_labels, all_preds, average='micro', zero_division=0)
        recall_micro = recall_score(all_labels, all_preds, average='micro', zero_division=0)
        f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)

        # 宏平均指标
        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)

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
            "precision_samples": precision_samples,
            "recall_samples": recall_samples,
            "f1_samples": f1_samples,
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "f1_micro": f1_micro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "mAP": map_score,
            "recall@2": recall_at_2,
            "recall@3": recall_at_3,
        }

        print(f"* Results on {split_name}:")
        for k, v in results.items():
            print(f"* {k}: {v:.4f}")

        return list(results.values())[10]  # 返回第10个指标作为主要评估指标
        
