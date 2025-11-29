import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np
from sklearn.cluster import KMeans
import pickle
import os

class ColorToneAnalyzer(nn.Module):
    """
    分析图像的色调冷暖程度的模块
    """
    def __init__(self):
        super().__init__()
        # 定义冷色调和暖色调的参考色谱
        # 冷色调：蓝色、青色、紫色等的RGB权重
        self.cool_weights = nn.Parameter(torch.tensor([0.2, 0.3, 0.5]).view(1, 3, 1, 1))
        self.warm_weights = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]).view(1, 3, 1, 1))

        # 使用简单的卷积层做特征提取
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 2)  # 输出冷暖程度
        
    def color_histogram_analysis(self, x):
        """基于颜色直方图的分析方法"""
        # 计算与冷暖色调的相似性
        cool_score = F.conv2d(x, self.cool_weights.to(x.device))
        warm_score = F.conv2d(x, self.warm_weights.to(x.device))
        
        # 获取平均分数
        cool_mean = cool_score.mean(dim=[2, 3])
        warm_mean = warm_score.mean(dim=[2, 3])
        
        # 归一化分数
        total = cool_mean + warm_mean
        cool_ratio = cool_mean / total
        warm_ratio = warm_mean / total
        
        return torch.cat([cool_ratio, warm_ratio], dim=1)
    
    def fourier_analysis(self, x):
        """基于频域分析的方法"""
        # 转换到YUV色彩空间，Y为亮度，U和V包含色度信息
        # 近似转换
        y = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        u = -0.147 * x[:, 0] - 0.289 * x[:, 1] + 0.436 * x[:, 2]
        v = 0.615 * x[:, 0] - 0.515 * x[:, 1] - 0.100 * x[:, 2]
        
        # 对色度通道进行傅里叶变换
        u_fft = torch.fft.fft2(u)
        v_fft = torch.fft.fft2(v)
        
        # 取幅度谱
        u_magnitude = torch.abs(u_fft)
        v_magnitude = torch.abs(v_fft)
        
        # 分析低频部分（整体色调）
        h, w = u_magnitude.shape[-2:]
        center_h, center_w = h // 2, w // 2
        radius = min(h, w) // 8  # 低频区域半径
        
        # 创建低频掩码
        y_grid, x_grid = torch.meshgrid(
            torch.arange(h, device=x.device),
            torch.arange(w, device=x.device),
            indexing='ij'
        )
        distance = torch.sqrt((y_grid - center_h) ** 2 + (x_grid - center_w) ** 2)
        low_freq_mask = (distance < radius).float()
        
        # 计算U和V通道的低频能量
        u_low_energy = (u_magnitude * low_freq_mask).sum(dim=[1, 2])
        v_low_energy = (v_magnitude * low_freq_mask).sum(dim=[1, 2])
        
        # U负值通常对应更多蓝色(冷)，V正值通常对应更多红色(暖)
        cool_score = -u_low_energy  # 负U表示蓝色偏向
        warm_score = v_low_energy   # 正V表示红色偏向
        
        # 归一化
        scores = torch.stack([cool_score, warm_score], dim=1)
        scores = F.softmax(scores, dim=1)
        
        return scores
        
    def forward(self, x, use_fourier=True):
        """
        分析图像的冷暖色调
        
        Args:
            x: 图像输入张量，形状为 [B, 3, H, W]，已归一化到 [0, 1]
            use_fourier: 是否使用频域分析
            
        Returns:
            final_scores: 表示冷暖程度的张量，形状为 [B, 2]
            features: 提取的深度特征，形状为 [B, 32]
        """
        # 使用两种方法计算色调
        hist_scores = self.color_histogram_analysis(x)
        
        if use_fourier:
            fourier_scores = self.fourier_analysis(x)
            # 综合两种方法的结果
            color_tone = (hist_scores + fourier_scores) / 2
        else:
            color_tone = hist_scores
            
        # 也可以使用卷积网络提取更复杂的特征
        features = F.relu(self.conv1(x))
        features = F.max_pool2d(features, 2)
        features = F.relu(self.conv2(features))
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        deep_scores = F.softmax(self.fc(features), dim=1)
        
        # 组合简单分析和深度特征的结果
        final_scores = (color_tone + deep_scores) / 2
        
        return final_scores, features

class ModifiedVisualTransformer(nn.Module):
    """修改后的视觉Transformer，支持视觉提示学习"""
    def __init__(self, original_visual, visual_prompt_learner):
        super().__init__()
        self.original_visual = original_visual
        self.visual_prompt_learner = visual_prompt_learner
        
        # 复制原始属性
        self.input_resolution = original_visual.input_resolution
        self.output_dim = original_visual.output_dim
        self.conv1 = original_visual.conv1
        self.class_embedding = original_visual.class_embedding
        self.positional_embedding = original_visual.positional_embedding
        self.ln_pre = original_visual.ln_pre
        self.transformer = original_visual.transformer
        self.ln_post = original_visual.ln_post
        self.proj = original_visual.proj
    
    def forward(self, x, text_ctx=None):
        patch_features_raw = self.conv1(x)  # shape = [*, width, grid, grid]
        patch_features = patch_features_raw.reshape(patch_features_raw.shape[0], patch_features_raw.shape[1], -1)  # shape = [*, width, grid ** 2]
        patch_features = patch_features.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        
        x = torch.cat([self.class_embedding.to(patch_features.dtype) + torch.zeros(patch_features.shape[0], 1, patch_features.shape[-1], dtype=patch_features.dtype, device=patch_features.device), patch_features], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        
        # 添加视觉提示
        if hasattr(self.visual_prompt_learner, 'forward'):
            # 传入 patch features 和 text_ctx
            visual_prompts = self.visual_prompt_learner(patch_features, text_ctx)
            if visual_prompts.dim() == 3 and visual_prompts.shape[0] == 1:
                visual_prompts = visual_prompts.expand(x.shape[0], -1, -1)
            elif visual_prompts.dim() == 2:
                visual_prompts = visual_prompts.unsqueeze(0).expand(x.shape[0], -1, -1)
            
            # 在class token之后插入视觉提示
            x = torch.cat([x[:, :1], visual_prompts, x[:, 1:]], dim=1)
        
        x = self.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # 移除视觉提示部分，只保留原始token
        if hasattr(self.visual_prompt_learner, 'forward'):
             prompt_len = visual_prompts.shape[1]
             cls_token = x[:, 0, :].unsqueeze(1)
             patch_tokens = x[:, 1 + prompt_len :, :]
             x = torch.cat([cls_token, patch_tokens], dim=1)


        x = self.ln_post(x[:, 0, :])
        
        if self.proj is not None:
            x = x @ self.proj
        
        return x

class VisualPromptLearner(nn.Module):
    """增强的视觉提示学习器 - 支持动态生成和文生图耦合"""
    def __init__(self, cfg, clip_model, dataloader=None, device='cuda'):
        super().__init__()
        self.cfg = cfg
        
        self.n_visual_ctx = min(cfg.TRAINER.EMOCOOP.N_VISUAL_CTX, 16) # 增加最大值
        self.vpt_deep = False
        
        vision_width = clip_model.visual.transformer.width
        patch_width = clip_model.visual.conv1.out_channels
        text_width = clip_model.ln_final.weight.shape[0]

        # 1. 动态视觉提示的Meta-Net
        self.visual_meta_net = nn.Sequential(
            nn.Linear(patch_width, patch_width * 2),
            nn.ReLU(inplace=True),
            nn.Linear(patch_width * 2, vision_width)
        )

        # 2. 耦合V-T的交叉注意力模块
        self.coupling_layer = nn.MultiheadAttention(
            embed_dim=vision_width, 
            num_heads=4, # 可调超参数
            batch_first=True
        )
        
        # 用于将文本特征维度映射到视觉特征维度
        if text_width != vision_width:
            self.text_to_vision_projection = nn.Linear(text_width, vision_width)
        else:
            self.text_to_vision_projection = nn.Identity()

        # 初始化视觉聚类工具
        self.cluster_initializer = VisualClusterInitializer(
            n_clusters=self.n_visual_ctx
        )
        
        # 尝试使用聚类中心初始化
        if dataloader is not None:
            print("Initializing visual prompts with cluster centers...")
            cache_file = f"./cache/visual_clusters_{self.n_visual_ctx}.pkl"
            cluster_prompts = self.cluster_initializer.initialize_visual_prompts(
                dataloader, target_dim=vision_width, device=device, cache_file=cache_file
            )
            self.visual_ctx = nn.Parameter(cluster_prompts.squeeze(0))
        else:
            # 后备随机初始化
            print("Using random initialization for visual prompts")
            self.visual_ctx = nn.Parameter(
                torch.randn(self.n_visual_ctx, vision_width)
            )
            nn.init.normal_(self.visual_ctx, std=0.01)
        
        print(f"Enhanced Visual Prompt Tuning initialized:")
        print(f"  - Number of visual contexts: {self.n_visual_ctx}")
        print(f"  - Deep VPT: {self.vpt_deep}")
        print(f"  - Visual context shape: {self.visual_ctx.shape}")
        print(f"  - Using dynamic generation and V-T coupling.")

    def forward(self, patch_features, text_ctx):
        """
        Args:
            patch_features (torch.Tensor): [B, num_patches, patch_dim]
            text_ctx (torch.Tensor): [num_text_ctx, text_dim]
        """
        # 1. 动态生成
        # 使用patch特征的均值作为全局图像表征
        if patch_features.dim() == 3:
            global_patch_feature = patch_features.mean(dim=1)  # ViT: [B, D]
        else:
            global_patch_feature = patch_features  # ResNet: [B, D]
        bias = self.visual_meta_net(global_patch_feature) # [B, vision_width]
        
        # 将bias应用到静态的visual_ctx上
        dynamic_visual_ctx = self.visual_ctx.unsqueeze(0) + bias.unsqueeze(1) # [B, n_visual_ctx, vision_width]

        # 2. 耦合V-T的交叉注意力
        if text_ctx is not None:
            # 将text_ctx扩展到batch维度
            text_ctx_expanded = text_ctx.unsqueeze(0).expand(dynamic_visual_ctx.shape[0], -1, -1)
            
            # 将文本特征投影到与视觉特征兼容的维度
            projected_text_ctx = self.text_to_vision_projection(text_ctx_expanded)
            
            # 使用动态生成的视觉提示作为query，文本上下文作为key和value
            coupled_visual_ctx, _ = self.coupling_layer(
                query=dynamic_visual_ctx,
                key=projected_text_ctx,
                value=projected_text_ctx,
            )
            
            # 将耦合后的提示与原始动态提示相结合 (残差连接)
            final_visual_ctx = dynamic_visual_ctx + coupled_visual_ctx
        else:
            # 如果没有文本上下文，直接使用动态生成的视觉提示
            final_visual_ctx = dynamic_visual_ctx
        
        return final_visual_ctx

    def visualize_all_prompt_types(self, all_features, text_ctx, save_dir='prompt_visualizations', dim_reduce='pca'):
        """
        生成三种prompt类型（static/dynamic/coupled）的聚类分布图和聚类指标柱状图。
        all_features: numpy array, test集特征 [N, D]
        text_ctx: torch.Tensor or numpy array, 文本上下文
        save_dir: 保存图片的目录
        dim_reduce: 'pca'|'tsne'|'umap'
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        from sklearn.metrics.pairwise import cosine_similarity
        import os
        os.makedirs(save_dir, exist_ok=True)
        # 降维方法选择
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            TSNE = None
        try:
            import umap
        except ImportError:
            umap = None
        def reduce_dim(X, method='pca'):
            if method == 'tsne' and TSNE is not None:
                return TSNE(n_components=2, random_state=42).fit_transform(X)
            elif method == 'umap' and umap is not None:
                return umap.UMAP(n_components=2, random_state=42).fit_transform(X)
            else:
                return PCA(n_components=2).fit_transform(X)

        prompt_types = ['static', 'dynamic', 'coupled']
        results = {}
        dynamic_embeds = None
        coupled_embeds = None
        patch_width = self.visual_meta_net[0].in_features
        for prompt_type in prompt_types:
            if prompt_type == 'static':
                prompt_embeds = self.visual_ctx.detach().cpu().numpy()  # [K, D]
            else:
                # 自动投影并伪造patch维度
                if all_features.shape[1] != patch_width:
                    import torch.nn as nn
                    projection_layer = nn.Linear(all_features.shape[1], patch_width).to(self.visual_ctx.device)
                    nn.init.xavier_uniform_(projection_layer.weight)
                    with torch.no_grad():
                        patch_features_proj = projection_layer(torch.tensor(all_features, dtype=torch.float32).to(self.visual_ctx.device))
                    patch_features = patch_features_proj.unsqueeze(1)  # [N, 1, patch_width]
                else:
                    patch_features = torch.tensor(all_features, dtype=torch.float32).to(self.visual_ctx.device).unsqueeze(1)
            if prompt_type == 'dynamic':
                dynamic_prompts = self.forward(patch_features, None)  # [N, K, D]
                prompt_embeds = dynamic_prompts.detach().cpu().numpy().reshape(-1, dynamic_prompts.shape[-1])  # [N*K, D]
                dynamic_embeds = prompt_embeds
            elif prompt_type == 'coupled':
                if text_ctx is None:
                    if hasattr(self, 'text_ctx') and self.text_ctx is not None:
                        text_ctx = self.text_ctx
                        print("[警告] text_ctx为None，自动使用self.text_ctx")
                    else:
                        # 更改：移除可能存在的硬编码路径或错误信息
                        print("[警告] text_ctx 为 None，无法生成 coupled prompt，将跳过。")
                        continue  # 跳过此 prompt 类型
                if isinstance(text_ctx, np.ndarray):
                    text_ctx_tensor = torch.tensor(text_ctx, dtype=torch.float32).to(self.visual_ctx.device)
                else:
                    text_ctx_tensor = text_ctx.to(self.visual_ctx.device)
                print(f"[DEBUG] text_ctx shape: {text_ctx_tensor.shape}, mean: {text_ctx_tensor.float().mean().item():.6f}, std: {text_ctx_tensor.float().std().item():.6f}, all_zero: {bool((text_ctx_tensor==0).all())}")
                def forward_with_attn(patch_features, text_ctx_tensor):
                    global_patch_feature = patch_features.mean(dim=1)
                    bias = self.visual_meta_net(global_patch_feature)
                    dynamic_visual_ctx = self.visual_ctx.unsqueeze(0) + bias.unsqueeze(1)
                    text_ctx_expanded = text_ctx_tensor.unsqueeze(0).expand(dynamic_visual_ctx.shape[0], -1, -1)
                    projected_text_ctx = self.text_to_vision_projection(text_ctx_expanded)
                    coupled_visual_ctx, attn_weights = self.coupling_layer(
                        query=dynamic_visual_ctx,
                        key=projected_text_ctx,
                        value=projected_text_ctx,
                        need_weights=True,
                        average_attn_weights=False
                    )
                    attn_std = attn_weights.std().item()
                    print(f"[DEBUG] attention weights std: {attn_std:.6f}")
                    final_visual_ctx = dynamic_visual_ctx + coupled_visual_ctx
                    return final_visual_ctx
                coupled_prompts = forward_with_attn(patch_features, text_ctx_tensor)  # [N, K, D]
                prompt_embeds = coupled_prompts.detach().cpu().numpy().reshape(-1, coupled_prompts.shape[-1])  # [N*K, D]
                coupled_embeds = prompt_embeds
            if prompt_type == 'static' or prompt_type == 'dynamic' or prompt_type == 'coupled':
                feat_2d = reduce_dim(all_features, dim_reduce)
                prompt_2d = reduce_dim(prompt_embeds, dim_reduce)
                plt.figure(figsize=(10, 8))
                plt.scatter(feat_2d[:,0], feat_2d[:,1], s=10, c='orange', label='Test Features', alpha=0.6)
                plt.scatter(prompt_2d[:,0], prompt_2d[:,1], s=50, c='black', marker='*', label=f'{prompt_type.capitalize()} Prompts')
                plt.legend()
                plt.title(f'{prompt_type.capitalize()} Prompt {dim_reduce.upper()} Visualization')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'{prompt_type}_prompt_cluster_{dim_reduce}.png'))
                plt.close()
                kmeans = KMeans(n_clusters=self.visual_ctx.shape[0], random_state=42, n_init=10)
                labels = kmeans.fit_predict(feat_2d)
                sil = silhouette_score(feat_2d, labels)
                ch = calinski_harabasz_score(feat_2d, labels)
                db = davies_bouldin_score(feat_2d, labels)
                results[prompt_type] = {'silhouette': sil, 'calinski': ch, 'davies': db}
        metrics = ['silhouette', 'calinski', 'davies']
        for metric in metrics:
            plt.figure()
            plt.bar(results.keys(), [results[t][metric] for t in results])
            plt.title(f'{metric.capitalize()} Score Comparison')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{metric}_bar.png'))
            plt.close()
        if dynamic_embeds is not None and coupled_embeds is not None:
            cos_sims = cosine_similarity(dynamic_embeds, coupled_embeds)
            diag_sims = np.diag(cos_sims)
            plt.figure()
            plt.hist(diag_sims, bins=30, color='blue', alpha=0.7)
            plt.title('Cosine Similarity between Dynamic and Coupled Prompts')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'dynamic_vs_coupled_cosine_hist.png'))
            plt.close()
            print(f'[DEBUG] dynamic-coupled cosine mean: {diag_sims.mean():.4f}, std: {diag_sims.std():.4f}')
        print(f'[可视化] 已保存三类prompt聚类图和指标柱状图到 {save_dir}')

    def visualize_multilabel_clustering(self, all_features, all_labels, all_preds, all_probs, 
                                      class_names=None, save_dir='multilabel_analysis', dim_reduce='pca'):
        """
        多标签分类结果的聚类分析可视化
        all_features: numpy array, test集特征 [N, D]
        all_labels: numpy array, 真实标签 [N, num_classes]
        all_preds: numpy array, 预测标签 [N, num_classes] 
        all_probs: numpy array, 预测概率 [N, num_classes]
        class_names: list, 类别名称列表
        save_dir: 保存目录
        dim_reduce: 降维方法 'pca'|'tsne'|'umap'
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        import os
        import pandas as pd
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 降维方法选择
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            TSNE = None
        try:
            import umap
        except ImportError:
            umap = None
            
        def reduce_dim(X, method='pca'):
            if method == 'tsne' and TSNE is not None:
                return TSNE(n_components=2, random_state=42).fit_transform(X)
            elif method == 'umap' and umap is not None:
                return umap.UMAP(n_components=2, random_state=42).fit_transform(X)
            else:
                return PCA(n_components=2).fit_transform(X)
        
        num_classes = all_labels.shape[1]
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(num_classes)]
        
        print(f"[多标签聚类分析] 开始分析 {all_features.shape[0]} 个样本，{num_classes} 个类别")
        
        # 1. 特征空间聚类分析
        print("1. 特征空间聚类分析...")
        feat_2d = reduce_dim(all_features, dim_reduce)
        
        # 基于特征进行聚类
        n_clusters = min(8, all_features.shape[0] // 10)  # 自适应聚类数
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feat_2d)
        
        # 计算聚类质量指标
        sil_score = silhouette_score(feat_2d, cluster_labels)
        ch_score = calinski_harabasz_score(feat_2d, cluster_labels)
        db_score = davies_bouldin_score(feat_2d, cluster_labels)
        
        # 可视化特征聚类
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(feat_2d[:, 0], feat_2d[:, 1], c=cluster_labels, 
                            cmap='tab10', s=20, alpha=0.7)
        plt.colorbar(scatter, label='Cluster ID')
        plt.title(f'特征空间聚类分析 ({dim_reduce.upper()})\n'
                 f'Silhouette: {sil_score:.3f}, CH: {ch_score:.1f}, DB: {db_score:.3f}')
        plt.xlabel(f'{dim_reduce.upper()} Component 1')
        plt.ylabel(f'{dim_reduce.upper()} Component 2')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_clustering.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 类别分布分析
        print("2. 类别分布分析...")
        
        # 计算每个聚类的类别分布
        cluster_class_dist = np.zeros((n_clusters, num_classes))
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            if np.sum(cluster_mask) > 0:
                cluster_class_dist[i] = np.mean(all_labels[cluster_mask], axis=0)
        
        # 可视化聚类-类别分布热力图
        plt.figure(figsize=(max(12, num_classes*0.8), 8))
        sns.heatmap(cluster_class_dist, annot=True, fmt='.2f', 
                   xticklabels=class_names, yticklabels=[f'Cluster_{i}' for i in range(n_clusters)],
                   cmap='YlOrRd', cbar_kws={'label': 'Average Label Probability'})
        plt.title('聚类-类别分布热力图')
        plt.xlabel('情感类别')
        plt.ylabel('特征聚类')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'cluster_class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 预测置信度分析
        print("3. 预测置信度分析...")
        
        # 计算每个样本的最大置信度
        max_confidences = np.max(all_probs, axis=1)
        mean_confidences = np.mean(all_probs, axis=1)
        
        # 按聚类分析置信度分布
        plt.figure(figsize=(15, 5))
        
        # 最大置信度分布
        plt.subplot(1, 3, 1)
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            if np.sum(cluster_mask) > 0:
                plt.hist(max_confidences[cluster_mask], alpha=0.7, label=f'Cluster_{i}', bins=20)
        plt.xlabel('最大预测置信度')
        plt.ylabel('样本数量')
        plt.title('各聚类最大置信度分布')
        plt.legend()
        
        # 平均置信度分布
        plt.subplot(1, 3, 2)
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            if np.sum(cluster_mask) > 0:
                plt.hist(mean_confidences[cluster_mask], alpha=0.7, label=f'Cluster_{i}', bins=20)
        plt.xlabel('平均预测置信度')
        plt.ylabel('样本数量')
        plt.title('各聚类平均置信度分布')
        plt.legend()
        
        # 置信度vs聚类质量
        plt.subplot(1, 3, 3)
        cluster_sizes = [np.sum(cluster_labels == i) for i in range(n_clusters)]
        cluster_avg_conf = [np.mean(max_confidences[cluster_labels == i]) for i in range(n_clusters)]
        plt.scatter(cluster_sizes, cluster_avg_conf, s=100, alpha=0.7)
        for i in range(n_clusters):
            plt.annotate(f'C{i}', (cluster_sizes[i], cluster_avg_conf[i]), 
                        xytext=(5, 5), textcoords='offset points')
        plt.xlabel('聚类大小')
        plt.ylabel('平均最大置信度')
        plt.title('聚类大小vs置信度')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confidence_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 错误分析
        print("4. 错误分析...")
        
        # 计算预测错误
        prediction_errors = np.abs(all_labels - all_preds)
        false_positives = (all_preds == 1) & (all_labels == 0)
        false_negatives = (all_preds == 0) & (all_labels == 1)
        
        # 按聚类分析错误分布
        plt.figure(figsize=(15, 5))
        
        # 错误率分布
        plt.subplot(1, 3, 1)
        error_rates = []
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            if np.sum(cluster_mask) > 0:
                cluster_errors = np.mean(prediction_errors[cluster_mask])
                error_rates.append(cluster_errors)
                plt.bar(f'C{i}', cluster_errors, alpha=0.7)
        plt.ylabel('平均错误率')
        plt.title('各聚类错误率')
        
        # 假阳性率
        plt.subplot(1, 3, 2)
        fp_rates = []
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            if np.sum(cluster_mask) > 0:
                cluster_fp = np.mean(false_positives[cluster_mask])
                fp_rates.append(cluster_fp)
                plt.bar(f'C{i}', cluster_fp, alpha=0.7, color='red')
        plt.ylabel('假阳性率')
        plt.title('各聚类假阳性率')
        
        # 假阴性率
        plt.subplot(1, 3, 3)
        fn_rates = []
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            if np.sum(cluster_mask) > 0:
                cluster_fn = np.mean(false_negatives[cluster_mask])
                fn_rates.append(cluster_fn)
                plt.bar(f'C{i}', cluster_fn, alpha=0.7, color='orange')
        plt.ylabel('假阴性率')
        plt.title('各聚类假阴性率')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'error_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. 类别预测质量分析
        print("5. 类别预测质量分析...")
        
        # 计算每个类别的预测质量
        class_metrics = {}
        for i in range(num_classes):
            class_preds = all_preds[:, i]
            class_labels = all_labels[:, i]
            class_probs = all_probs[:, i]
            
            # 计算该类别的指标
            tp = np.sum((class_preds == 1) & (class_labels == 1))
            fp = np.sum((class_preds == 1) & (class_labels == 0))
            tn = np.sum((class_preds == 0) & (class_labels == 0))
            fn = np.sum((class_preds == 0) & (class_labels == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[class_names[i]] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': np.sum(class_labels)
            }
        
        # 可视化类别指标
        metrics_df = pd.DataFrame(class_metrics).T
        plt.figure(figsize=(max(12, num_classes*0.6), 8))
        metrics_df[['precision', 'recall', 'f1']].plot(kind='bar', ax=plt.gca())
        plt.title('各类别预测质量指标')
        plt.xlabel('情感类别')
        plt.ylabel('指标值')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'class_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. 生成分析报告
        print("6. 生成分析报告...")
        
        report = {
            '数据集信息': {
                '样本数量': all_features.shape[0],
                '类别数量': num_classes,
                '特征维度': all_features.shape[1]
            },
            '聚类质量': {
                '聚类数量': n_clusters,
                'Silhouette Score': sil_score,
                'Calinski-Harabasz Score': ch_score,
                'Davies-Bouldin Score': db_score
            },
            '预测质量': {
                '平均最大置信度': np.mean(max_confidences),
                '平均置信度': np.mean(mean_confidences),
                '总体错误率': np.mean(prediction_errors),
                '假阳性率': np.mean(false_positives),
                '假阴性率': np.mean(false_negatives)
            },
            '类别性能': class_metrics
        }
        
        # 保存报告
        with open(os.path.join(save_dir, 'analysis_report.txt'), 'w', encoding='utf-8') as f:
            f.write("多标签分类聚类分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            for section, data in report.items():
                f.write(f"{section}:\n")
                f.write("-" * 30 + "\n")
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, dict):
                            f.write(f"  {key}:\n")
                            for sub_key, sub_value in value.items():
                                f.write(f"    {sub_key}: {sub_value:.4f}\n")
                        else:
                            f.write(f"  {key}: {value}\n")
                f.write("\n")
        
        print(f"[多标签聚类分析] 分析完成！结果保存在 {save_dir}")
        print(f"生成的文件:")
        print(f"  - feature_clustering.png: 特征空间聚类图")
        print(f"  - cluster_class_distribution.png: 聚类-类别分布热力图")
        print(f"  - confidence_analysis.png: 置信度分析图")
        print(f"  - error_analysis.png: 错误分析图")
        print(f"  - class_performance.png: 类别性能图")
        print(f"  - analysis_report.txt: 详细分析报告")
        
        return report

class VisualClusterInitializer:
    """使用视觉patch聚类中心初始化视觉prompt的工具类"""
    def __init__(self, n_clusters=4, feature_extractor='resnet'):
        self.n_clusters = n_clusters
        self.feature_extractor = feature_extractor
        self.cluster_centers = None
        
    def extract_visual_features(self, dataloader, device='cuda'):
        """从训练数据中提取视觉特征"""
        print("Extracting visual features for clustering...")
        
        # 使用预训练的ResNet提取特征
        if self.feature_extractor == 'resnet':
            feature_model = resnet50(pretrained=True)
            feature_model = nn.Sequential(*list(feature_model.children())[:-1])  # 移除最后的分类层
            feature_model.eval()
            feature_model.to(device)
        
        all_features = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > 100:  # 限制样本数量以节省时间
                    break
                    
                images = batch['img'].to(device)
                
                if self.feature_extractor == 'resnet':
                    features = feature_model(images)
                    features = features.view(features.size(0), -1)  # flatten
                
                all_features.append(features.cpu().numpy())
                
                if (batch_idx + 1) % 20 == 0:
                    print(f"Processed {batch_idx + 1} batches")
        
        if not all_features:
            print("No features extracted, using random initialization")
            return None
            
        all_features = np.concatenate(all_features, axis=0)
        print(f"Extracted features shape: {all_features.shape}")
        
        return all_features
    
    def perform_clustering(self, features):
        """对特征进行K-means聚类"""
        if features is None:
            return None
            
        print(f"Performing K-means clustering with {self.n_clusters} clusters...")
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        kmeans.fit(features)
        
        self.cluster_centers = kmeans.cluster_centers_
        print(f"Clustering completed. Cluster centers shape: {self.cluster_centers.shape}")
        
        return self.cluster_centers
    
    def initialize_visual_prompts(self, dataloader, target_dim=768, device='cuda', cache_file=None):
        """
        初始化视觉prompts
        
        Args:
            dataloader: 训练数据加载器
            target_dim: 目标维度（ViT的特征维度）
            device: 计算设备
            cache_file: 缓存文件路径
        
        Returns:
            初始化后的视觉prompt tensor
        """
        # 尝试从缓存加载
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cluster centers from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                self.cluster_centers = pickle.load(f)
        else:
            # 提取特征并聚类
            features = self.extract_visual_features(dataloader, device)
            self.cluster_centers = self.perform_clustering(features)
            
            # 保存到缓存
            if cache_file and self.cluster_centers is not None:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.cluster_centers, f)
                print(f"Saved cluster centers to cache: {cache_file}")
        
        if self.cluster_centers is None:
            print("Using random initialization for visual prompts")
            return torch.randn(1, self.n_clusters, target_dim) * 0.02
        
        # 转换为torch tensor并投影到目标维度
        cluster_tensor = torch.tensor(self.cluster_centers, dtype=torch.float32)
        
        if cluster_tensor.shape[1] != target_dim:
            # 投影到目标维度（仅用于初始化，不参与训练）
            projection = nn.Linear(cluster_tensor.shape[1], target_dim, bias=False)
            with torch.no_grad():
                nn.init.xavier_uniform_(projection.weight)
                cluster_tensor = projection(cluster_tensor)
        
        # 添加batch维度
        cluster_tensor = cluster_tensor.unsqueeze(0)  # [1, n_clusters, target_dim]
        
        return cluster_tensor
