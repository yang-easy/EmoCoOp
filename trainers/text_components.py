import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gensim.downloader as api
import numpy as np
from clip import clip
from gensim.models import KeyedVectors

class GloVeInitializer:
    """使用GloVe向量初始化文本prompt的工具类"""
    def __init__(self, glove_path=None):
        self.glove_model = None
        # 优先从环境变量 GLOVE_PATH 读取路径，如果未设置，则使用一个占位符路径
        # 用户在运行前应设置此环境变量，或在代码中直接提供路径
        self.glove_path = glove_path or os.getenv("GLOVE_PATH", "path/to/your/glove-wiki-gigaword-300.kv")
        self.embedding_dim = 300  # GloVe 词向量维度

    def load_glove_model(self):
        """加载本地GloVe模型"""
        if self.glove_model is None:
            print(f"Loading GloVe model from local path: {self.glove_path}")
            if os.path.exists(self.glove_path):
                try:
                    self.glove_model = KeyedVectors.load(self.glove_path, mmap='r')
                    print("GloVe model loaded successfully")
                except Exception as e:
                    print(f"Failed to load GloVe model: {e}")
                    print("Will use random initialization instead")
                    self.glove_model = None
            else:
                print(f"GloVe model file not found at: {self.glove_path}")
                self.glove_model = None

    def get_word_embedding(self, word):
        """获取单词的GloVe嵌入"""
        if self.glove_model is None:
            return None

        word = word.lower().strip()
        
        if word in self.glove_model:
            return self.glove_model[word]
        else:
            variants = [
                word.replace('_', ''),
                word.replace('-', ''),
            ]
            for variant in variants:
                if variant in self.glove_model:
                    return self.glove_model[variant]
            print(f"Word '{word}' not found in GloVe model")
            return None

    def initialize_prompt_embeddings(self, prompt_text, target_dim=512):
        """
        使用GloVe向量初始化prompt embeddings

        Args:
            prompt_text: 初始化的文本
            target_dim: 目标维度（如CLIP的文本嵌入维度）

        Returns:
            初始化后的embedding tensor
        """
        self.load_glove_model()
        words = prompt_text.split()
        embeddings = []

        for word in words:
            glove_emb = self.get_word_embedding(word)
            if glove_emb is not None:
                embeddings.append(glove_emb)
            else:
                embeddings.append(np.random.normal(0, 0.02, self.embedding_dim))

        if not embeddings:
            print("No valid embeddings found, using random initialization")
            return torch.randn(len(words), target_dim) * 0.02

        glove_embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32)

        if self.embedding_dim != target_dim:
            projection = nn.Linear(self.embedding_dim, target_dim, bias=False)
            with torch.no_grad():
                nn.init.xavier_uniform_(projection.weight)
                glove_embeddings = projection(glove_embeddings)

        return glove_embeddings

class TextPromptLearner(nn.Module):
    """增强的文本提示学习器 - 支持GloVe初始化"""
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.EMOCOOP.N_CTX
        ctx_init = cfg.TRAINER.EMOCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        
        # 新增 meta_net 定义
        hidden_dim = max(clip_model.visual.output_dim // 32, 64)
        self.meta_net = nn.Sequential(
            nn.Linear(clip_model.visual.output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, ctx_dim)
        )
        
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # 初始化GloVe工具
        self.glove_initializer = GloVeInitializer()
        
        # 修复：正确处理CTX_INIT列表
        if ctx_init:
            if isinstance(ctx_init, list):
                ctx_init = ctx_init[0]
            
            if n_ctx <= 10:
                ctx_init = ctx_init.replace("_", " ")
                
                # 使用GloVe初始化（如果可用）
                print(f"Attempting GloVe initialization for context: '{ctx_init}'")
                glove_embeddings = self.glove_initializer.initialize_prompt_embeddings(
                    ctx_init, target_dim=ctx_dim
                )
                
                if glove_embeddings.shape[0] >= n_ctx:
                    # 使用GloVe嵌入
                    ctx_vectors = glove_embeddings[:n_ctx]
                    print(f"Successfully initialized {n_ctx} context vectors with GloVe")
                else:
                    # GloVe嵌入不够，补充随机初始化
                    print(f"GloVe embeddings ({glove_embeddings.shape[0]}) < n_ctx ({n_ctx}), using hybrid initialization")
                    ctx_vectors = torch.zeros(n_ctx, ctx_dim, dtype=dtype)
                    ctx_vectors[:glove_embeddings.shape[0]] = glove_embeddings
                    # 剩余部分随机初始化
                    remaining = n_ctx - glove_embeddings.shape[0]
                    ctx_vectors[glove_embeddings.shape[0]:] = torch.empty(remaining, ctx_dim, dtype=dtype)
                    nn.init.normal_(ctx_vectors[glove_embeddings.shape[0]:], std=0.02)
                
                prompt_prefix = ctx_init
            else:
                # n_ctx > 10，使用随机初始化
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)
        else:
            # 随机初始化
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(clip._tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx
        bias = self.meta_net(im_features)
        bias = bias.unsqueeze(1)
        ctx = ctx.unsqueeze(0)
        ctx_shifted = ctx + bias
        
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts
