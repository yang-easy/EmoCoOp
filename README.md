# EmoCoOp: Dynamic Vision-Language Coupling for Multi-Label Emotion Classification in Dongba Paintings

**作者**  
Rongyi Yang（杨荣懿）  
邮箱: 12024115003@stu.ynu.edu.cn  
Wenhua Qian（钱文华，通讯作者）  
邮箱: whqian@ynu.edu.cn  
Peng Liu（刘朋）  
邮箱: pengliu0606@gmail.com  

单位：云南大学信息学院，昆明 650504，中国

本仓库为论文“EmoCoOp: Dynamic Vision-Language Coupling for Multi-Label Emotion Classification in Dongba Paintings”配套代码。

> 论文尚未公开发表，暂不提供预印本或引用信息。

## 项目简介
EmoCoOp 是一个面向东巴绘画多标签情感分类的视觉-语言模型，基于CLIP框架，融合了动态视觉提示、辅助情绪头、自适应损失权重等机制，支持冷暖色调分析、深度视觉特征聚类、GloVe文本初始化等功能。

## 主要特性
- **动态视觉提示学习（VPT）**：动态生成视觉prompt，支持聚类初始化和V-T耦合。
- **辅助情绪分类头**：提升情感识别能力，辅助主任务优化。
- **自适应损失权重**：根据验证集表现动态调整辅助损失权重。
- **GloVe初始化**：文本prompt可用GloVe词向量初始化，提升语义泛化。
- **冷暖色调分析（可选）**：支持色彩情感特征分析（如需融合请参考`visual_components.py`）。

## 训练参数说明
训练模型时需指定以下主要参数：
- `--dataset-config-file`：数据集配置文件（如`configs/datasets/emotion_dataset.yaml`）。
- `--config-file`：训练器配置文件（如`configs/trainers/CoCoOp/vit_emotion.yaml`）。
- `--trainer`：Trainer名称，推荐使用`EmoCoOp`（如需兼容原CoCoOp也可用`CoCoOp`）。
- `--output-dir`：输出目录。
- 其他可选参数：`--eval-only`（仅评估），`--model-dir`（加载模型目录），`--load-epoch`（加载指定epoch权重）。

## 训练模型示例
```bash
python train.py \
  --dataset-config-file configs/datasets/emotion_dataset.yaml \
  --config-file configs/trainers/CoCoOp/vit_emotion.yaml \
  --trainer EmoCoOp \
  --output-dir output/emocoop
```

## 复现/评估模型示例
```bash
python train.py \
  --dataset-config-file configs/datasets/emotion_dataset.yaml \
  --config-file configs/trainers/CoCoOp/vit_emotion.yaml \
  --trainer EmoCoOp \
  --output-dir output/emocoop-120 \
  --eval-only \
  --model-dir output/emocoop-120 \
  --load-epoch 80
```


## 依赖环境
- Python >= 3.8
- PyTorch >= 1.10
- torchvision, scikit-learn, pandas, numpy, gensim 等
- 详见`requirements.txt`

## 目录结构简述
- `train.py`：主训练入口
- `trainers/`：核心模型与训练器实现（如`cocoop.py`、`visual_components.py`等）
- `configs/`：数据集与训练器配置
- `output/`：训练与评估结果输出

## 参考与致谢
本项目部分实现参考了CLIP、CoOp/CoCoOp等开源工作，感谢原作者贡献。

## EmoCoOp: Brief English Description

Dongba paintings, a unique Naxi cultural heritage, present complex symbolism and diverse visual aesthetics. Existing vision–language models struggle with domain generalization in few-shot multi-label emotion classification for these artworks. To address this, we propose EmoCoOp—a dynamic vision-language prompt coupling framework. EmoCoOp leverages latent visual priors to guide prompt learning and utilizes a meta-network for adaptive visual prompt generation. A vision-text prompt coupling mechanism enables deep multimodal integration, while a dual-path chromatic affective inference module models both coloration and emotional expression. Experiments show that EmoCoOp achieves state-of-the-art results (75.73% mAP, 82.21% Recall@2), significantly surpassing previous methods. This framework advances multi-label emotion classification for ethnic artworks and cross-modal understanding.

## Availability of Supporting Data

The raw data of this study, which concern Dongba paintings and involve ethnic cultural particularity as well as group privacy, are not publicly available due to ethical considerations related to the protection of cultural heritage and privacy.

---
如有问题欢迎提交issue或联系作者。
