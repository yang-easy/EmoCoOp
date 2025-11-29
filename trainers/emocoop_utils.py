import torch
import torch.nn as nn
import torch.nn.functional as F

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = cfg.clip._MODELS[backbone_name] if hasattr(cfg, 'clip') else None
    if url is None:
        from clip import clip
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)
    else:
        model_path = cfg.clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    from clip import clip
    model = clip.build_model(state_dict or model.state_dict())
    return model

def optimize_cfg_for_memory(cfg):
    """优化配置以节省显存"""
    cfg.defrost()
    cfg.TRAINER.EMOCOOP.N_VISUAL_CTX = min(cfg.TRAINER.EMOCOOP.N_VISUAL_CTX, 4)
    cfg.TRAINER.EMOCOOP.VPT_DEEP = False
    if cfg.DATALOADER.TRAIN_X.BATCH_SIZE > 16:
        cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 16
    if cfg.DATALOADER.TEST.BATCH_SIZE > 32:
        cfg.DATALOADER.TEST.BATCH_SIZE = 32
    cfg.TRAINER.EMOCOOP.PREC = "amp"
    cfg.freeze()
    print("Configuration optimized for memory usage:")
    print(f"  - N_VISUAL_CTX: {cfg.TRAINER.EMOCOOP.N_VISUAL_CTX}")
    print(f"  - VPT_DEEP: {cfg.TRAINER.EMOCOOP.VPT_DEEP}")
    print(f"  - Train batch size: {cfg.DATALOADER.TRAIN_X.BATCH_SIZE}")
    print(f"  - Test batch size: {cfg.DATALOADER.TEST.BATCH_SIZE}")
    print(f"  - Precision: {cfg.TRAINER.EMOCOOP.PREC}")

def forward_with_debug(self, image, label=None):
    print(f"Input image shape: {image.shape}")
    logit_scale = self.logit_scale.exp()
    image_features = self.image_encoder(image.type(self.dtype))
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    print(f"Image features shape: {image_features.shape}")
    prompts = self.prompt_learner(image_features)
    print(f"Prompts shape: {prompts.shape}")
    logits = []
    for i, (pts_i, imf_i) in enumerate(zip(prompts, image_features)):
        print(f"Processing batch {i}: pts_i shape = {pts_i.shape}")
        if len(pts_i.shape) == 3:
            pts_i = pts_i.squeeze(0)
            print(f"After squeeze: pts_i shape = {pts_i.shape}")
        text_features = self.encode_text(pts_i)
        print(f"Text features shape: {text_features.shape}")
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        l_i = logit_scale * imf_i @ text_features.t()
        logits.append(l_i)
    logits = torch.stack(logits)
    print(f"Final logits shape: {logits.shape}")
    if self.training and label is not None:
        print(f"Label shape: {label.shape}")
        if len(label.shape) > 1:
            return F.binary_cross_entropy_with_logits(logits, label.float())
        else:
            return F.cross_entropy(logits, label)
    return logits
