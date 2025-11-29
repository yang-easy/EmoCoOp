from . import emotion_dataset
from .emotion_dataset import EmotionDataset

DATASET_REGISTRY = {
    "emotion_dataset": EmotionDataset
}