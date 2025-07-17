import torch
from torch.nn.utils.rnn import pad_sequence

PAD_IDX = 0
MAX_SEQ_TRAINING = 300  

def apply_padding(keypoints):
    keypoints = pad_sequence(keypoints, batch_first=True, padding_value=PAD_IDX)
    keypoints = keypoints.flatten(start_dim=2).float()  # [B, T, 270]
    if keypoints.shape[1] > MAX_SEQ_TRAINING:
        keypoints = keypoints[:, :MAX_SEQ_TRAINING, :]
    padding_mask = (keypoints == PAD_IDX).all(dim=2)  # [B, T]
    return keypoints, padding_mask

def collate_fn(batch):
    keypoints, names, labels, _ = zip(*batch)  # each keypoint: [T, 135, 2]
    keypoints, padding_mask = apply_padding(list(keypoints))  # [B, T, 270], [B, T]
    labels = torch.tensor(labels, dtype=torch.long)
    return keypoints, labels, padding_mask, names