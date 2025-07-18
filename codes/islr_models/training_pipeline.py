import os
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from training.dataset.dataloader import SimpleHDF5Dataset
from training.models.util import collate_fn
from training.models.gcn_bert.util import collate_fn_gcnbert
from training.models.pose_tgcn.pose_tgcn import PoseGTCN
from training.models.gcn_bert.gcn_bert import GCN_BERT
from training.models.spoter.spoter import SPOTER
from training.models.silt.silt import SILT
from training.train import training_pipeline
# === Paths
train_path     = "/data/cristian/paper_2025/Testing/ISLR/WLASL/WLASL100/WLASL100_135-Train.hdf5"
val_path       = "/data/cristian/paper_2025/Testing/ISLR/WLASL/WLASL100/WLASL100_135-Val.hdf5"
test_path      = "/data/cristian/paper_2025/Testing/ISLR/WLASL/WLASL100/WLASL100_135-Test.hdf5"
map_label_path = "/data/cristian/paper_2025/Testing/ISLR/WLASL/WLASL100/wlasl_100_maplabels.json"


if __name__ == "__main__":
    import argparse
    # === Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run training loop")
    parser.add_argument("--model_name", type=str, default="gcn_bert", help="Name of the model to train")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience")
    parser.add_argument("--best_model_path", type=str, default="../../results/models/gcn_bert/wlasl_best.pth", help="Path to save the best model")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use for training (cuda or cpu)")
    args = parser.parse_args()
    collate_fn_final = collate_fn  # Default collate function
    if args.model_name == "gcn_bert":
        collate_fn_final = collate_fn_gcnbert
        model = GCN_BERT(num_classes=100, hidden_features=2, seq_len=50, num_joints=135,nhead=5)
    elif args.model_name == "spoter":
        model = SPOTER(num_classes=100, hidden_dim=270, num_heads=3, num_layers_1=3, num_layers_2=3,
                     dim_feedforward_encoder=64, dim_feedforward_decoder=64, dropout=0.3, norm_first=True, batch_first=True) # v2

        #model = SPOTER(num_classes=100, hidden_dim=270, num_heads=3, num_layers_1=3, num_layers_2=3,
        #             dim_feedforward_encoder=256, dim_feedforward_decoder=256, dropout=0.3, norm_first=True, batch_first=True) # v3

        #model = SPOTER(num_classes=100, hidden_dim=270, num_heads=6, num_layers_1=6, num_layers_2=6,
        #             dim_feedforward_encoder=1024, dim_feedforward_decoder=1024, dropout=0.3, norm_first=True, batch_first=True)
    elif args.model_name == "silt":
        model = SILT(num_classes=100, hidden_dim=270, num_heads=3, num_layers_1=3, num_layers_2=3,
                     dim_feedforward_encoder=64, dim_feedforward_decoder=64, dropout=0.3, norm_first=True, batch_first=True)
    elif args.model_name == "pose_tgcn":
        model = PoseGTCN(input_feature=100, num_joints=135, hidden_feature=32, num_class=100, p_dropout=0.5, num_stage=20, is_resi=True)
        collate_fn_final = collate_fn_gcnbert
    else:
        raise ValueError(f"Model {args.model_name} not supported") 
       
    # === Loaders
    train_loader = DataLoader(SimpleHDF5Dataset(train_path,map_label_path,augmentation=True,noise_std=0.001), batch_size=64, shuffle=True, collate_fn=collate_fn_final)
    val_loader   = DataLoader(SimpleHDF5Dataset(val_path,map_label_path),   batch_size=64, collate_fn=collate_fn_final)
    test_loader  = DataLoader(SimpleHDF5Dataset(test_path,map_label_path),   batch_size=64, collate_fn=collate_fn_final)

    # === Device
    print(f"Using device: {args.device}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your PyTorch installation or use CPU.")
    else:
        device = torch.device(args.device)
    model = model.to(device)
    print(f"Using device: {device}")
    training_pipeline(args,model,train_loader, val_loader, test_loader, device)

    #python training_pipeline.py --train --model_name="gcn_bert" --epochs=2000 --best_model_path="../../results/models/gcn_bert/wlasl_bestv2.pth" --patience=500 --device="cuda:3"
    #python training_pipeline.py --train --model_name="spoter" --epochs=2000 --best_model_path="../../results/models/spoter/wlasl_best.pth"  --patience=500
    #python training_pipeline.py --train --model_name="silt" --epochs=2000 --best_model_path="../../results/models/silt/wlasl_best.pth" --device="cuda:5"  --patience=500   
    #python training_pipeline.py --train --model_name="pose_tgcn" --epochs=2000 --best_model_path="../../results/models/pose_tgcn/wlasl_best.pth" --device="cuda:4"  --patience=500

    