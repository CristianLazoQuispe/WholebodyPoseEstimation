from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch
import os
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    classification_report,
    precision_recall_fscore_support
)
import numpy as np
from collections import defaultdict
import torch
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from collections import defaultdict
import torch.nn.utils as nn_utils





def training_pipeline(args,model,train_loader, val_loader, test_loader, device):

    # === Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)

    # === Configuración
    EPOCHS = args.epochs
    PATIENCE = args.patience
    BEST_MODEL_PATH = args.best_model_path
    os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)

    if args.train:
        print("🚀 Training mode activated")
        best_val_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(EPOCHS):
            model.train()
            train_loss, train_preds, train_targets = 0.0, [], []

            loop = tqdm(train_loader, total=len(train_loader), ncols=100, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
            for x, y, mask, _ in loop:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                optimizer.zero_grad()
                out = model(x, mask)
                loss = criterion(out, y)
                loss.backward()
                # Clip gradients: prevent exploding gradients
                nn_utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                train_loss += loss.item()
                train_preds.extend(out.argmax(dim=1).cpu().numpy())
                train_targets.extend(y.cpu().numpy())

                acc = accuracy_score(train_targets, train_preds)
                loop.set_postfix(loss=loss.item(), acc=acc)

            # === Validation
            model.eval()
            val_loss, val_preds, val_targets = 0.0, [], []
            with torch.no_grad():
                val_loop = tqdm(val_loader, total=len(val_loader), ncols=100, desc=f"Epoch {epoch+1}/{EPOCHS} [Val  ]")
                for x, y, mask, _ in val_loop:
                    x, y, mask = x.to(device), y.to(device), mask.to(device)
                    out = model(x, mask)
                    loss = criterion(out, y)

                    val_loss += loss.item()
                    val_preds.extend(out.argmax(dim=1).cpu().numpy())
                    val_targets.extend(y.cpu().numpy())

                    acc_val = accuracy_score(val_targets, val_preds)
                    val_loop.set_postfix(loss=loss.item(), acc=acc_val)

            epoch_val_acc = accuracy_score(val_targets, val_preds)

            # === Early stopping check
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                epochs_no_improve = 0
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"✅ New best model saved! Val Acc: {best_val_acc:.4f}")
            else:
                epochs_no_improve += 1
                print(f"⏳ No improvement for {epochs_no_improve} epochs")

            if epochs_no_improve >= PATIENCE:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break

    # === Final Test
    print("📦 Loading best model...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    test_loss, test_preds, test_targets, test_logits, test_labels_str = 0.0, [], [], [], []

    with torch.no_grad():
        test_loop = tqdm(test_loader, total=len(test_loader), ncols=100, desc="[TEST]")
        for x, y, mask, names in test_loop:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            out = model(x, mask)
            loss = criterion(out, y)

            test_loss += loss.item()
            test_preds.extend(out.argmax(dim=1).cpu().numpy())
            test_targets.extend(y.cpu().numpy())
            test_logits.append(out.cpu())
            test_labels_str.extend(names)

            acc_test = accuracy_score(test_targets, test_preds)
            test_loop.set_postfix(loss=loss.item(), acc=acc_test)

    # === Metrics


    # === Prepare outputs
    logits_all = torch.cat(test_logits, dim=0)  # [N, C]
    probs_all = torch.softmax(logits_all, dim=1).cpu().numpy()  # [N, C]
    targets_np = np.array(test_targets)
    preds_top1 = np.argmax(probs_all, axis=1)

    # === Per-instance Top-1 and Top-5 accuracy
    acc_top1 = accuracy_score(targets_np, preds_top1)
    acc_top5 = top_k_accuracy_score(targets_np, probs_all, k=5)

    print(f"\n✅ [Test] Per-instance Top-1 Accuracy: {acc_top1:.4f}")
    print(f"🎯 [Test] Per-instance Top-5 Accuracy: {acc_top5:.4f}")

    # === Per-class Top-1 accuracy using classification report
    report = classification_report(
        targets_np, preds_top1, digits=4, zero_division=0, output_dict=True
    )
    per_class_top1_acc = np.mean([
        v['recall'] for k, v in report.items() if k.isdigit()
    ])
    print(f"📊 [Test] Per-class Top-1 Accuracy (mean recall): {per_class_top1_acc:.4f}")

    # === Per-class Top-5 accuracy (manual)
    per_class_top5 = defaultdict(list)
    for i, true in enumerate(targets_np):
        top5 = np.argsort(probs_all[i])[-5:][::-1]
        per_class_top5[true].append(int(true in top5))
    mean_per_class_top5 = np.mean([np.mean(v) for v in per_class_top5.values()])
    print(f"📊 [Test] Per-class Top-5 Accuracy: {mean_per_class_top5:.4f}")

    # === Print sample predictions (Top-5)
    print("\n📌 Example Top-5 Predictions (first 5 samples):")
    for i in range(min(5, len(targets_np))):
        top5 = np.argsort(probs_all[i])[-5:][::-1]
        truth = train_loader.dataset.map_labels['label_to_id'][str(targets_np[i])]
        top5_labels = [train_loader.dataset.map_labels['label_to_id'][str(idx)] for idx in top5]
        print(f"[{i}] Ground Truth: {truth} | Top-5 Preds: {top5_labels}")
