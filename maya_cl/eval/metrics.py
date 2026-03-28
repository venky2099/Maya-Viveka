# metrics.py â€” AA, BWT, FWT computation for CIL benchmark reporting
# These are the exact metrics used by TACOS and NACA â€” ensures fair comparison

import torch
import numpy as np


class CLMetrics:
    """
    Tracks accuracy matrix R where R[i][j] = accuracy on task j after training on task i.
    From this matrix computes:
      AA  â€” Average Accuracy (mean accuracy across all tasks after final task)
      BWT â€” Backward Transfer (mean forgetting of prior tasks)
      FWT â€” Forward Transfer (mean zero-shot improvement on future tasks)
    """

    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks
        # R[i][j]: accuracy on task j evaluated after training task i
        self.R = np.full((num_tasks, num_tasks), fill_value=np.nan)

    def update(self, trained_up_to: int, task_id: int, accuracy: float) -> None:
        # called after each eval pass
        self.R[trained_up_to][task_id] = accuracy

    def average_accuracy(self) -> float:
        # AA = mean of last row (accuracy on all tasks after training all tasks)
        last_row = self.R[self.num_tasks - 1]
        valid = last_row[~np.isnan(last_row)]
        return float(np.mean(valid)) if len(valid) > 0 else 0.0

    def backward_transfer(self) -> float:
        """
        BWT = (1 / T-1) * sum_{j=0}^{T-2} [ R[T-1][j] - R[j][j] ]
        Negative BWT = forgetting. Positive = backward knowledge transfer.
        """
        bwt_vals = []
        for j in range(self.num_tasks - 1):
            if not np.isnan(self.R[self.num_tasks - 1][j]) and not np.isnan(self.R[j][j]):
                bwt_vals.append(self.R[self.num_tasks - 1][j] - self.R[j][j])
        return float(np.mean(bwt_vals)) if bwt_vals else 0.0

    def forward_transfer(self) -> float:
        """
        FWT = (1 / T-1) * sum_{i=1}^{T-1} [ R[i-1][i] - R_random ]
        r_random = 0.01 for 10-class CIFAR-10 (random baseline).
        """
        r_random = 0.01
        fwt_vals = []
        for i in range(1, self.num_tasks):
            if not np.isnan(self.R[i - 1][i]):
                fwt_vals.append(self.R[i - 1][i] - r_random)
        return float(np.mean(fwt_vals)) if fwt_vals else 0.0

    def summary(self) -> dict:
        return {
            "AA":  round(self.average_accuracy() * 100, 2),
            "BWT": round(self.backward_transfer() * 100, 2),
            "FWT": round(self.forward_transfer() * 100, 2),
        }

    def print_matrix(self) -> None:
        print("\nAccuracy Matrix R[trained_up_to][task_id]:")
        header = "       " + "  ".join([f"T{j}" for j in range(self.num_tasks)])
        print(header)
        for i in range(self.num_tasks):
            row = f"After T{i}:  "
            for j in range(self.num_tasks):
                val = self.R[i][j]
                row += f"{val*100:5.1f}  " if not np.isnan(val) else "  ---  "
            print(row)


def evaluate_task(model, loader, device: torch.device,
                  encoder, t_steps: int,
                  task_classes: list = None) -> float:
    """
    TIL mode: task_classes provided â€” restrict prediction to those 2 classes.
    CIL mode: task_classes=None â€” predict across all 10 classes.
    """
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            spike_seq = encoder(images)
            model.reset()
            logits = model(spike_seq)          # [B, 10]

            if task_classes is not None:
                # TIL: mask all non-task logits to -inf
                # model only chooses between the 2 valid classes
                mask = torch.full_like(logits, float('-inf'))
                mask[:, task_classes] = logits[:, task_classes]
                preds = mask.argmax(dim=1)
            else:
                # CIL: full 10-class prediction, no hint
                preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    model.train()
    return correct / total if total > 0 else 0.0
