"""
Mean Teacher Trainer
────────────────────
Training loop:
  1. Sample batch (image, auxiliary, mask)
  2. StyCona augment: image + auxiliary → augmented
  3. Generate views: augmented → strong_view, weak_view
  4. Student(strong_view) → student_pred
  5. Teacher(weak_view)   → teacher_pred  (no grad)
  6. L = L_sup(student_pred, mask) + λ(t) · L_cons(student_pred, teacher_pred)
  7. Backprop → update student θ
  8. EMA → update teacher ξ
"""

import os
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

from models.ema import EMAModel
from losses.losses import SupervisedLoss, ConsistencyLoss, consistency_weight
from utils.metrics import compute_metrics, collapse_stats
from utils.helpers import save_checkpoint, AverageMeter


class MeanTeacherTrainer:

    def __init__(self, cfg: dict, student: nn.Module, device: torch.device):
        self.cfg = cfg
        self.device = device

        # ── Student & Teacher ────────────────
        self.student = student.to(device)
        self.teacher = EMAModel(student, decay=cfg["mean_teacher"]["ema_decay"]).to(device)


        # ── Losses ───────────────────────────
        lcfg = cfg["loss"]
        self.sup_loss = SupervisedLoss(
            focal_weight=lcfg["supervised"]["focal_weight"],
            dice_weight=lcfg["supervised"]["dice_weight"],
            l1_weight=lcfg["supervised"].get("l1_weight", 0.5),
            focal_gamma=lcfg["supervised"].get("focal_gamma", 2.0),
            focal_alpha=lcfg["supervised"].get("focal_alpha", None),
        )
        self.cons_loss = ConsistencyLoss(loss_type=lcfg["consistency"]["type"])

        # ── Optimiser & scheduler ────────────
        tcfg = cfg["training"]
        self.optimizer = Adam(
            self.student.parameters(),
            lr=tcfg["lr"],
            weight_decay=tcfg["weight_decay"],
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=tcfg["epochs"] - tcfg.get("warmup_epochs", 0),
        )

        # ── Book-keeping ─────────────────────
        self.epochs = tcfg["epochs"]
        self.save_dir = Path(tcfg["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_dice = 0.0
        self.save_every = tcfg.get("save_every", 0)
        self._last_val_dice = float("nan")
        self._last_val_iou = float("nan")

        log_dir = Path(tcfg.get("log_dir", "./logs"))
        log_dir.mkdir(parents=True, exist_ok=True)

        # CSV log
        self.csv_path = log_dir / "train_log.csv"
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_dice", "val_iou",
                                    "pred_entropy", "pred_fg_rate"])

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(log_dir)) if _TB_AVAILABLE else None

    # ──────────────────────────────────────────

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        val_every = self.cfg["training"].get("val_every", 5)

        for epoch in range(1, self.epochs + 1):
            train_loss, ent, fg = self._train_one_epoch(train_loader, epoch)
            self.scheduler.step()

            if epoch % val_every == 0:
                metrics = self._validate(val_loader)
                self._last_val_dice = metrics["dice"]
                self._last_val_iou = metrics["iou"]

                if self._last_val_dice > self.best_dice:
                    self.best_dice = self._last_val_dice
                    save_checkpoint(
                        self.student, self.optimizer, epoch,
                        self.save_dir / "best.pth",
                    )

            # ── Per-epoch save ───────────────
            if self.save_every and epoch % self.save_every == 0:
                save_checkpoint(
                    self.student, self.optimizer, epoch,
                    self.save_dir / f"epoch_{epoch:04d}.pth",
                )

            # ── Console log ─────────────────
            val_str = f"  val Dice={self._last_val_dice:.4f}  IoU={self._last_val_iou:.4f}" if epoch % val_every == 0 else ""
            print(f"[Epoch {epoch:3d}/{self.epochs}]  loss={train_loss:.4f}  H={ent:.3f}  fg={fg:.4f}{val_str}")

            # ── CSV log — val columns carry the last computed value ──
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, f"{train_loss:.6f}",
                                        f"{self._last_val_dice:.4f}",
                                        f"{self._last_val_iou:.4f}",
                                        f"{ent:.4f}", f"{fg:.5f}"])

            # ── TensorBoard ──────────────────
            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Collapse/pred_entropy", ent, epoch)
                self.writer.add_scalar("Collapse/pred_fg_rate", fg, epoch)
                if epoch % val_every == 0:
                    self.writer.add_scalar("Val/Dice", self._last_val_dice, epoch)
                    self.writer.add_scalar("Val/IoU", self._last_val_iou, epoch)

    # ──────────────────────────────────────────

    def _train_one_epoch(self, loader: DataLoader, epoch: int) -> tuple[float, float, float]:
        self.student.train()
        self.teacher.model.eval()

        mt_cfg = self.cfg["mean_teacher"]
        lam = consistency_weight(
            epoch, mt_cfg["consistency_weight_max"], mt_cfg["consistency_rampup_epochs"]
        )

        loss_meter = AverageMeter()
        # Collapse detectors: entropy catches flat-uncertain, fg_rate catches all-background.
        ent_meter, fg_meter = AverageMeter(), AverageMeter()
        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)

        for batch in pbar:
            strong = batch["strong_view"].to(self.device)  # (B, 3, H, W)
            weak = batch["weak_view"].to(self.device)       # (B, 3, H, W)
            masks = batch["mask"].to(self.device)           # (B, H, W)

            # ── Forward ──────────────────────
            student_out = self.student(strong)
            with torch.no_grad():
                teacher_out = self.teacher(weak)

            # ── Loss ─────────────────────────
            l_sup = self.sup_loss(student_out, masks)
            l_cons = self.cons_loss(student_out, teacher_out)
            loss = l_sup + lam * l_cons

            # ── Backward ─────────────────────
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ── EMA update ───────────────────
            self.teacher.update(self.student)

            with torch.no_grad():
                ent, fg = collapse_stats(student_out)

            n = strong.shape[0]
            loss_meter.update(loss.item(), n)
            ent_meter.update(ent, n)
            fg_meter.update(fg, n)
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", lam=f"{lam:.3f}",
                             H=f"{ent_meter.avg:.3f}", fg=f"{fg_meter.avg:.4f}")

        return loss_meter.avg, ent_meter.avg, fg_meter.avg

    # ──────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict:
        self.student.eval()
        all_preds, all_masks = [], []

        for batch in loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"]

            logits = self.student(images)
            preds = logits.argmax(dim=1).cpu()

            all_preds.append(preds)
            all_masks.append(masks)

        all_preds = torch.cat(all_preds)
        all_masks = torch.cat(all_masks)
        return compute_metrics(all_preds, all_masks, num_classes=self.cfg["model"]["num_classes"])
