"""
Spectral-Directional Consistency Trainer
────────────────────────────────────────
Directional supervision over StyCona's SVD axes (see spectral_direction.html):

  x_style   = blend sigma only  → appearance changes, structure does not
                                → HARD INVARIANCE:  D(f(x), teacher(x_style))
  x_content = mix U,V only      → structure changes
                                → plain supervision against y (no invariance)

  L = L_sup + lam_c * L_cont + lam_s(t) * L_style

EMA teacher supplies the style target: stop-gradient alone left the style term
at 0.6% of the loss, so nothing was stabilising the run.
Ablations are config toggles, not code paths:
  B : lambda_content = 0
  C : both lambdas on
  D : data.style_twin = "jitter"   (same loss, photometric twin)
"""

import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
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


class SpectralTrainer:

    def __init__(self, cfg: dict, model: nn.Module, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        # Teacher targets the style branch — an EMA target is a moving average of
        # the student, which a raw stop-gradient copy is not.
        self.teacher = EMAModel(model, decay=cfg["spectral"]["ema_decay"]).to(device)

        lcfg = cfg["loss"]
        self.sup_loss = SupervisedLoss(
            focal_weight=lcfg["supervised"]["focal_weight"],
            dice_weight=lcfg["supervised"]["dice_weight"],
            l1_weight=lcfg["supervised"].get("l1_weight", 0.5),
            focal_gamma=lcfg["supervised"].get("focal_gamma", 2.0),
            focal_alpha=lcfg["supervised"].get("focal_alpha", None),
        )
        self.cons_loss = ConsistencyLoss(loss_type=lcfg["consistency"]["type"])

        scfg = cfg["spectral"]
        self.lam_c = scfg["lambda_content"]
        self.lam_s_max = scfg["lambda_style_max"]
        self.rampup = scfg["style_rampup_epochs"]
        # Guardrail: abort the run if the foreground rate collapses toward zero
        # while the style term is ramping.  None disables it.
        self.fg_floor = scfg.get("fg_rate_floor", None)

        tcfg = cfg["training"]
        self.optimizer = Adam(self.model.parameters(), lr=tcfg["lr"],
                              weight_decay=tcfg["weight_decay"])
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=tcfg["epochs"] - tcfg.get("warmup_epochs", 0))

        self.epochs = tcfg["epochs"]
        self.save_dir = Path(tcfg["save_dir"]); self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_dice = 0.0
        self.save_every = tcfg.get("save_every", 0)
        self._last_val_dice = float("nan")
        self._last_val_iou = float("nan")

        log_dir = Path(tcfg.get("log_dir", "./logs")); log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = log_dir / "train_log.csv"
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "l_sup", "l_cont", "l_style",
                                    "val_dice", "val_iou", "pred_entropy", "pred_fg_rate"])
        self.writer = SummaryWriter(log_dir=str(log_dir)) if _TB_AVAILABLE else None

    # ──────────────────────────────────────────

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        val_every = self.cfg["training"].get("val_every", 5)

        for epoch in range(1, self.epochs + 1):
            stats = self._train_one_epoch(train_loader, epoch)
            self.scheduler.step()

            if epoch % val_every == 0:
                metrics = self._validate(val_loader)
                self._last_val_dice = metrics["dice"]
                self._last_val_iou = metrics["iou"]
                if self._last_val_dice > self.best_dice:
                    self.best_dice = self._last_val_dice
                    save_checkpoint(self.model, self.optimizer, epoch,
                                    self.save_dir / "best.pth")

            if self.save_every and epoch % self.save_every == 0:
                save_checkpoint(self.model, self.optimizer, epoch,
                                self.save_dir / f"epoch_{epoch:04d}.pth")

            val_str = (f"  val Dice={self._last_val_dice:.4f}  IoU={self._last_val_iou:.4f}"
                       if epoch % val_every == 0 else "")
            print(f"[Epoch {epoch:3d}/{self.epochs}]  loss={stats['loss']:.4f}  "
                  f"sup={stats['l_sup']:.4f}  cont={stats['l_cont']:.4f}  "
                  f"sty={stats['l_style']:.4f}  H={stats['ent']:.3f}  "
                  f"fg={stats['fg']:.4f}{val_str}")

            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch, f"{stats['loss']:.6f}", f"{stats['l_sup']:.6f}",
                    f"{stats['l_cont']:.6f}", f"{stats['l_style']:.6f}",
                    f"{self._last_val_dice:.4f}", f"{self._last_val_iou:.4f}",
                    f"{stats['ent']:.4f}", f"{stats['fg']:.5f}"])

            if self.writer:
                self.writer.add_scalar("Loss/train", stats["loss"], epoch)
                self.writer.add_scalar("Loss/sup", stats["l_sup"], epoch)
                self.writer.add_scalar("Loss/content", stats["l_cont"], epoch)
                self.writer.add_scalar("Loss/style", stats["l_style"], epoch)
                self.writer.add_scalar("Collapse/pred_entropy", stats["ent"], epoch)
                self.writer.add_scalar("Collapse/pred_fg_rate", stats["fg"], epoch)
                if epoch % val_every == 0:
                    self.writer.add_scalar("Val/Dice", self._last_val_dice, epoch)
                    self.writer.add_scalar("Val/IoU", self._last_val_iou, epoch)

            if self.fg_floor is not None and stats["fg"] < self.fg_floor:
                print(f"\n!! Collapse guard: pred_fg_rate={stats['fg']:.5f} "
                      f"< floor {self.fg_floor}. The style term is flattening the "
                      f"prediction — lower spectral.lambda_style_max or lengthen "
                      f"style_rampup_epochs. Stopping at epoch {epoch}.")
                break

    # ──────────────────────────────────────────

    def _train_one_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.model.train()
        self.teacher.model.eval()
        lam_s = consistency_weight(epoch, self.lam_s_max, self.rampup)

        meters = {k: AverageMeter() for k in ("loss", "l_sup", "l_cont", "l_style", "ent", "fg")}
        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)

        for batch in pbar:
            x = batch["image"].to(self.device)
            x_style = batch["x_style"].to(self.device)
            x_content = batch["x_content"].to(self.device)
            y = batch["mask"].to(self.device)

            out = self.model(x)
            l_sup = self.sup_loss(out, y)

            # Content axis: structure moved, so only the label constrains it.
            if self.lam_c > 0:
                l_cont = self.sup_loss(self.model(x_content), y)
            else:
                l_cont = torch.zeros((), device=self.device)

            # Style axis: structure is unchanged by construction (same leaf,
            # CAST restyle), so the whole probability map must hold still.
            if lam_s > 0:
                with torch.no_grad():
                    out_s = self.teacher(x_style)
                l_style = self.cons_loss(out, out_s)
            else:
                l_style = torch.zeros((), device=self.device)

            loss = l_sup + self.lam_c * l_cont + lam_s * l_style

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.teacher.update(self.model)

            with torch.no_grad():
                ent, fg = collapse_stats(out)

            n = x.shape[0]
            for k, v in (("loss", loss.item()), ("l_sup", l_sup.item()),
                         ("l_cont", l_cont.item()), ("l_style", l_style.item()),
                         ("ent", ent), ("fg", fg)):
                meters[k].update(v, n)
            pbar.set_postfix(loss=f"{meters['loss'].avg:.4f}", lam_s=f"{lam_s:.3f}",
                             fg=f"{meters['fg'].avg:.4f}")

        return {k: m.avg for k, m in meters.items()}

    # ──────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict:
        self.model.eval()
        all_preds, all_masks = [], []
        for batch in loader:
            logits = self.model(batch["image"].to(self.device))
            all_preds.append(logits.argmax(dim=1).cpu())
            all_masks.append(batch["mask"])
        return compute_metrics(torch.cat(all_preds), torch.cat(all_masks),
                               num_classes=self.cfg["model"]["num_classes"])
