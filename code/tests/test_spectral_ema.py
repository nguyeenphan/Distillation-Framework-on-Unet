"""EMA teacher in SpectralTrainer: it must exist, stay frozen, and trail the student."""
import sys, tempfile
from pathlib import Path

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trainers.spectral import SpectralTrainer


def _cfg(tmp: str) -> dict:
    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs/spectral.yaml").read_text())
    cfg["training"] |= {"epochs": 1, "save_dir": tmp, "log_dir": tmp}
    return cfg


def test_ema_teacher_trails_student():
    with tempfile.TemporaryDirectory() as tmp:
        model = nn.Conv2d(3, 2, 1)
        tr = SpectralTrainer(_cfg(tmp), model, torch.device("cpu"))

        assert not any(p.requires_grad for p in tr.teacher.model.parameters())
        before = tr.teacher.model.weight.clone()

        batch = {k: torch.rand(2, 3, 16, 16) for k in ("image", "x_style", "x_content")}
        batch["mask"] = torch.randint(0, 2, (2, 16, 16))
        # lam_s ramps from 0, so epoch 1 of 30 still weights the style branch.
        stats = tr._train_one_epoch([batch], epoch=1)

        # decay=0.999 moves the teacher by only 0.001*delta — exact compare, not allclose.
        assert not torch.equal(tr.teacher.model.weight, before), "teacher never updated"
        # Teacher is an average, so it must differ from the updated student.
        assert not torch.allclose(tr.teacher.model.weight, model.weight)
        assert stats["l_style"] > 0, "style branch silent — twin or lam_s is a no-op"
        assert 0.0 <= stats["fg"] <= 1.0
