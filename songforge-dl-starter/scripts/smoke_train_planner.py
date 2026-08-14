from __future__ import annotations

import torch
import torch.nn.functional as F

from songforge.models.planner.model import SongPlannerTransformer
from songforge.training.seed import seed_everything


def main() -> None:
    seed_everything(42)
    model = SongPlannerTransformer(vocab_size=128, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, max_seq_len=128)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.randint(0, 128, (4, 32))
    for step in range(3):
        logits = model(tokens[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, 128), tokens[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        print(f"step={step} loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
