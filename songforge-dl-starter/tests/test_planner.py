import torch
from songforge.models.planner.model import SongPlannerTransformer


def test_planner_shape_backward_and_generation():
    model = SongPlannerTransformer(vocab_size=64, d_model=32, nhead=4, num_layers=1, dim_feedforward=64, max_seq_len=64)
    x = torch.randint(0, 64, (2, 12))
    logits = model(x)
    assert logits.shape == (2, 12, 64)
    logits.mean().backward()
    generated = model.generate(x[:, :3], max_new_tokens=2, top_k=8)
    assert generated.shape == (2, 5)
