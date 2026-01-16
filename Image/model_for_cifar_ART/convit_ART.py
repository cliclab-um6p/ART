import torch
import torch.nn as nn
import timm


class ConVitSmallART(nn.Module):
    """
    ViT-Small (patch-16, 224) with ART regulariser.

      • TRAIN mode  (model.train()):    returns (logits , λ_ART · Σ‖∂F/∂x‖₁)
      • EVAL mode   (model.eval()):     returns logits  (tensor only)

    This matches ART’s training loop, which adds the extra loss term
    only when model.training is True, and fixes evaluation/attack crashes.
    """

    def __init__(
        self,
        *,
        pretrained: bool,
        img_size: int,
        num_classes: int,
        ART: float,                 # λ_ART
        patch_size: int = 16,
        **_,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            "convit_tiny",
            pretrained=pretrained,
            img_size=img_size,
            num_classes=num_classes,
        )

        self.coeff = ART
        self._saved_io = []  # list[(inp, out)] captured by hooks

        # one hook pair per transformer block
        for blk in self.backbone.blocks:
            blk.attn.register_forward_pre_hook(self._mark_requires_grad)
            blk.attn.register_forward_hook(self._capture_io)

    # ---------- hooks -------------------------------------------------
    @staticmethod
    def _mark_requires_grad(module, inputs):
        x = inputs[0]
        if not x.requires_grad:
            x.requires_grad_(True)          # make a grad leaf
        module._ART_inp = x                 # stash for after-hook

    def _capture_io(self, module, _inputs, output):
        self._saved_io.append((module._ART_inp, output))

    # ---------- forward ----------------------------------------------
    def forward(self, x):
        self._saved_io.clear()
        logits = self.backbone(x)

        # In eval / attack mode just return logits (tensor) – no tuple
        if not self.training:
            return logits

        # Training: compute ART penalty
        ART_loss = 0.0
        for inp, out in self._saved_io:
            g = torch.autograd.grad(
                out.sum(), inp, retain_graph=True, create_graph=True
            )[0]
            ART_loss += g.abs().sum()

        return logits, self.coeff * ART_loss

    # ART launcher calls this; we do nothing.
    def init_ART(self, *_, **__):
        pass
