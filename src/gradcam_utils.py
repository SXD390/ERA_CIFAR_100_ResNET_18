from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


def denormalize_cifar100(img_tensor: torch.Tensor) -> torch.Tensor:
    """
    img_tensor: (3, H, W), normalized
    returns: (3, H, W) in [0, 1]
    """
    device = img_tensor.device
    mean = torch.tensor(CIFAR100_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(CIFAR100_STD, device=device).view(3, 1, 1)
    img = img_tensor * std + mean
    img = img.clamp(0, 1)
    return img


def get_resnet_target_layer(model: torch.nn.Module):
    """
    For ResNet-18 CIFAR, we use the last conv layer in layer4 as target.
    Handles DataParallel by unwrapping .module.
    """
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    return model.layer4[-1].conv2


class GradCAM:
    """
    Lightweight Grad-CAM implementation.

    Usage:
        target_layer = get_resnet_target_layer(model)
        cam = GradCAM(model, target_layer, device)
        heatmap = cam.generate(input_tensor)  # (H, W) numpy array
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module, device: torch.device):
        self.model = model
        self.target_layer = target_layer
        self.device = device

        self.activations = None
        self.gradients = None

        self.fwd_hook = self.target_layer.register_forward_hook(self._forward_hook)
        self.bwd_hook = self.target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def remove_hooks(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        input_tensor: (1, 3, H, W) normalized, on self.device
        target_class: class index, if None uses argmax prediction
        returns: heatmap (H, W) as numpy array in [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()

        x = input_tensor.to(self.device)

        with torch.enable_grad():
            outputs = self.model(x)  # (1, num_classes)

            if target_class is None:
                target_class = outputs.argmax(dim=1).item()

            score = outputs[0, target_class]
            score.backward(retain_graph=True)

        activations = self.activations   # (B, C, H, W)
        gradients = self.gradients       # (B, C, H, W)

        # Global average pool gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        cam = (weights * activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = F.relu(cam)

        cam = cam[0]  # (1, H, W)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        _, _, H_in, W_in = x.shape
        cam_upsampled = F.interpolate(
            cam.unsqueeze(0),
            size=(H_in, W_in),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        return cam_upsampled.detach().cpu().numpy()