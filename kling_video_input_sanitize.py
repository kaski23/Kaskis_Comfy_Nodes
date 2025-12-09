import torch
import math
import torch.nn.functional as F
from comfy.comfy_types import ComfyNodeABC


class VideoSanitizer(ComfyNodeABC):

    # -------------------------------------------------------------
    # Node Interface
    # -------------------------------------------------------------
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE", {}),   # (B,H,W,C)
                "min_length": ("INT", {"default": 25, "min": 1, "max": 99999}),
                "max_length": ("INT", {"default": 250, "min": 1, "max": 99999}),

                # size constraints
                "min_width":  ("INT", {"default": 720, "min": 1, "max": 8192}),
                "min_height": ("INT", {"default": 720, "min": 1, "max": 8192}),
                "max_width":  ("INT", {"default": 1920, "min": 1, "max": 8192}),
                "max_height": ("INT", {"default": 1920, "min": 1, "max": 8192}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sanitize"
    CATEGORY = "Kling/Video"


    # -------------------------------------------------------------
    # Main function
    # -------------------------------------------------------------
    def sanitize(
        self,
        video: torch.Tensor,
        min_length: int, max_length: int,
        min_width: int, min_height: int,
        max_width: int, max_height: int
    ):

        frames = video
        n = frames.shape[0]

        # Extend if too short
        if n < min_length:
            frames = self._ping_pong_extend(frames, target=min_length)
            n = frames.shape[0]

        # Shorten if too long
        if n > max_length:
            frames = self._adaptive_shorten(frames, target=max_length)
            n = frames.shape[0]

        # Min/Max Resize
        frames = self._resize_to_constraints(
            frames,
            min_width=min_width,  min_height=min_height,
            max_width=max_width,  max_height=max_height,
        )

        return (frames,)


    # -------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------
    def _ping_pong_extend(self, video: torch.Tensor, target: int) -> torch.Tensor:
        frames = [video]
        forward = True
        length_accum = video.shape[0]

        while length_accum < target:
            block = video.flip(0) if forward else video
            frames.append(block)
            length_accum += block.shape[0]
            forward = not forward

        out = torch.cat(frames, dim=0)
        return out[:target]


    def _adaptive_shorten(self, video: torch.Tensor, target: int) -> torch.Tensor:
        length = video.shape[0]
        if length <= target:
            return video

        # linspace über Indizes
        indices = torch.linspace(0, length - 1, target)
        indices = indices.long()

        return video[indices]


    def _resize_to_constraints(
        self,
        video: torch.Tensor,
        min_width: int, min_height: int,
        max_width: int, max_height: int,
    ) -> torch.Tensor:

        """
        video: (B,H,W,C)
        """

        B, H, W, C = video.shape

        scale_up = 1.0
        scale_down = 1.0

        # --- Mindestgröße ---
        if H < min_height:
            scale_up = max(scale_up, min_height / H)
        if W < min_width:
            scale_up = max(scale_up, min_width / W)

        # --- Maximalgröße ---
        if H > max_height:
            scale_down = min(scale_down, max_height / H)
        if W > max_width:
            scale_down = min(scale_down, max_width / W)

        # Effektiver Skalierungsfaktor
        scale = scale_up * scale_down

        # Keine Größenänderung nötig
        if scale == 1.0:
            return video

        new_h = int(round(H * scale))
        new_w = int(round(W * scale))

        vid = video.permute(0, 3, 1, 2)  # (B,H,W,C) → (B,C,H,W)

        vid = F.interpolate(
            vid,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False
        )

        return vid.permute(0, 2, 3, 1).contiguous()
