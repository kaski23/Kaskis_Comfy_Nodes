import torch
import math
import torch.nn.functional as F
from comfy.comfy_types import ComfyNodeABC


class VideoSanitizer(ComfyNodeABC):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE", {}),   # (B,H,W,C)
                "min_length": ("INT", {"default": 25, "min": 0, "max": 99999}),
                "max_length": ("INT", {"default": 250, "min": 0, "max": 99999}),

                "min_width":  ("INT", {"default": 720, "min": 0, "max": 8192}),
                "min_height": ("INT", {"default": 720, "min": 0, "max": 8192}),
                "max_width":  ("INT", {"default": 1920, "min": 0, "max": 8192}),
                "max_height": ("INT", {"default": 1920, "min": 0, "max": 8192}),
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
        # -----------------------------------------------------------
        frames = video
        n, h, w, _ = frames.shape
        
        if min_length < max_length:
            # --- Extend ---
            if n < min_length:
                frames = self._ping_pong_extend(frames, target=min_length)
                n = frames.shape[0]

            # --- Shorten ---
            if n > max_length:
                frames = self._adaptive_shorten(frames, target=max_length)
                n = frames.shape[0]
        elif min_length >= max_length and max_length != 0:
            raise ValueError ("min_length >= max_length")
        
        
        # -----------------------------------------------------------
        #width-check
        if min_width + max_width == 0:
            min_width, max_width = w, w
            
        elif min_width > 0 and max_width == 0:
            max_width = 999999999   
            
        elif min_width >= max_width:
            raise ValueError ("min_width >= max_width")
        
        #height-check
        if min_height + max_height == 0:
            min_height, max_height = h, h
            
        elif min_height > 0 and max_height == 0:
            max_height = 999999999   
            
        elif min_height >= max_height:
            raise ValueError ("min_height >= max_height")

        # --- Resize ---
       
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

        # Gleichverteilte Indizes – perfekte Sample-Verteilung
        indices = torch.linspace(0, length - 1, target).long()
        return video[indices]


    def _resize_to_constraints(
        self,
        video: torch.Tensor,
        min_width: int, min_height: int,
        max_width: int, max_height: int,
    ) -> torch.Tensor:

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

        scale = scale_up * scale_down

        if scale == 1.0:
            return video

        new_h = int(round(H * scale))
        new_w = int(round(W * scale))

        vid = video.permute(0, 3, 1, 2)

        vid = F.interpolate(
            vid,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False
        )

        return vid.permute(0, 2, 3, 1).contiguous()
