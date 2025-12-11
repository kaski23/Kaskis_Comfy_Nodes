import torch
import math
import torch.nn.functional as F
from comfy.comfy_types import ComfyNodeABC


class ConformVideo(ComfyNodeABC):

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
        
        if min_length + max_length == 0:
            min_length, max_length = n, n
            
        elif min_length > 0 and max_length == 0:
            max_length = 999999999
        
        elif min_length >= max_length:
            raise ValueError ("min_length >= max_length")
            
        # --- Extend ---
        if n < min_length:
            frames = self._ping_pong_extend(frames, target=min_length)
            n = frames.shape[0]

        # --- Shorten ---
        if n > max_length:
            frames = self._adaptive_shorten(frames, target=max_length)
            n = frames.shape[0]
        
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



class ConformAudio(ComfyNodeABC):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "min_seconds": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100000.0}),
                "max_seconds": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100000.0}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "conform"
    CATEGORY = "Audio/Processing"


    # -------------------------------------------------------------
    # Main function
    # -------------------------------------------------------------
    def conform(self, audio, min_seconds: float, max_seconds: float):
        waveform = audio["waveform"]   # (B, C, S)
        sample_rate = audio["sample_rate"]

        B, C, S = waveform.shape

        # Umrechnen der Sekunden in Samples
        min_length = int(round(min_seconds * sample_rate)) if min_seconds > 0 else 0
        max_length = int(round(max_seconds * sample_rate)) if max_seconds > 0 else 0


        if min_length + max_length == 0:
            min_length, max_length = B, B
            
        elif min_length > 0 and max_length == 0:
            max_length = 999999999
        
        elif min_length >= max_length:
            raise ValueError ("min_length >= max_length")


        # ---------------------------------------------------------
        # EXTEND WITH SILENCE
        # ---------------------------------------------------------
        if S < min_length:
            pad_amount = min_length - S
            silence = torch.zeros((B, C, pad_amount), dtype=waveform.dtype, device=waveform.device)
            waveform = torch.cat([waveform, silence], dim=2)
            _, _, S = waveform.shape

        # ---------------------------------------------------------
        # SHORTEN (evenly spaced sampling)
        # ---------------------------------------------------------
        if S > max_length:
            indices = torch.linspace(0, S - 1, max_length).long().to(waveform.device)
            waveform = waveform[:, :, indices]

        return ({
            "waveform": waveform,
            "sample_rate": sample_rate
        },)



class ExtendVideoNearestFrame(ComfyNodeABC):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE", {}),          # (B,H,W,C)
                "target_length": ("INT", {"default": 200, "min": 1, "max": 999999}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "extend"
    CATEGORY = "Video/Processing"


    # -------------------------------------------------------------
    # Main function
    # -------------------------------------------------------------
    def extend(self, video: torch.Tensor, target_length: int):

        src_len = video.shape[0]

        # Wenn schon lang genug → nicht ändern
        if src_len >= target_length:
            return (video,)

        # Gleichmäßig verteilte Duplikate erzeugen
        indices = torch.linspace(
            0, src_len - 1, target_length,
            device=video.device
        ).long()

        extended = video[indices]

        return (extended,)
