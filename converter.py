# converter.py

class IntToString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": -2_147_483_648, "max": 2_147_483_647, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "to_string"
    CATEGORY = "string"

    def to_string(self, value: int):
        return (str(value),)
        

import torch
import torch.nn.functional as F
from comfy.comfy_types import ComfyNodeABC



class AudioResampler(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "convert_to": (["none", "int8", "int16", "int32", "float16", "float32"],),
                "resample": ("BOOLEAN",{ "default": False }),
                "resample_from_original_samplerate": ("BOOLEAN",{ "default": True }),
                "override_input_samplerate_khz": ("INT",{ "default": 48, "min": 1, "max": 192000 }),
                "output_samplerate_khz": ("INT",{ "default": 48, "min": 1, "max": 192000 }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    CATEGORY = "audio"
    FUNCTION = "process"

    def process(self, audio, convert_to, resample, resample_from_original_samplerate, override_input_samplerate_khz, output_samplerate_khz):

        # Extract container content
        waveform: torch.Tensor = audio["waveform"]
        samplerate_in: int     = audio["sample_rate"]

        # Sanitize input -> float32 normalized PCM
        if waveform.dtype == torch.int8:
            # signed int8 range: -128 .. +127
            waveform = waveform.to(torch.float32) / 128.0

        elif waveform.dtype == torch.int16:
            # signed int16 range: -32768 .. +32767
            waveform = waveform.to(torch.float32) / 32768.0

        elif waveform.dtype == torch.int32:
            # signed int32 range: -2^31 .. +2^31-1
            waveform = waveform.to(torch.float32) / 2147483648.0

        elif waveform.dtype == torch.int64:
            # signed int64 range: -2^63 .. +2^63-1
            waveform = waveform.to(torch.float32) / 9223372036854775808.0

        elif waveform.dtype.is_floating_point:
            # preserve meaning — only enforce float32 and clamp range
            waveform = waveform.to(torch.float32).clamp(-1.0, 1.0)

        else:
            # unknown integral format — assume already scaled but convert dtype
            waveform = waveform.to(torch.float32)
        

        
        
        
        samplerate_out = samplerate_in
        
        # Step 2 — Decide samplerate target
        if resample:
            if not resample_from_original_samplerate:
                samplerate_in = override_input_samplerate_khz * 1000
            
            samplerate_out = output_samplerate_khz * 1000

            # Only resample if different
            if samplerate_out != samplerate_in:
                B, C, T = waveform.shape

                ratio = samplerate_out / float(samplerate_in)
                new_t = int(round(T * ratio))

                # 1D interpolation
                wf = waveform.reshape(B*C, 1, T)
                wf = F.interpolate(wf, size=new_t, mode="linear", align_corners=False)
                waveform = wf.reshape(B, C, new_t)


            print(f"[Audio Resampler]: sampled from {samplerate_in} to {samplerate_out}")


        # Step 3 — Convert to requested dtype
        if convert_to == "int8":
            waveform = (waveform * 128.0).round().to(torch.int8)

        elif convert_to == "int16":
            waveform = (waveform * 32767.0).round().to(torch.int16)

        elif convert_to == "int32":
            waveform = (waveform * 2147483647.0).round().to(torch.int32)

        elif convert_to == "float16":
            waveform = waveform.to(torch.float16)

        elif convert_to == "float32":
            waveform = waveform.to(torch.float32)
            
            
        return (
            {
                "waveform": waveform,
                "sample_rate": samplerate_out,
            },
        )



