import asyncio
from comfy.comfy_types import ComfyNodeABC

class ImageDelay(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The images to preview."}),
                "delay": ("INT", {"default": 50, "min": 0, "max": 1000})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "utils/timing"

    async def run(self, image, delay):
        delay = delay / 1000.0
        await asyncio.sleep(delay)
        return (image,)
