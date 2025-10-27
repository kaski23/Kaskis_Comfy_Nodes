import os, random, json
import numpy as np
import folder_paths
import comfy.utils as utils

from PIL import Image
try:
    from PIL.PngImagePlugin import PngInfo
except ImportError:
    PngInfo = None



class PreviewImageLimited:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to preview."}),
                "max_megapixels": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 50.0,
                    "step": 0.1,
                    "tooltip": "Maximale Auflösung der Preview in Megapixel (Breite*Höhe/1e6)."
                }),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "preview_images"
    OUTPUT_NODE = True

    CATEGORY = "image"
    DESCRIPTION = "Preview mit skalierter Auflösung (Megapixel-Limit)."

    def preview_images(self, images, max_megapixels=1.0, prompt=None, extra_pnginfo=None):
        results = []
        for (batch_number, image) in enumerate(images):
            arr = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

            # --- Skalierung ---
            w, h = img.size
            current_mpx = (w * h) / 1e6
            if current_mpx > max_megapixels:
                scale = (max_megapixels / current_mpx) ** 0.5
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Dateiname
            full_output_folder = self.output_dir
            filename = f"preview_{self.prefix_append}_{batch_number:03}.png"

            # Speichern
            metadata = None
            if not utils.args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            img.save(os.path.join(full_output_folder, filename), pnginfo=metadata, compress_level=self.compress_level)

            results.append({
                "filename": filename,
                "subfolder": "",
                "type": self.type
            })

        return {"ui": {"images": results}}