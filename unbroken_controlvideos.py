import re
from pathlib import Path

from comfy.comfy_types import ComfyNodeABC, IO
from comfy_api.input_impl import VideoFromFile


class VideoFileCollector:
    VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".mpeg"}
    FILE_PATTERN = re.compile(r"^(\d_\d{2}_\d{2}[A-Za-z]?)")

    def __init__(self, basepath: str, subfolder: str, useUnbrokenFilepatterns: bool = True):
        self.basepath = Path(basepath).expanduser().resolve()
        self.useUnbrokenFilepatterns = useUnbrokenFilepatterns

        sub_path = Path(subfolder)
        if sub_path.is_absolute():
            self.folder_path = sub_path.resolve()
        else:
            self.folder_path = (self.basepath / sub_path).resolve()

        if not self.folder_path.exists():
            raise FileNotFoundError(f"Subfolder not found: {self.folder_path}")
        if not self.folder_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {self.folder_path}")

        self.files = self.collect_files()

    def collect_files(self):
        files = []
        for file in sorted(self.folder_path.iterdir()):
            if file.is_file():
                if file.suffix.lower() not in self.VIDEO_EXTENSIONS:
                    raise ValueError(f"File is not a recognized video: {file.name}")

                if self.useUnbrokenFilepatterns:
                    match = self.FILE_PATTERN.match(file.stem)
                    if not match:
                        raise ValueError(f"Invalid filename format: {file.name}")
                    identifier = match.group(1)
                else:
                    # Einfach Dateiname ohne Endung
                    identifier = file.stem

                files.append([str(file), identifier])
        return files


class CollectVideoNode(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "basepath": ("STRING", {"default": ""}),
                "subfolder": ("STRING", {"default": ""}),
                "index": ("INT", {"default": 0, "min": 0, "max": 99999}),
                "useUnbrokenFilepatterns": ("BOOLEAN", {"default": True}),
                "enableDebugMsg": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "VIDEO")
    RETURN_NAMES = ("File-ID", "Video")
    FUNCTION = "collect"
    CATEGORY = "Custom"

    def collect(self, basepath: str, subfolder: str, index: int, useUnbrokenFilepatterns: bool, enableDebugMsg: bool):
        collector = VideoFileCollector(basepath, subfolder, useUnbrokenFilepatterns)

        if not (0 <= index < len(collector.files)):
            raise IndexError(f"Index {index} out of range, total files: {len(collector.files)}")

        file_path, file_id = collector.files[index]
        video_dict = VideoFromFile(file_path)
        
        if enableDebugMsg:
            print(f"Unbroken Video Handler:")
            print(f"Using filepath: {file_path} with file-ID {file_id}")
        
        return file_id, video_dict
