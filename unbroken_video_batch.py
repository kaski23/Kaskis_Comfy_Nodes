import re
from pathlib import Path
from datetime import datetime

from comfy.comfy_types import ComfyNodeABC, IO
from comfy_api.input_impl import VideoFromFile


class VideoFileCollector:
    VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".mpeg"}
    FILE_PATTERN = re.compile(r"^(\d_\d{2}_\d{2}[A-Za-z]?)")

    def __init__(self, folderpath: str, use_unbroken_ids: bool):
        self.folder_path = Path(folderpath).expanduser().resolve()
        self.use_unbroken_ids = use_unbroken_ids

        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {self.folder_path}")
        if not self.folder_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {self.folder_path}")

        self.files = self.collect_files()

    def collect_files(self):
        
        files = []
        
        for file in sorted(self.folder_path.iterdir()):
            
            if not file.is_file() or file.suffix.lower() not in self.VIDEO_EXTENSIONS:
                continue
                
            if not self.use_unbroken_ids:
                identifier = file.stem
                files.append([str(file), identifier])
                
            else:
                match = self.FILE_PATTERN.match(file.stem)
                if match:
                    identifier = match.group(1)
                    files.append([str(file), identifier])
                    
        return files


class CollectVideosNode(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folderpath": ("STRING", {"default": ""}),
                "index": ("INT", {"default": 0, "min": 0, "max": 99999}),
                "useUnbrokenIDs": ("BOOLEAN", {"default": False}),
                "enableDebugMsg": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "VIDEO")
    RETURN_NAMES = ("File-ID", "Video")
    FUNCTION = "collect"
    CATEGORY = "Unbroken"

    def collect(self, folderpath: str, index: int, useUnbrokenIDs: bool, enableDebugMsg: bool):
        
        collector = VideoFileCollector(folderpath, useUnbrokenIDs)

        if not (0 <= index < len(collector.files)):
            raise IndexError(f"Index {index} out of range, total files: {len(collector.files)}")

        file_path, file_id = collector.files[index]
        video_dict = VideoFromFile(file_path)
        
        if enableDebugMsg:
            print(f"Unbroken Video Handler: Using Entry {index} of {len(collector.files)}, getting filepath {file_path} and file-ID {file_id}")
            print(f"Unbroken Video Handler: Starting generation at {datetime.now()}")
        
        return file_id, video_dict
