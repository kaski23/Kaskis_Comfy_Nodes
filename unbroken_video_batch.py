import re
from pathlib import Path
from datetime import datetime
from huggingface_hub import snapshot_download

from comfy.comfy_types import ComfyNodeABC, IO
from comfy_api.input_impl import VideoFromFile

import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms


BASE_DIR = Path(__file__).resolve().parent

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

class VideoFaceAnalysis:
    def __init__(self, video_path):
        self.weights_path = BASE_DIR / "models" / "face_yolov9c.pt"
        self.video_path = Path(video_path)
        
    def get_frames_score_yolo(self):
        """
        Scannt ein Video, detektiert Gesichter mit YOLO-basierendem Modell,
        liefert den Frame als Torch Tensor, der die meisten Gesichter enthält.
        """

        # YOLO-Modell laden
        model = YOLO(self.weights_path)

        cap = cv2.VideoCapture(self.video_path)
        transform = transforms.ToTensor()

        best_frame_no = 0
        best_frame = None
        best_score = 0

        while True:
            #Iteriert durch das Video, verlässt den While-Loop nach dem letzten Frame
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # RGB Conversion für YOLO
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Current Frame
            current_frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            # Run inference, disable training helpers, single Frame inference, so lets drop the batch-dimension
            results = model(frame_rgb, verbose=False)[0]

            # Auswertung
            boxes = results.boxes      # YOLO Boxes-Struktur
            if len(boxes) == 0:
                continue
                
            confs = boxes.conf         # Tensor [N] mit Confidence je Face
            xyxy = boxes.xyxy          # Bounding Box Koordinaten
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])

            score = (areas * confs).sum()

            if score > best_score:
                best_score, best_frame, best_frame_no = score, frame_rgb, current_frame_no
                

        cap.release()

        if best_frame is None:
            return None, 0

        best_frame_torch = transform(best_frame).float()  # C,H,W als Float 0..1
        return best_frame_torch.permute(1, 2, 0).contiguous().unsqueeze(0), best_frame_no
    


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

    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE", "INT", "IMAGE", "INT")
    RETURN_NAMES = ("File-ID", "First Frame", "Best Frame", "Best Frame Index", "Last Frame", "Last Frame Index")
    FUNCTION = "run"
    CATEGORY = "Unbroken"

    def run(self, folderpath: str, index: int, useUnbrokenIDs: bool, enableDebugMsg: bool):
        
        collector = VideoFileCollector(folderpath, useUnbrokenIDs)

        if not (0 <= index < len(collector.files)):
            raise IndexError(f"Index {index} out of range, total files: {len(collector.files)}")

        video_path, file_id = collector.files[index]
        
        first_frame, last_frame, last_index = self.extract_first_last_frames(video_path)
        
        analyzer = VideoFaceAnalysis(video_path)
        best_frame, best_index = analyzer.get_frames_score_yolo()
        
        if enableDebugMsg:
            print(f"Unbroken Video Handler: Starting collection at {datetime.now()}")
            print(f"Unbroken Video Handler: Using Entry {index} of {len(collector.files)}, getting filepath {video_path} and file-ID {file_id}")
            print(f"Unbroken Video Handler: Found best frame at index {best_index}, Index of last Frame is {last_index}")
            
        
        return file_id, first_frame, best_frame, best_index, last_frame, last_index


    def extract_first_last_frames(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        ok, first = cap.read()
        if not ok:
            cap.release()
            raise ValueError("Video contains no readable frames")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        last_index = frame_count - 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, last_index)
        ok, last = cap.read()
        cap.release()
        if not ok:
            raise ValueError("Failed to read last frame")

        def to_tensor(frame):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        return to_tensor(first).permute(1, 2, 0).contiguous().unsqueeze(0), to_tensor(last).permute(1, 2, 0).contiguous().unsqueeze(0), last_index