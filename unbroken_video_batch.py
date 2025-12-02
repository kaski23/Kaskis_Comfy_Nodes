import re
from pathlib import Path
from datetime import datetime
from huggingface_hub import snapshot_download
import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms

from comfy.comfy_types import ComfyNodeABC, IO
from comfy_api.input_impl import VideoFromFile
from comfy.utils import ProgressBar




KASKIS_BASE_DIR = Path(__file__).resolve().parent
KASKIS_YOLO_MODEL = None

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
    def __init__(self, video_path, frames_total, enableDebugMsg, progress_callback=None):
        self.weights_path   = KASKIS_BASE_DIR / "models" / "face_yolov9c.pt"
        self.video_path     = Path(video_path)
        self.frames_total   = frames_total
        self.enableDebugMsg = enableDebugMsg
        
        self.progress = progress_callback
        self.progress_start = 150
        self.progress_end   = 900
        self.progress_step  = self.progress_step  = 0 if frames_total <= 0 else (self.progress_end - self.progress_start) / frames_total
        
    def get_frames_score_yolo(self):
        """
        Scannt ein Video, detektiert Gesichter mit YOLO-basierendem Modell,
        liefert den Frame als Torch Tensor, der die meisten Gesichter enthält.
        """

        # YOLO-Modell laden
        global KASKIS_YOLO_MODEL
        if KASKIS_YOLO_MODEL is None:
            if self.enableDebugMsg:
                print("Loading YOLO face model once")
            KASKIS_YOLO_MODEL = YOLO(str(self.weights_path))
        
        model = KASKIS_YOLO_MODEL
        if self.enableDebugMsg:
            print(f"loaded Model")
        self.progress(self.progress_start)

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
            current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            
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
                best_score, best_frame, best_frame_no = score, frame_rgb, current_frame_idx
                
            if self.enableDebugMsg:
                print(f"processed frame {current_frame_idx}")
            progress_value = int(self.progress_start + self.progress_step * current_frame_idx)
            self.progress(progress_value)
                

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
                "useUnbrokenIDs": ("BOOLEAN", {"default": True}),
                "enableDebugMsg": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE", "INT", "IMAGE", "INT")
    RETURN_NAMES = ("File-ID", "First Frame", "Best Frame", "Best Frame Index", "Last Frame", "Last Frame Index")
    FUNCTION = "run"
    CATEGORY = "Unbroken"

    def run(self, folderpath: str, index: int, useUnbrokenIDs: bool, enableDebugMsg: bool):
        
        if enableDebugMsg:
            print(f"Unbroken Video Handler: Starting collection at {datetime.now()}")
            
        collector = VideoFileCollector(folderpath, useUnbrokenIDs)
        pbar = ProgressBar(1000)
        progress = pbar.update_absolute
        

        if not (0 <= index < len(collector.files)):
            raise IndexError(f"Index {index} out of range, total files: {len(collector.files)}")
        
        video_path, file_id = collector.files[index]
        progress(50)
        
        first_frame, last_frame, last_index = self.extract_first_last_frames(video_path)
        if enableDebugMsg:
            print(f"Extracted first and last frame at {datetime.now()}")
        progress(100)
        
        
        if enableDebugMsg:
            print(f"Started Video Scoring at {datetime.now()}")
            
        analyzer = VideoFaceAnalysis(video_path, last_index, enableDebugMsg, progress)
        best_frame, best_index = analyzer.get_frames_score_yolo()
        
        if enableDebugMsg:
            print(f"Finished Video Scoring at {datetime.now()}")
        progress(1000)
        
        if enableDebugMsg:
            print(f"Unbroken Video Handler: Using Entry {index} of {len(collector.files)}, getting filepath {video_path} and file-ID {file_id}")
            print(f"Unbroken Video Handler: Found best frame at index {best_index}, Index of last Frame is {last_index}")
            
        
        return file_id, first_frame, best_frame, best_index, last_frame, last_index


    def extract_first_last_frames(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        # ---- First frame lesen ----
        ok, first = cap.read()
        if not ok:
            cap.release()
            raise ValueError("Video contains no readable frames")

        # ---- Letzten sinnvollen Frame indexieren ----
        reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        last_index = reported_frames - 1

        # Versuche den letzten Frame zu lesen
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_index)
        ok, last = cap.read()

        # Wenn nicht lesbar → fallback: gehe rückwärts
        if not ok:
            fallback_idx = last_index - 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, fallback_idx)
            ok, last = cap.read()
            if ok:
                last_index = fallback_idx  # aktualisieren
            else:
                cap.release()
                raise ValueError("Failed to read any valid last frame")

        cap.release()

        # ---- Frame → Tensor helper ----
        def to_tensor(frame):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        return (
            to_tensor(first).permute(1, 2, 0).contiguous().unsqueeze(0),
            to_tensor(last).permute(1, 2, 0).contiguous().unsqueeze(0),
            last_index,
        )
