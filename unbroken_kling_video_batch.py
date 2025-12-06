import pandas as pd
import torch
import re
import subprocess
import json
import numpy as np
import math
import torch.nn.functional as F

from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from comfy.comfy_types import ComfyNodeABC


class KlingVideoHandler(ComfyNodeABC):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": 99999}),
                "basepath": ("STRING", {"default": ""}),
                "videos_folder": ("STRING", {"default": ""}),
                "styleframes_folder": ("STRING", {"default": ""}),
                "prompts_folder": ("STRING", {"default": ""}),
            },
            "optional": {
                "fileid_regex": ("STRING", {"default": r"(\d_\d\d_\d\d[A-Za-z]?)"}),
                "prompts_prefix": ("STRING", {"default": ""}),
                "logging_flags": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",   "IMAGE", "FLOAT", "IMAGE",       "STRING", "INT")
    RETURN_NAMES = ("Video-ID", "Video", "fps",   "Styleframes", "Prompt", "n_frames")
    FUNCTION = "run"
    CATEGORY = "Unbroken"

    def run(
            self,
            index: int,
            basepath: str,
            videos_folder: str,
            styleframes_folder: str,
            prompts_folder: str,
            fileid_regex: str = "\d_\d\d_\d\d[A-Za-z]?",
            prompts_prefix: str = "",
            logging_flags: str = "",
    ):
        # Setup Settings
        basepath = Path(basepath)
        videos_folder = Path(videos_folder)
        styleframes_folder = Path(styleframes_folder)
        prompts_folder = Path(prompts_folder)

        Settings.set_video_folder(basepath / videos_folder)
        Settings.set_style_img_folder(basepath / styleframes_folder)
        Settings.set_prompts_folder(basepath / prompts_folder)

        Settings.set_filename_regex(fileid_regex)
        Settings.set_prompts_prefix(prompts_prefix)
        Settings.set_debug_flags(logging_flags)

        # Setup Provider
        provider = Provider()

        video_id    = provider.load_id_at_idx(index)
        video       = provider.load_video_at_idx(index)
        fps         = Settings.VIDEO_FPS
        styleframes = provider.load_style_images_at_idx(index)
        prompt      = provider.load_prompt_at_idx(index)
        n_frames    = provider.load_n_frames_at_idx(index)

        return video_id, video, fps, styleframes, prompt, n_frames


class Provider:
    def __init__(self):
        master_loader = MasterLoader()
        self._master_df = master_loader.master_df

        self.cls_name = "Provider"

        if self._master_df is None:
            Utils.debug_print(f"{self.cls_name}: Got an empty master_df, can't continue.")
            return

    def load_id_at_idx(self,idx: int) -> str:
        self._assert_idx_valid(idx)

        return self._master_df.iloc[idx]["ID"]

    def load_video_at_idx(self, idx: int) -> torch.Tensor:
        self._assert_idx_valid(idx)

        video_path = self._master_df.iloc[idx]["video_path"]
        n_frames = self._master_df.iloc[idx]["n_frames"]
        video = TorchUtils.load_video_as_tensor(video_path)

        video = self._conform_video(video, n_frames)

        return video

    def load_style_images_at_idx(self, idx) -> torch.Tensor:
        self._assert_idx_valid(idx)

        style_img_list = self._master_df.iloc[idx]["style_img_list"]
        frames = []

        for si in style_img_list:
            img = Image.open(si.path).convert("RGB")
            arr = np.array(img, dtype=np.float32)  # (H,W,C)
            t = torch.from_numpy(arr) / 255.0
            frames.append(t)

        return torch.stack(frames, dim=0)  # (B,H,W,C)

    def load_prompt_at_idx(self, idx) -> str:
        self._assert_idx_valid(idx)
        return Settings.PROMPTS_PREFIX + self._master_df.iloc[idx]["prompt"]

    def load_n_frames_at_idx(self, idx) -> int:
        self._assert_idx_valid(idx)
        return self._master_df.iloc[idx]["n_frames"]

    @staticmethod
    def _conform_video(video: torch.Tensor, n_frames: int) -> torch.Tensor:
        if n_frames < Settings.VIDEO_MIN_LENGTH_F:
            video = TorchUtils.ping_pong_extend(video)

        if n_frames > Settings.VIDEO_MAX_LENGTH_F:
            video = TorchUtils.adaptive_shorten(video)

        video = TorchUtils.upscale_min_height(video)

        return video

    def _assert_idx_valid(self, idx: int) -> bool:
        if self._master_df is None:
            raise IndexError(f"{self.cls_name}: Tried to access empty master_df, couldn't continue.")

        if idx < 0 or idx >= self._master_df.shape[0]:
            raise IndexError(
                f"{self.cls_name}: Index {idx} out of range. Valid range: 0..{self._master_df.shape[0] - 1}"
            )

        # Alle Prüfungen bestanden
        return True


class MasterLoader:
    def __init__(self):
        self.cls_name = "MasterLoader"
        video_manager = VideoLoader()
        imgs_manager = StyleImageLoader()
        prompts_manager = PromptLoader()

        try:
            video_manager.create_video_dataframe()
            imgs_manager.create_imgs_dataframe()
            prompts_manager.create_prompts_dataframe()

            video_df   = video_manager.video_dataframe
            imgs_df    = imgs_manager.images_dataframe
            prompts_df = prompts_manager.prompts_dataframe

        except Exception as e:
            Utils.debug_print(f"{self.cls_name}: Error when creating individual dataframes.")
            Utils.debug_print(f"{self.cls_name}: Got Exception: {e}")
            self.master_df = None
            return

        self.master_df = video_df.merge(imgs_df, on='ID', how='inner')
        self.master_df = self.master_df.merge(prompts_df, on='ID', how='left')
        self.master_df["prompt"] = self.master_df["prompt"].fillna("")


class VideoLoader:
    def __init__ (self):
        self._video_folder = Settings.VIDEO_FOLDER
        self.cls_name = "VideoLoader"

        self.video_dataframe = pd.DataFrame(columns = Settings.REQUIRED_VIDEO_COLS)

    def create_video_dataframe(self):
        files = Utils.list_files_matching(
            folder = self._video_folder,
            extensions = [".mov", ".mp4"])

        Utils.debug_print(f"{self.cls_name}: Found video files:", "IO, video_dataframe, debug")
        Utils.debug_print(f"{self.cls_name}: {files}", "IO, video_dataframe, debug")

        for p in files:
            vid_id   = Utils.extract_id_from_string(p.name)
            n_frames = Utils.get_n_frames(p)

            if vid_id is None:
                Utils.debug_print(f"{self.cls_name}: Could not extract video id from file {str(p)}", "IO, video_dataframe")
                continue

            if not n_frames > 0:
                Utils.debug_print(f"{self.cls_name}: Could not read frame count for: {str(p)}", "IO, video_dataframe")
                continue


            self.video_dataframe.loc[len(self.video_dataframe)] = {
                "ID": str(vid_id),
                "video_path": Path(p),
                "n_frames": int(n_frames)
            }

        self._consolidate_video_dataframe()

        Utils.debug_print(f"{self.cls_name}: Generated video-dataframe:", "IO, video_dataframe")
        Utils.debug_print(f"{self.video_dataframe}", "IO, video_dataframe")

    def _consolidate_video_dataframe(self):
        self.video_dataframe = self.video_dataframe.drop_duplicates(subset="ID", keep="first")


class StyleImageLoader:
    def __init__(self):
        self._images_folder = Settings.IMGS_FOLDER
        self.cls_name = "StyleImageLoader"

        self.images_dataframe = pd.DataFrame(columns=Settings.REQUIRED_IMAGE_COLS)


    def create_imgs_dataframe(self):
        files = Utils.list_files_matching(
            folder=self._images_folder,
            extensions=[".jpeg", ".jpg", ".png"])

        Utils.debug_print(f"{self.cls_name}: Found style images:", "IO, images_dataframe, debug")
        Utils.debug_print(f"{files}", "IO, images_dataframe, debug")

        for p in files:
            img_id = Utils.extract_id_from_string(p.name)
            frame_no = Utils.extract_frame_no_from_string(p.name)

            if img_id is None:
                Utils.debug_print(f"{self.cls_name}: Could not extract image id from file {str(p)}", "IO, images_dataframe")
                continue

            if frame_no is None:
                Utils.debug_print(f"{self.cls_name}: No frame number tag _fxx_ in: {str(p)}", "IO, images_dataframe")
                continue

            self.images_dataframe.loc[len(self.images_dataframe)] = {
                "ID": str(img_id),
                "style_img_list": [Utils.StyleImage(Path(p), int(frame_no))]
            }

        Utils.debug_print(f"{self.cls_name}: Style-image-dataframe before consolidation:", "IO, images_dataframe, debug")
        Utils.debug_print(f"{self.images_dataframe}", "IO, images_dataframe, debug")

        self._consolidate_images_dataframe()

        Utils.debug_print(f"{self.cls_name}: Generated style-image-dataframe:", "IO, images_dataframe")
        Utils.debug_print(f"{self.images_dataframe}", "IO, images_dataframe")


    def _consolidate_images_dataframe(self):
        df = self.images_dataframe

        def merge(rows):
            seen = set()
            result = []

            for item in rows:  # item ist die Liste aus StyleImages

                for style_img in item:  # style_img ist ein einzelnes Utils.StyleImage
                    path, frame_no = style_img.path, style_img.frame_no

                    if frame_no not in seen:
                        seen.add(frame_no)
                        result.append(Utils.StyleImage(path=path, frame_no=frame_no))

            result.sort(key=lambda x: x.frame_no)
            return result

        self.images_dataframe = (
            df.groupby("ID")
            .agg(style_img_list=("style_img_list", merge))
            .reset_index()
        )


class PromptLoader:
    def __init__(self):
        self._prompts_folder = Settings.PROMPTS_FOLDER
        self._required_cols = Settings.REQUIRED_PROMPT_COLS
        self._optional_cols = Settings.OPTIONAL_PROMPT_COLS
        self._prompts_cols = self._required_cols + self._optional_cols

        self.cls_name = "PromptLoader"
        self.prompts_dataframe = pd.DataFrame(columns=self._prompts_cols)


    def create_prompts_dataframe(self):
        files = Utils.list_files_matching(
            folder=self._prompts_folder,
            extensions=[".csv"])

        Utils.debug_print(f"{self.cls_name}: Found prompt tables:", "IO, prompts_dataframe, debug")
        Utils.debug_print(f"{files}", "IO, prompts_dataframe, debug")

        for p in files:
            current_df =  pd.read_csv(p)
            current_df = self._filter_csv(current_df)
            if current_df is None:
                Utils.debug_print(f"{self.cls_name}: Could not read file {p}", "IO, prompts_dataframe")
                continue

            self.prompts_dataframe = pd.concat([self.prompts_dataframe,current_df], axis=0)

        self._consolidate_prompts_dataframe()

        Utils.debug_print(f"{self.cls_name}:Generated prompts-dataframe:", "IO, prompts_dataframe")
        Utils.debug_print(f"{self.prompts_dataframe}", "IO, prompts_dataframe")


    def _filter_csv(self, df: pd.DataFrame):
        # Kills the df if not all required cols are provided
        if not set(self._required_cols).issubset(df.columns):
            return None

        # Add optional columns if not in df
        for col in self._optional_cols:
            if col not in df.columns:
                df[col] = ""

        # Drops all columns that are not needed
        df = df[self._required_cols + self._optional_cols]

        # Fills NaNs with "", drops entries where id and prompt isn't provided
        df = df.fillna("")
        df = df[~df[self._required_cols].eq("").any(axis=1)]

        return df

    def _consolidate_prompts_dataframe(self):
        if self.prompts_dataframe.empty:
            return

        df = self.prompts_dataframe

        # Schritt 1: Group by ID & join within each column
        agg = {
            col: (lambda s: " ".join([x for x in s if x]))
            for col in self._prompts_cols if col != "ID"
        }

        df = (
            df.groupby("ID", as_index=False)
            .agg(agg)
        )

        # Schritt 2: Alle Prompt-Spalten zusammenführen → eine Spalte "Prompts"
        prompt_cols = [c for c in self._prompts_cols if c != "ID"]

        df["prompt"] = (
            df[prompt_cols]
            .apply(lambda row: " ".join([x for x in row if x]), axis=1)
        )

        # Schritt 3: alte Prompt-Spalten löschen
        df = df[Settings.REQUIRED_PROMPT_COLS]

        self.prompts_dataframe = df


class Settings:
    DEBUG_FLAGS_STRING = ""
    DEBUG_FLAGS = set()
    POSS_DEBUG_FLAGS = "IO, video_dataframe, images_dataframe, prompts_dataframe, utils,"

    FILE_ID_REGEX = None
    FRAME_NO_REGEX = re.compile(r"_f(\d+)_")

    VIDEO_FOLDER = None
    IMGS_FOLDER = None
    PROMPTS_FOLDER = None

    ID_COLUMN            = ["ID"]
    REQUIRED_VIDEO_COLS  = ID_COLUMN + ["video_path", "n_frames"]
    REQUIRED_IMAGE_COLS  = ID_COLUMN + ["style_img_list"]
    REQUIRED_PROMPT_COLS = ID_COLUMN + ["prompt"]
    OPTIONAL_PROMPT_COLS = ["prompt1", "prompt2", "prompt3"]

    VIDEO_MIN_LENGTH_S = 3
    VIDEO_MAX_LENGTH_S = 10
    VIDEO_FPS = 25
    VIDEO_MIN_LENGTH_F = VIDEO_MIN_LENGTH_S * VIDEO_FPS
    VIDEO_MAX_LENGTH_F = VIDEO_MAX_LENGTH_S * VIDEO_FPS
    VIDEO_MIN_HEIGHT = 720

    PROMPTS_PREFIX = ""

    GET_N_FRAMES_ERROR_RETURN = -1



    @classmethod
    def set_video_folder(cls, folder: Path):
        if folder.is_dir():
            cls.VIDEO_FOLDER = Path(folder).expanduser().resolve()
        else:
            Utils.debug_print(f"Could not set video folder to {folder}")

    @classmethod
    def set_style_img_folder(cls, folder: Path):
        if folder.is_dir():
            cls.IMGS_FOLDER = Path(folder).expanduser().resolve()
        else:
            Utils.debug_print(f"Could not set images folder to {folder}")

    @classmethod
    def set_prompts_folder(cls, folder: Path):
        if folder.is_dir():
            cls.PROMPTS_FOLDER = Path(folder).expanduser().resolve()
        else:
            Utils.debug_print(f"Could not set prompts folder to {folder}")

    @classmethod
    def set_debug_flags(cls, flags: str):
        cls.DEBUG_FLAGS_STRING = flags
        cls.DEBUG_FLAGS = cls._parse(flags)

    @classmethod
    def set_prompts_prefix(cls, prefix: str):
        if isinstance(prefix, str):
            cls.PROMPTS_PREFIX = prefix
        else:
            Utils.debug_print(f"Could not set prompts prefix to {prefix}")

    @classmethod
    def set_filename_regex(cls, regex: str):
        if isinstance(regex, str):
            cls.FILE_ID_REGEX = re.compile(regex)
        else:
            Utils.debug_print(f"Could not set regex to {regex}")

    @staticmethod
    def _parse(flags: str) -> set[str]:
        return {f.strip() for f in flags.split(",") if f.strip()}


class Utils:

    @staticmethod
    def extract_id_from_string(string: str) -> str:
        if Settings.FILE_ID_REGEX is None:
            Utils.debug_print("No Regex provided for File-ID")
            return "FAULTY_REGEX_PARSER"

        match = Settings.FILE_ID_REGEX.match(string)
        return match.group(1) if match else None

    @staticmethod
    def extract_frame_no_from_string(string: str) -> int | None:
        match = Settings.FRAME_NO_REGEX.search(string)
        return int(match.group(1)) if match else None

    @staticmethod
    def debug_print(message: str, flags: str = "") -> None:
        requested = {f.strip() for f in flags.split(",") if f.strip()}
        if requested.issubset(Settings.DEBUG_FLAGS) or flags == "":
            print(message)

    @staticmethod
    def list_files_matching(folder: Path | str, extensions: list[str]) -> list[Path]:
        folder = Path(folder).expanduser().resolve()
        return [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ]

    @staticmethod
    def get_n_frames(path: str | Path) -> int:
        try:

            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-count_packets",
                "-show_entries", "stream=nb_read_packets",
                "-of", "json",
                str(path)
            ]
            out = subprocess.check_output(cmd).decode("utf-8")
            data = json.loads(out)
            return int(data["streams"][0]["nb_read_packets"])
        except Exception as e:
            Utils.debug_print(f"Excepted error as {e} for file {str(path)}")
            return Settings.GET_N_FRAMES_ERROR_RETURN


    @dataclass(frozen=True)
    class StyleImage:
        path: Path
        frame_no: int


class TorchUtils:
    @staticmethod
    def load_video_as_tensor(path: str) -> torch.Tensor:
        # ffmpeg-Command → RGB24 raw frames per pipe ausgeben
        cmd = [
            "ffmpeg",
            "-i", str(path),
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-"
        ]

        # ffprobe: Auflösung + Anzahl Frames holen
        probe = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,nb_frames",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path)
            ],
            capture_output=True,
            text=True
        )

        w, h, n = probe.stdout.splitlines()
        w, h = int(w), int(h)

        # ffmpeg-Process öffnen
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )

        frames = []
        frame_size = w * h * 3

        while True:
            raw = proc.stdout.read(frame_size)
            if not raw or len(raw) < frame_size:
                break

            frame = np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
            tensor = torch.from_numpy(frame).float() / 255.0
            frames.append(tensor)

        proc.stdout.close()
        proc.wait()

        if not frames:
            raise RuntimeError("No frames read from ffmpeg")

        return torch.stack(frames, dim=0)  # (B,H,W,C)

    @staticmethod
    def ping_pong_extend(video: torch.Tensor) -> torch.Tensor:
        frames = [video]
        forward = True
        target = Settings.VIDEO_MIN_LENGTH_F

        while sum(v.shape[0] for v in frames) < target:
            if forward:
                frames.append(video.flip(0))  # reverse frames
            else:
                frames.append(video)  # forward frames
            forward = not forward

        out = torch.cat(frames, dim=0)
        return out[:target]  # sauber abschneiden, falls Überhang

    @staticmethod
    def adaptive_shorten(video: torch.Tensor) -> torch.Tensor:
        length = video.shape[0]
        target = Settings.VIDEO_MAX_LENGTH_F

        if length <= target:
            return video

        factor = math.ceil(length / target)
        return video[::factor][:target]

    @staticmethod
    def upscale_min_height(video: torch.Tensor) -> torch.Tensor:
        """
        video: (B,H,W,C) float [0..1]
        """
        B, H, W, C = video.shape
        min_h = Settings.VIDEO_MIN_HEIGHT

        if H >= min_h:
            return video

        scale = min_h / H
        new_w = int(W * scale)

        # (B,H,W,C) -> (B,C,H,W)
        vid = video.permute(0, 3, 1, 2)

        vid = F.interpolate(
            vid,
            size=(min_h, new_w),
            mode="bilinear",
            align_corners=False
        )

        # Zurück nach (B,H,W,C)
        return vid.permute(0, 2, 3, 1).contiguous()
