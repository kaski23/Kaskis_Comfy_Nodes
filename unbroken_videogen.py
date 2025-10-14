import pandas as pd
import os
import re
import subprocess
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import numpy as np
import torch

import folder_paths
from comfy.comfy_types import IO, ComfyNodeABC
from comfy_api.input_impl import VideoFromFile
import comfy.utils
import comfy.model_management as mm

"""
Possible logging flags:
log_provider, log_utils, log_prompt_generation, 
log_init, log_styleframe_generation, log_controlvideo_generation, 
log_prompt_generation, warnings
"""

class VideoHandler(ComfyNodeABC):
 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": 99999}),
                "matchlength": ("INT", {"default": 3, "min": 1, "max": 99999}),
                "id_by_splitting": ("BOOLEAN", {"default": True}),
                "splitting_symbol": ("STRING", {"default": "_"}),
                "basepath": ("STRING", {"default": ""}),
                "controlvideos_folder": ("STRING", {"default": ""}),
                "styleframes_folder": ("STRING", {"default": ""}),
                "prompts_folder": ("STRING", {"default": ""}),
                "use_prompts_and_noprompts": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "logging_flags": ("STRING",{"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING", "VIDEO", "IMAGE", "IMAGE", "MASK", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("Video-ID", "Controlvideo", "Stylevideo","first_styleframe", "Stylevideo-Mask", "Prompt", "width", "height", "n_frames")
    FUNCTION = "main"
    CATEGORY = "UNBROKEN-specific"
    
    def main(
            self, 
            index, 
            matchlength, 
            id_by_splitting, splitting_symbol, 
            basepath, controlvideos_folder, styleframes_folder, prompts_folder, 
            use_prompts_and_noprompts,
            logging_flags
            ):
        
        self.logging_flags = set(
            flag.strip() for flag in logging_flags.split(",") if flag.strip()
        )
        
        self.provider = Provider(
                                basepath, controlvideos_folder, styleframes_folder, prompts_folder,  
                                matchlength, 
                                id_by_splitting, splitting_symbol, 
                                use_prompts_and_noprompts,
                                self.logging_flags
                                )
        df = self.provider.gen_df
        
        if index < 0 or index >= len(df):
            raise IndexError(
                f"Index {index} liegt außerhalb des gültigen Bereichs (0 – {len(df)-1})"
            )
                
        
        video_id = df.iloc[index]["id"]
        controlvideo = VideoFromFile(df.iloc[index]["controlvideo"])
        prompt = df.iloc[index]["prompt"]
        width = df.iloc[index]["width_pad"]
        height = df.iloc[index]["height_pad"]
        n_frames = df.iloc[index]["n_frames"]
        
        first_styleframe_tupel = df.iloc[index]["styleframe"][0]
        _, frame_path = first_styleframe_tupel
        first_styleframe = load_image_as_comfy_tensor(frame_path, df.iloc[index]["width"], df.iloc[index]["height"])
        
        
        stylevideo, mask = generate_stylevideo(
                                df.iloc[index]["styleframe"], 
                                df.iloc[index]["width"], 
                                df.iloc[index]["height"], 
                                n_frames
        )
                            
        
        
        
        return str(video_id), controlvideo, stylevideo, first_styleframe, mask, str(prompt), int(width), int(height), int(n_frames)
        
        
         
       


def generate_stylevideo(styleframe_tupel, width: int, height: int, n_frames: int):
    video, mask = generate_wan_nullInput(width, height, n_frames)

    for frame_no, path in styleframe_tupel:
        # Index-Handling
        if frame_no < 0:
            target_index = 0
        elif frame_no >= n_frames:
            target_index = n_frames - 1
        else:
            target_index = frame_no

        # Styleframe einsetzen
        video[target_index] = load_image_as_comfy_tensor(path, width, height)[0]
        mask[target_index] = torch.full((1, height, width), 1.0)[0]

    return video, mask
    

def load_image_as_comfy_tensor(path: str, width: int, height: int) -> torch.Tensor:
    """
    Lädt ein Bild (png, jpg, jpeg) und gibt einen Comfy-kompatiblen Torch-Tensor zurück.
    Rückgabe-Shape: [1, H, W, 3], dtype=dtype, Wertebereich 0.0–1.0
    """
    
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(arr)[None, ...]  # [1, H, W, 3]
    
    return tensor


def generate_wan_nullInput(width: int, height: int, n_frames: int) -> (torch.Tensor, torch.Tensor):
    """
    Erzeugt einen neutralgrauen Video-Tensor.
    
    Args:
        n_frames: Anzahl der Frames
        height: Bildhöhe
        width: Bildbreite
        device: "cpu" oder "cuda"
    
    Returns:
        Torch-Tensor der Form [n_frames, height, width, 3], Werte 0.5
    """
    
    return (
            torch.full((n_frames, height, width, 3), 0.5), 
            torch.full((n_frames, height, width), 0.0)
    )


    
 
class Provider:
    def __init__(
        self, 
        basepath: str = "", controlvideos_folder: str = "", styleframes_folder: str = "", prompts_folder: str = "", 
        matchlength=3, 
        id_by_splitting=True, splitting_symbol="_", 
        use_prompts_and_noprompts = True,
        debug_flags=None):
        ### Flags ###
        if debug_flags is None:
            self.debug_flags = set()
            print("Unbroken-Videohandler: Possible debug-flags are: log_provider,log_utils,log_prompt_generation, log_init, log_styleframe_generation, log_controlvideo_generation, log_prompt_generation, warnings")
        else:
            self.debug_flags = set(debug_flags)

        self.log_current = "log_provider"

        ### Object Variables ###
        self.possible_resolutions = [(512, 512), (848, 480), (480, 848), (1024, 1024), (1280, 720), (720, 1280)]

        self.mt = MasterTable(
                        basepath, 
                        controlvideos_folder, 
                        styleframes_folder, 
                        prompts_folder,
                        matchlength,
                        id_by_splitting,
                        splitting_symbol,
                        debug_flags
        ).master_df

        self.gen_df = self.generate_generation_dataframe(self.mt, use_prompts_and_noprompts)



    ############################################################
    # Generiert den Dataframe, der zur Videogenerierung fertig indiziert ist
    ############################################################

    def generate_generation_dataframe(self, df, prompt_and_no_prompt = True):

        #width und height auf Wan anpassen
        df[["width_pad", "height_pad"]] = df.apply(
            lambda r: self.optimal_video_res(r["width"], r["height"]),
            axis=1, result_type="expand"
        )
        
        #Kürzt das Video auf die optimale Framerate
        df["n_frames"] = (df["n_frames"] // 4) * 4 + 1

        df["controlvideo"] = ""

        #Use both prompts and no prompts
        if prompt_and_no_prompt:
            for _, row in df.iterrows():
                if row["prompt"] != "":
                    new_row = row.copy()
                    new_row["prompt"] = ""
                    df.loc[len(df)] = new_row

        #Filter Table for unavailable Controlvideos
        df["controlvideo"] = df["controlvideo_combined"]
        df.loc[df["controlvideo"] == "", "controlvideo"] = df["controlvideo_depth"]
        df.loc[df["controlvideo"] == "", "controlvideo"] = df["controlvideo_normal"]

        df = df.drop(["controlvideo_combined", "controlvideo_normal", "controlvideo_depth"], axis = 1)



        return df



    #################################################################################
    ###                           CLASS UTILITIES                                 ###
    #################################################################################

    ############################################################
    # Generiert Log-Prints, falls die Log-Category
    # in den self.debug_flags enthalten ist
    ############################################################

    def log(self, msg, *flags):
        if not flags or all(flag in self.debug_flags for flag in flags):
            print(f"Unbroken Videohandler / MasterTable-Class:")
            print(msg)
            print(f"Log-Categories {flags}\n")

    ############################################################
    # Generiert Error-Prints
    ############################################################

    def print_error(self, msg):
        return "Unbroken Videohandler / Provider-Class: " + msg
        

    ############################################################
    # Generiert die optimale Zielauflösung
    ############################################################
    def optimal_video_res(self, width_in: int, height_in: int) -> tuple[int, int]:
        self.log(
            f"\nLog from 'optimal_video_res()':",
            "log_provider", "log_utils"
        )


        def test_resolution(width_in, width_poss, height_in, height_poss):

            if width_in - width_poss > height_in - height_poss:
                scaling = width_poss / width_in
            else:
                scaling = height_poss / height_in


            self.log(f"For {width_in}, {height_in}, testing {width_poss}, {height_poss}, scaling {scaling}",
                     "log_provider", "log_utils"
            )
            width_in *= scaling
            height_in *= scaling
            self.log(f"Size after scaling: {width_in}, {height_in}",
                     "log_provider", "log_utils"
            )

            waste_area = abs(width_in * height_in - width_poss * height_poss) / (width_in*height_in)

            self.log(f"wastearea: {waste_area}",
                     "log_provider", "log_utils"
            )

            if (scaling-1) < 0:
                scaling = (scaling -1) * 2
            else:
                scaling = scaling - 1

            score = abs(scaling) + waste_area

            self.log(f"score: {score}, \tscaling score: {abs(scaling)}\n",
                     "log_provider", "log_utils"
                     )

            return score


        best_score = float("inf")
        width_out = 0
        height_out = 0

        for w, h in self.possible_resolutions:
            test = test_resolution(width_in, w, height_in, h)
            if best_score > test:
                best_score = test
                width_out = w
                height_out = h


        self.log(f"For width, height: \t {width_in}, {height_in} \t now using {width_out}, {height_out}\n",
            "log_provider", "log_utils"
        )

        return width_out, height_out
        
        



class MasterTable:

    def __init__(
        self, 
        basepath: str = "", controlvideos_folder: str = "", styleframes_folder: str = "", prompts_folder: str = "", 
        matchlength = 1, 
        id_by_splitting = True, 
        splitting_symbol = "_", 
        debug_flags = None
        ):

        ### Flags ###
        if debug_flags is None:
            self.debug_flags = set()
            print("Unbroken-Videohandler: Possible debug-flags are: log_provider,log_utils,log_prompt_generation, log_init, log_styleframe_generation, log_controlvideo_generation, log_prompt_generation, warnings")
        else:
            self.debug_flags = set(debug_flags)

        self.log_current = "log_init"

        ### Filename-to-ID-Logik
        self.id_by_splitting = id_by_splitting  # Setzen, wenn IDs aus Filenamekomponenten, die durch '_' getrennt sind generiert werden soll
        self.splitting_symbol = splitting_symbol

        ### Init und Check der Variablen ###
        self.matchlength = matchlength  # Entweder Anzahl der String-Teile im Namen, die durch '_' separiert werden, oder Anzahl an Buchstaben
        self.basepath = Path(basepath)
        self.subfolders = [f.name for f in self.basepath.iterdir() if f.is_dir()]

        self.log(
            f"path_generation-log from 'init()':\n"
            f"Basepath \t= {self.basepath}\n"
            f"Subfolders \t= {self.subfolders}\n"
            f"Matchlength \t= {self.matchlength}\n"
            , self.log_current)

        # Kontrolliere, ob alle Ordner da sind, wenn ja, anlegen
        expected = {controlvideos_folder, styleframes_folder, prompts_folder}
        if not expected.issubset(set(self.subfolders)):
            raise ValueError(self.print_error(
                f"__init__():\nNicht alle Subfolders sind korrekt angelegt. Erwartet werden: {expected}"))
        
        self.controlvideos_folder = controlvideos_folder
        self.styleframes_folder = styleframes_folder
        self.prompts_folder = prompts_folder
        

        # Kontrolliere, ob die Matchlength passt
        if self.matchlength < 1 or not isinstance(self.matchlength, int):
            raise ValueError(self.print_error("Matchlength muss >= 1 sein"))

        ### Anlegen des Master-Dfs ###

        self.log(
            f"\n\n"
            f"STARTING GENERATION OF STYLEFRAME TABLE:"
            , "log_init", "log_styleframe_generation")
        styleframe_table = self.get_styleframe_table()

        self.log(
            f"\n\n"
            f"STARTING GENERATION OF CONTROLVIDEOS TABLE:"
            , "log_init", "log_controlvideo_generation")
        controlvideos_table = self.get_controlvideos_table()

        self.log(
            f"\n\n"
            f"STARTING GENERATION OF PROMPTS TABLE:"
            , "log_init", "log_prompt_generation")
        prompts_table = self.get_prompts_table()

        self.master_df = (
            styleframe_table
            .merge(controlvideos_table, on="id", how="inner")
            .merge(prompts_table, on="id", how="left")
        )
        
        self.consolidate()

        self.log(
            f"\n\n"
            f"GENERATED MASTER-DATAFRAME:"
            f"{self.master_df}"
            , "log_init")
            
        

    ############################################################
    # Checkt den Master-df auf Fehler
    # Erwartet wird ein df mit id, styleframe, 
    # controlvideo_depth, controlvideo_normal, controlvideo_combined, n_frames, width, height
    # prompt
    ############################################################
    def consolidate(self):
        # Checkt, ob überhaupt etwas im DF steht
        if self.master_df.empty:
            raise ValueError(self.print_error(
                "master_df ist leer – vermutlich keine gemeinsamen IDs gefunden "
                "(prüfe matchlength oder ID-Generierung)"
            ))
        
        
        # Checkt, ob die kritischen Spalten vorhanden sind
        required_columns = [
                        "id", "styleframe", "controlvideo_combined", "controlvideo_normal", 
                        "controlvideo_depth", "n_frames", "width", "height"
                        ]
                        
        for col in required_columns:
            if (col == "controlvideo_combined") or (col == "controlvideo_normal") or (col == "controlvideo_depth"):
                if (("controlvideo_combined"    not in self.master_df.columns) 
                and ("controlvideo_depth"       not in self.master_df.columns)
                and ("controlvideo_normal"      not in self.master_df.columns)):
                    raise ValueError(self.print_error(
                        "master_df enthält keine Controlvideos"
                    ))
                
            elif col not in self.master_df.columns:
                raise ValueError(self.print_error(
                        f"master_df enthält keine Spalte {col}"
                    ))
                    
                    
        # Nachträglich nichtkritische Spalten ergänzen, sollte eine Spalte nicht vorhanden sein keine Einträge stehen
        for col in ["controlvideo_combined", "controlvideo_normal", "controlvideo_depth", "prompt"]:
            if col not in self.master_df.columns:
                self.master_df[col] = pd.NA
                
                    
        # Checkt, ob alle kritischen Infos durchgehend vorhanden sind
        if self.master_df["id"].isna().any():
            raise ValueError(self.print_error(
                        "master_df enthält keine ids an mindestens einer Stelle"
                    ))
        
        if self.master_df["n_frames"].isna().any():
            raise ValueError(self.print_error(
                        "master_df enthält keine frameanzahl an mindestens einer Stelle"
                    ))
                   
        if self.master_df["width"].isna().any():
            raise ValueError(self.print_error(
                        "master_df enthält keine width an mindestens einer Stelle"
                    ))
                    
        if self.master_df["height"].isna().any():
            raise ValueError(self.print_error(
                        "master_df enthält keine height an mindestens einer Stelle"
                    ))

        # Checkt, ob es mindestens ein Controlvideo pro Video gibt
        cols = ["controlvideo_combined", "controlvideo_normal", "controlvideo_depth"]
        if not (self.master_df[cols].notna().any(axis=1).all()):
            raise ValueError(self.print_error(
                    "master_df enthält keine Controlvideos an mindestens einer Stelle"
                ))
              
              
        
        # Aufräumen
        self.master_df["prompt"]                    = self.master_df["prompt"].fillna("")
        self.master_df["controlvideo_combined"]     = self.master_df["controlvideo_combined"].fillna("")
        self.master_df["controlvideo_normal"]       = self.master_df["controlvideo_normal"].fillna("")
        self.master_df["controlvideo_depth"]        = self.master_df["controlvideo_depth"].fillna("")
        
        
        
    ############################################################
    # Generiert einen DataFrame für die Styleframes
    ############################################################

    def get_styleframe_table(self) -> pd.DataFrame:
        
        styleframe_df = pd.DataFrame(columns=["id", "styleframe"])
        style_path = self.basepath / self.styleframes_folder

        self.log_current = "log_styleframe_generation"
        self.log(
            f"\n"
            f"Log from 'get_styleframe_table()':\n"
            f"Path to Style-Images: \t {style_path}\n"
            , "log_styleframe_generation")

        index = 0
        for file in style_path.iterdir():
            if file.is_file():
                filestem = file.stem

                self.log(f"from 'get_styleframe_table()' -> 'calling generate_id()'...", "log_styleframe_generation",
                         "log_utils")
                file_id = self.generate_id(filestem)

                self.log(f"from 'get_styleframe_table()' -> 'calling get_frame_number_from_string()'...",
                         "log_styleframe_generation", "log_utils")
                frame_no = self.get_frame_number_from_string(filestem)

                styleframe_df.loc[index] = {"id": file_id, "styleframe": (frame_no, file)}
                index += 1

        double_entries = styleframe_df["id"].value_counts()[lambda x: x > 1]

        styleframe_df = styleframe_df.groupby("id")["styleframe"].agg(list).reset_index()

        self.log(f"\n"
                 f"generated Dataframe:\n"
                 f"{styleframe_df}"
                 , "log_styleframe_generation")

        self.log(f"\n"
                 f"IDs that were merged during Styleframe-Table Generation:\n"
                 f"{double_entries}"
                 , "warnings", "log_styleframe_generation")

        return styleframe_df

    ############################################################
    # Generiert einen DataFrame für die Controlvideos
    ############################################################

    def get_controlvideos_table(self) -> pd.DataFrame:
        
        control_path = self.basepath / self.controlvideos_folder

        self.log_current = "log_controlvideo_generation"
        self.log(
            f"\n"
            f"Log from 'get_controlvideos_table()':\n"
            f"Path to Controlvideo-Images: \t {control_path}\n"
            , "log_controlvideo_generation")

        files = []
        for file in control_path.iterdir():
            if file.is_file():
                files.append(file)

        # Hilfsfunktion für parallele Verarbeitung
        def process_file(file):

            filestem = file.stem

            self.log("from 'get_controlvideos_table()' -> 'calling generate_id()'...",
                     "log_controlvideo_generation", "log_utils")
            file_id = self.generate_id(filestem)

            self.log("from 'get_controlvideos_table()' -> 'calling get_video_infos()'...",
                     "log_controlvideo_generation", "log_utils")
            frame_count, width, height = self.get_video_infos(file)

            if "depth" in filestem:
                return {"id": file_id, "controlvideo_depth": file, "n_frames": frame_count, "width": width, "height": height}

            elif "normal" in filestem:
                return {"id": file_id, "controlvideo_normal": file, "n_frames": frame_count, "width": width, "height": height}

            elif "combined" in filestem:
                return {"id": file_id, "controlvideo_combined": file, "n_frames": frame_count, "width": width, "height": height}

            else:
                self.log(
                    f"\nWarning: found file in controls that doesn't match criteria\nFile: {file}",
                    "warnings"
                )
                return None

        # Ergebnisse hier sammeln
        results = []

        # ThreadPool starten
        with ThreadPoolExecutor() as executor:
            # Dictionary: Future -> File
            futures = {}
            for file in files:
                future = executor.submit(process_file, file)
                futures[future] = file

            # Auf Ergebnisse warten
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    bad_file = futures[future]
                    self.log(f"Error while processing {bad_file}: {e}", "warnings")

        # DataFrame bauen
        controlvideos_df = pd.DataFrame(results)

        # Nach Duplicates suchen
        # pro ID und Spalte: wie viele verschiedene Werte (ohne NaN) gibt es?
        uniques_per_id = (
            controlvideos_df
            .groupby("id")
            .agg(lambda x: x.dropna().nunique())
        )

        # IDs mit Konflikten = wenn irgendeine Spalte mehr als 1 unique Value hat
        conflicts = uniques_per_id[(uniques_per_id > 1).any(axis=1)]

        if not conflicts.empty:
            self.log(
                f"\nWarning: Mehrere unterschiedliche Controlvideos pro ID gefunden, "
                f"nur der erste wird behalten!\n{conflicts}",
                "warnings"
            )

        controlvideos_df = controlvideos_df.groupby("id").first().reset_index()
        
        
        


        self.log(f"\n"
                 f"generated Dataframe:\n"
                 f"{controlvideos_df}",
                 "log_controlvideo_generation")

        return controlvideos_df

    ############################################################
    # Generiert einen DataFrame für die Prompts
    ############################################################

    def get_prompts_table(self) -> pd.DataFrame:
        prompt_path = self.basepath / self.prompts_folder
        self.log_current = "log_prompt_generation"

        self.log(
            f"\nLog from 'get_prompts_table()':\n"
            f"Path to Prompts: \t {prompt_path}\n",
            "log_prompt_generation"
        )

        all_prompts = []

        for file in prompt_path.iterdir():
            if file.is_file() and file.suffix.lower() == ".csv":
                try:
                    df = pd.read_csv(file)

                    # Prüfen, ob die nötigen Spalten existieren
                    if not {"id", "prompt"}.issubset(df.columns):
                        self.log(
                            f"Warnung von 'get_prompts_table()': CSV {file} enthält nicht beide Spalten 'id' und 'prompt'. "
                            f"Gefundene Spalten: {list(df.columns)}",
                            "warnings"
                        )
                        continue

                    self.log(f"Adding prompts from {file}", "log_prompt_generation")

                    all_prompts.append(df[["id", "prompt"]])

                except Exception as e:
                    self.log(f"Fehler beim Einlesen von {file}: {e}", "warnings")

        # Falls keine gültigen CSVs gefunden wurden → leeres DF zurückgeben
        if not all_prompts:
            return pd.DataFrame(columns=["id", "prompt"])

        # Alle zusammenführen
        prompts_df = pd.concat(all_prompts, ignore_index=True)

        self.log(f"\n"
                 f"generated Dataframe:\n"
                 f"{prompts_df}"
                 , "log_prompt_generation")

        return prompts_df

    #################################################################################
    ###                           CLASS UTILITIES                                 ###
    #################################################################################

    ############################################################
    # Generiert Log-Prints, falls die Log-Category
    # in den self.debug_flags enthalten ist
    ############################################################

    def log(self, msg, *flags):
        if not flags or all(flag in self.debug_flags for flag in flags):
            print(f"Unbroken Videohandler / MasterTable-Class:")
            print(msg)
            print(f"Log-Categories {flags}\n")

    ############################################################
    # Generiert Error-Prints
    ############################################################

    def print_error(self, msg):
        return "Unbroken Videohandler / MasterTable-Class: " + msg

    ############################################################
    # Gibt die Zahl x zurück, die in einem String in der Form
    # _fx_, _fxx_, usw. vorkommt
    ############################################################

    def get_frame_number_from_string(self, filestem: str = "") -> int:
        match = re.search(r"_f(\d+)_", filestem)
        frame = 0

        if match:
            frame = int(match.group(1))

        self.log(
            f"\tutilities-log from 'get_frame_number_from_string()':\n"
            f"\tinput filestem: \t {filestem}\n"
            f"\tFound Frame?:   \t {bool(match)}\n"
            f"\tFrame-Number:   \t {frame}\n"
            , "log_utils", self.log_current)

        return frame

    ############################################################
    # Gibt einen ID-String zurück:
    # wenn self.id_by_splitting, dann die ersten self.matchlength
    # String-Komponenten, die durch '_' getrennt sind
    #
    # wenn !self.id_by_splitting, dann die ersten self.matchlength
    # Characters
    ############################################################

    def generate_id(self, filestem: str = "") -> str:

        if not isinstance(filestem, str):
            raise ValueError(self.print_error(f"generate_id(): \nEs wurde kein String übergeben"))

        # Wenn wir die ID durch Splitten an self.splitting_symbol generieren wollen
        if self.id_by_splitting:
            parts = filestem.split(self.splitting_symbol)

            # Check ob genügend Elemente durch den split("_") entstehen
            if len(parts) < self.matchlength:
                raise ValueError(self.print_error(
                    f"generate_id(): \n"
                    f"Zu wenige '_' im String: erwartet >= {self.matchlength-1}, übergebener String: {filestem}"
                    )
                )

            # Zu einer ID zusammenbauen
            file_id = "_".join(parts[:self.matchlength])

            # Logging
            self.log(
                f"\tutilities-log from 'generate_id()':\n"
                f"\tinput filestem: \t {filestem}\n"
                f"\tGenerated ID:   \t {file_id}\n"
                , "log_utils", self.log_current)

            return file_id

        # Wenn wir die ID anhand der ersten n Stellen generieren wollen
        else:
            return filestem[:self.matchlength]

    ############################################################
    # Gibt die Anzahl an Frames in einem Video-File zurück
    ############################################################

    def get_video_infos(self, video_path: str):
        cmd = [
            "ffprobe",
            "-v", "error",
            "-count_frames",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,width,height",
            "-of", "json",
            video_path
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        info = json.loads(result.stdout)

        stream = info["streams"][0]

        frame_count = int(stream.get("nb_read_frames", 0))
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))

        self.log(
            f"\tutilities-log from 'get_video_infos()':\n"
            f"\tinput file:     \t {video_path}\n"
            f"\tfound frames:   \t {frame_count}\n"
            f"\twidth:          \t {width}\n"
            f"\theight:         \t {height}\n",
            "log_utils", self.log_current
        )

        return frame_count, width, height