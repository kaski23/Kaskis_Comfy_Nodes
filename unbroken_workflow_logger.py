import os
import csv
from pathlib import Path
import pandas as pd


class WorkflowLoggerNode:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_id": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "prompt1": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "prompt_neg": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "shift": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "seed": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "directory": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "csv_filename": ("STRING", {
                    "multiline": False,
                    "default": "workflow_log.csv"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("id", "prompt", "prompt1", "prompt_neg", "shift", "seed")
    FUNCTION = "log_to_csv"
    OUTPUT_NODE = True
    CATEGORY = "Unbroken"

    def log_to_csv(self, shot_id, prompt, prompt1, prompt_neg, shift, seed, directory, csv_filename):
        # Delete unnecessary whitespaces
        shot_id = shot_id.strip()
        
        # Check if any valid id was provided
        if not shot_id:
            raise ValueError("Error: ID is required and cannot be empty or whitespaces")
        
        
        # Make directory a path-object for Windows/Linux Compatibility
        directory = Path(directory).expanduser().resolve()
        
        # Check if a valid directory was provided
        if not directory.exists():
            raise ValueError("Error: directory could not be found")
            
        if not directory.is_dir():
            raise ValueError(f"Provided path is not a directory: {directory}")


        # Map output_file
        output_file = directory / csv_filename
        
        #generate Entry
        fieldnames = ['id', 'prompt', 'prompt1', 'prompt_neg', 'shift', 'seed']

        new_row = {
            'id': shot_id,
            'prompt': prompt,
            'prompt1': prompt1,
            'prompt_neg': prompt_neg,
            'shift': shift,
            'seed': seed
        }
        
        # Using pandas for IO
        if output_file.is_file():
            df = pd.read_csv(output_file)
            
            # Handle potentially missing cols in file
            for col in fieldnames:
                if col not in df.columns:
                    df[col] = None
        else:
            df = pd.DataFrame(columns = fieldnames)

        
        # Append new row to dataframe
        df.loc[len(df)] = [new_row.get(col, None) for col in fieldnames]
        
        # Merge ID-Duplicates, keep the first entry, enforces no-overwrite behavior
        before = len(df)
        df = df.drop_duplicates(subset=["id"], keep="first")
        after = len(df)
        
        if before == after:
            print(f"Inserted new ID '{new_row['id']}'")
        else:
            print(f"Merged duplicate ID '{new_row['id']}' — kept earliest entry")

        # Save to csv
        df.to_csv(output_file, index=False)
        
        # returns
        return (shot_id, prompt, prompt1, prompt_neg, shift, seed)
        
        