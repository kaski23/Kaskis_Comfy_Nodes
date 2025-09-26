# switch.py

class PairSwitchNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "select_pair": ("INT", {"default": 1, "min": 1, "max": 3}),
                "I1": ("IMAGE", {}), "S1": ("STRING", {}),
            },
            "optional": {
                "I2": ("IMAGE", {}), "S2": ("STRING", {}),
                "I3": ("IMAGE", {}), "S3": ("STRING", {}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("Image_out", "String_out")
    FUNCTION = "switch"
    CATEGORY = "utils"
    
    def switch(self, select_pair,
               I1=None, S1=None,
               I2=None, S2=None,
               I3=None, S3=None,):
                   
        pairs = [
            (I1, S1), (I2, S2), (I3, S3)
        ]
        
        idx = max(1, min(select_pair, len(pairs))) - 1
        chosen = pairs[idx]
        if chosen[0] is None or chosen[1] is None:
            raise ValueError(f"PairSwitchNode: selected pair {select_pair} is not fully connected.")
        return chosen