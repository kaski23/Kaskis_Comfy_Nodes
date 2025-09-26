# converter.py

class IntToString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": -2_147_483_648, "max": 2_147_483_647, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "to_string"
    CATEGORY = "string"

    def to_string(self, value: int):
        return (str(value),)