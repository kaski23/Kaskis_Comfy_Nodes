class StringSplitSelect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": False}),
                "delimiter": ("STRING", {"default": "_"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "split_and_select"
    CATEGORY = "string"

    def split_and_select(self, text: str, delimiter: str, index: int):
        if not delimiter:
            return ("NONE",)  # leeres Trennzeichen macht keinen Sinn

        parts = text.split(delimiter)
        if 0 <= index < len(parts):
            return (parts[index],)
        else:
            return ("NONE",)
