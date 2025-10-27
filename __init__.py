from .load_image_with_filename import LoadImageWithFilename
from .load_video_with_filename import LoadVideoWithFilename
from .string_split_select import StringSplitSelect
from .switch import PairSwitchNode
from .converter import IntToString
from .unbroken_videogen import VideoHandler
from .unbroken_controlvideos import CollectVideoNode
from .preview_image_limited_resolution import PreviewImageLimited

NODE_CLASS_MAPPINGS = {
    "LoadImageWithFilename": LoadImageWithFilename,
    "LoadVideoWithFilename": LoadVideoWithFilename,
    "StringSplitSelect": StringSplitSelect,
    "IntToString": IntToString,
    "PairSwitchNode": PairSwitchNode,
    "Unbroken-Video-Handler": VideoHandler,
    "CollectVideoNode": CollectVideoNode,
    "PreviewImageLimited": PreviewImageLimited,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageWithFilename": "Load Image (with Filename)",
    "LoadVideoWithFilename": "Load Video (with Filename)",
    "StringSplitSelect": "String Split & Select",
    "IntToString": "Int → String",
    "PairSwitchNode": "Pair Switch",
    "VideoHandler": "Unbroken-Video-Handler",
    "CollectVideoNode": "Unbroken-Controlvideo-Handler",
    "PreviewImageLimited": "Preview Image (MP Limit)",
}
