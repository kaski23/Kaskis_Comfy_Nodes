from .load_image_with_filename import LoadImageWithFilename
from .load_video_with_filename import LoadVideoWithFilename
from .string_split_select import StringSplitSelect
from .switch import PairSwitchNode
from .converter import IntToString
from .unbroken_videogen import VideoHandler
from .unbroken_controlvideos import CollectKeyedVideosNode
from .preview_image_limited_resolution import PreviewImageLimited
from .unbroken_video_batch import CollectVideosNode
from .unbroken_kling_video_batch import KlingVideoHandler
from .async_delay import ImageDelay
from .unbroken_workflow_logger import WorkflowLoggerNode

NODE_CLASS_MAPPINGS = {
    "LoadImageWithFilename": LoadImageWithFilename,
    "LoadVideoWithFilename": LoadVideoWithFilename,
    "StringSplitSelect": StringSplitSelect,
    "IntToString": IntToString,
    "PairSwitchNode": PairSwitchNode,
    "Unbroken-Video-Handler": VideoHandler,
    "CollectKeyedVideosNode": CollectKeyedVideosNode,
    "CollectVideosNode": CollectVideosNode,
    "KlingVideoHandler": KlingVideoHandler,
    "PreviewImageLimited": PreviewImageLimited,
    "ImageDelay": ImageDelay,
    "WorkflowLoggerNode": WorkflowLoggerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageWithFilename": "Load Image (with Filename)",
    "LoadVideoWithFilename": "Load Video (with Filename)",
    "StringSplitSelect": "String Split & Select",
    "IntToString": "Int → String",
    "PairSwitchNode": "Pair Switch",
    "VideoHandler": "Unbroken-Video-Handler",
    "CollectKeyedVideosNode": "Unbroken-Controlvideo-Handler",
    "CollectVideosNode": "Unbroken-Video-Batchloader",
    "KlingVideoHandler": "Unbroken-Kling-Video-Batchloader",
    "PreviewImageLimited": "Preview Image (MP Limit)",
    "ImageDelay": "Async Image Delay (ms)",
    "WorkflowLoggerNode": "WorkflowLoggerNode",
}
