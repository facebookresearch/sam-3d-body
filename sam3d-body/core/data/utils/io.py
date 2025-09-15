import time
from typing import Any, List
import os
import braceexpand
import cv2
import numpy as np

from PIL import Image

# from pytorchvideo.data.video import VideoPathHandler


def expand(s):
    return os.path.expanduser(os.path.expandvars(s))


def expand_urls(urls: str|List[str]):
    if isinstance(urls, str):
        urls = [urls]
    urls = [u for url in urls for u in braceexpand.braceexpand(expand(url))]
    return urls


def load_image_from_file(
    data_info: dict,
    backend: str = "cv2",
    image_format: str = "rgb",
    retry: int = 10,
) -> dict:
    img = load_image(data_info["img_path"], backend, image_format, retry)
    data_info['img'] = img
    data_info['img_shape'] = img.shape[:2]
    data_info['ori_shape'] = img.shape[:2]
    return data_info


def _pil_load(path: str, image_format: str) -> Image.Image:
    with Image.open(path) as img:
        if img is not None and image_format.lower() == "rgb":
            img = img.convert("RGB")
    return img


def _cv2_load(path: str, image_format: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is not None and image_format.lower() == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_image(
    path: str,
    backend: str = "pil",
    image_format: str = "rgb",
    retry: int = 10,
) -> Any:
    for i_try in range(retry):
        if backend == "pil":
            img = _pil_load(path, image_format)
        elif backend == "cv2":
            img = _cv2_load(path, image_format)
        else:
            raise ValueError("Invalid backend {} for loading image.".format(backend))

        if img is not None:
            return img
        else:
            print("Reading {} failed. Will retry.".format(path))
            time.sleep(1.0)
        if i_try == retry - 1:
            raise Exception("Failed to load image {}".format(path))


def resize_image(img, target_size, center=None, scale=None):
    height, width = img.shape[:2]
    aspect_ratio = width / height

    # Calculate the new size while maintaining the aspect ratio
    if aspect_ratio > 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size

    # Resize the image using OpenCV
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new blank image with the target size
    final_img = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

    # Paste the resized image onto the blank image, centering it
    start_x = (target_size - new_width) // 2
    start_y = (target_size - new_height) // 2
    final_img[start_y:start_y + new_height, start_x:start_x + new_width] = resized_img

    if center is not None and scale is not None:
        ratio_width = new_width / width
        ratio_height = new_height / height

        new_scale = np.stack(
            [scale[:, 0] * ratio_width, scale[:, 1] * ratio_height],
            axis=1
        )
        new_center = np.stack(
            [center[:, 0] * ratio_width, center[:, 1] * ratio_height],
            axis=1
        )
        new_center[:, 0] += start_x
        new_center[:, 1] += start_y
    else:
        new_center, new_scale = None, None
    return aspect_ratio, final_img, new_center, new_scale


# def get_video_duration(path: str, decoder: str) -> Optional[float]:
#     try:
#         if decoder == "ffmpeg":
#             probe = ffmpeg.probe(path)
#             video_stream = next((stream for stream in probe['streams']
#                                 if stream['codec_type'] == 'video'), None)
            
#             if "duration" in video_stream:
#                 duration = float(video_stream['duration'])
#             else:
#                 duration = float(probe["format"]["duration"])
#         else:
#             handler = VideoPathHandler()
#             video = handler.video_from_path(
#                 filepath=path,
#                 decode_audio=False,
#                 decoder=decoder,
#             )
#             duration = float(video.duration)
        
#         return duration
#     except:
#         return None


# def load_video(
#     path: str,
#     start_sec: Optional[float] = None,
#     end_sec: Optional[float] = None,
#     backend: str = "ffmpeg",
#     target_fps: Optional[int] = None,
#     target_frames: Optional[int] = None,
#     target_size: Union[int, List] = [360, 640],  # Only for ffmpeg: [height, width]
# ):
#     """
#     Load a video clip by decoding a video file.
#     If start_sec and end_sec are not provided, decode the whole video.

#     Codes adapted from VideoFeatureExtractor:
#     https://github.com/ArrowLuo/VideoFeatureExtractor/tree/master
#     and PyTorchVideo:
#     https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/labeled_video_dataset.py
#     """
    
#     if start_sec is not None and end_sec is not None:
#         clip_duration = end_sec - start_sec
#     else:
#         clip_duration = get_video_duration(path, backend)
    
#     # Check whether decode fps can be inferred.
#     if target_fps is not None:
#         fps = target_fps
#     elif target_frames is not None and clip_duration is not None:
#         fps = math.ceil(target_frames / clip_duration)
#     else:
#         fps = None  # decode with the original frame rate

#     if backend == "ffmpeg":
#         if start_sec is not None and end_sec is not None:
#             cmd = ffmpeg.input(path, ss=start_sec, t=clip_duration)
#         else:
#             cmd = ffmpeg.input(path)

#         if fps is not None:
#             cmd = cmd.filter('fps', fps=fps)
#         if type(target_size) == int:
#             target_size = [target_size, target_size]
#         # width x height for ffmpeg
#         cmd = cmd.filter('scale', target_size[1], target_size[0])
        
#         out, _ = (
#             cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
#             .run(capture_stdout=False, quiet=True)
#         )

#         clip = np.frombuffer(out, np.uint8).reshape(
#             [-1, target_size[0], target_size[1], 3]
#         )
#         clip = torch.tensor(clip).permute(3,0,1,2).to(torch.float32)

#         # Resample the video if needed. 
#         # (Sometimes ffmpeg doesn't give exact fps as expected.)
#         if target_fps is not None and clip_duration is not None:
#             frames_num = int(clip_duration * fps)
#         elif target_frames is not None:
#             frames_num = target_frames

#         if frames_num < clip.size(1):
#             clip = clip[:frames_num]
#         elif frames_num > clip.size(1):
#             clip = torch.cat(
#                 (
#                     clip,
#                     clip[:, -1].unsqueeze(1).repeat(
#                         1, frames_num-clip.size(1), 1, 1)
#                 ),
#                 dim=1
#             )
#     else:
#         handler = VideoPathHandler()
#         video = handler.video_from_path(
#             filepath=path,
#             decode_audio=False,
#             decoder=backend,
#         )

#         # clip is a dictionary with keys:
#         # {
#         #   "video": torch.float32 tensor with shape [C, T, H, W]
#         # }
#         if start_sec is None and end_sec is  None:
#             start_sec, end_sec = 0, video.duration
#         clip = video.get_clip(start_sec, end_sec)
#         clip = clip["video"]

#         # Subsampling to the target frame rate if needed
#         if fps is not None and fps != video.rate:
#             stride = video.rate / float(fps)
#             T = clip.size(1)
#             clip = clip[:, np.arange(0, T, stride).astype(np.long)]

#         # Force garbage collection to release video container immediately
#         # otherwise memory can spike.
#         video.close()
#         gc.collect()

#     return clip