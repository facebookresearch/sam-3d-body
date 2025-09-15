import random

import cv2
import numpy as np

from core.metadata.atlas70 import pose_info

from core.visualization.skeleton_visualizer import SkeletonVisualizer

visualizer_atlas70 = SkeletonVisualizer(line_width=1, radius=2)
visualizer_atlas70.set_pose_meta(pose_info)


def tighten_bounding_box(img, mask, padding=10):
    # Find non-zero mask regions
    coords = cv2.findNonZero(mask)
    if coords is None:
        raise ValueError("No non-zero regions found in mask")

    # Get bounding box
    x, y, w, h = cv2.boundingRect(coords)

    # Add padding
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(mask.shape[1], x + w + padding)
    y_end = min(mask.shape[0], y + h + padding)

    # Crop image and mask
    tightened_img = img[y_start:y_end, x_start:x_end]
    tightened_mask = mask[y_start:y_end, x_start:x_end]

    return tightened_img, tightened_mask


def place_object(img_person, keypoints_2d, img_object, fo_ratio):
    """
    return: img_person, mask_object
    """
    img_object, mask_object = tighten_bounding_box(
        img_object[:, :, :3] * 255, img_object[:, :, 3]
    )
    object_area = mask_object.sum()

    # Get a tight bounding box around the keypoints_2d
    valid = keypoints_2d[:, 2] >= 0.5
    if not valid.any():
        print("Playcing object failed!!")
        return img_person, None
    bbox = np.array(
        [
            max(np.min(keypoints_2d[valid, 0]), 0),
            max(np.min(keypoints_2d[valid, 1]), 0),
            min(np.max(keypoints_2d[valid, 0]), img_person.shape[1]),
            min(np.max(keypoints_2d[valid, 1]), img_person.shape[0]),
            min(np.max(keypoints_2d[valid, 1]), img_person.shape[0]),
        ]
    )
    bbox_center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    bbox_scale = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])

    # import cv2

    # img_keypoints = visualizer_atlas70.draw_skeleton(img_person.copy(), keypoints_2d)
    # img_vis_1 = cv2.rectangle(
    #     img_keypoints,
    #     (
    #         int(bbox_center[0] - bbox_scale[0] // 2),
    #         int(bbox_center[1] - bbox_scale[1] // 2),
    #     ),
    #     (
    #         int(bbox_center[0] + bbox_scale[0] // 2),
    #         int(bbox_center[1] + bbox_scale[1] // 2),
    #     ),
    #     (0, 255, 0),
    #     2,
    # )

    scale_x = min(
        bbox_scale[0], bbox_center[0] * 2, (img_person.shape[1] - bbox_center[0]) * 2
    )
    scale_y = min(
        bbox_scale[1], bbox_center[1] * 2, (img_person.shape[0] - bbox_center[1]) * 2
    )
    person_area = scale_x * scale_y
    target_area = person_area * fo_ratio

    # Compute scaling factor to achieve target area
    scale = np.sqrt(target_area / object_area)
    try:
        new_dims = (int(img_object.shape[1] * scale), int(img_object.shape[0] * scale))

        # Resize image and mask
        resized_img_object = cv2.resize(
            img_object, new_dims, interpolation=cv2.INTER_LINEAR
        )
        resized_mask_object = cv2.resize(
            mask_object, new_dims, interpolation=cv2.INTER_NEAREST
        )  # Nearest for binary mask
        # Ensure mask remains binary
        resized_mask_object = (resized_mask_object > 0.5).astype(np.float32)[:, :, None]
    except:
        print("new_dims: ", new_dims)
        print("Playcing object failed!!")
        return img_person, None

    # Randomly place object within person bounding box
    x = random.randint(
        max(int(bbox_center[0] - bbox_scale[0] // 2), 0),
        min(int(bbox_center[0] + bbox_scale[0] // 2), img_person.shape[1]),
    )
    y = random.randint(
        max(int(bbox_center[1] - bbox_scale[1] // 2), 0),
        min(int(bbox_center[1] + bbox_scale[1] // 2), img_person.shape[0]),
    )

    x_min = max(x - resized_img_object.shape[1] // 2, 0)
    y_min = max(y - resized_img_object.shape[0] // 2, 0)
    x_max = min(x + resized_img_object.shape[1] // 2, img_person.shape[1])
    y_max = min(y + resized_img_object.shape[0] // 2, img_person.shape[0])

    # Compute crop region from the patch if it goes outside base bounds
    px_min = max(0, -(x - resized_img_object.shape[1] // 2))
    py_min = max(0, -(y - resized_img_object.shape[0] // 2))
    px_max = px_min + (x_max - x_min)
    py_max = py_min + (y_max - y_min)

    # Composite object onto person image
    try:
        img_person[y_min:y_max, x_min:x_max] = (
            img_person[y_min:y_max, x_min:x_max]
            * (1 - resized_mask_object[py_min:py_max, px_min:px_max])
            + resized_img_object[py_min:py_max, px_min:px_max]
            * resized_mask_object[py_min:py_max, px_min:px_max]
        )
    except:
        print("Playcing object failed!!")
        return img_person, None

    # try:
    img_mask = np.zeros_like(img_person[:, :, :1])
    img_mask[y_min:y_max, x_min:x_max] = (
        img_mask[y_min:y_max, x_min:x_max]
        + resized_mask_object[py_min:py_max, px_min:px_max]
    )
    # except:
    #     print("Retrieving object mask failed!!")
    #     return img_person, None

    # img_vis_2 = cv2.rectangle(
    #     img_person.copy(),
    #     (
    #         int(bbox_center[0] - bbox_scale[0] // 2),
    #         int(bbox_center[1] - bbox_scale[1] // 2),
    #     ),
    #     (
    #         int(bbox_center[0] + bbox_scale[0] // 2),
    #         int(bbox_center[1] + bbox_scale[1] // 2),
    #     ),
    #     (0, 255, 0),
    #     2,
    # )

    # img_vis = np.concatenate([img_vis_1, img_vis_2], axis=1)
    # num = random.randint(0, 99999)
    # cv2.imwrite(f"visualization/debug_dataloader_fo/{num:05d}.jpg", img_vis[:, :, ::-1])

    return img_person, img_mask
