"""
Utility functions for SAM 3D Body demo notebook
"""

import os
from typing import Any, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from sam_3d_body import load_sam_3d_body_hf, SAM3DBodyEstimator
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
from sam_3d_body.visualization.renderer import Renderer
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def setup_sam_3d_body(
    hf_repo_id: str = "facebook/sam-3d-body-vith",
    detector_name: str = "vitdet",
    segmentor_name: str = "sam2",
    fov_name: str = "moge2",
    detector_path: str = "",
    segmentor_path: str = "",
    fov_path: str = "",
    device: str = "cuda",
):
    """
    Set up SAM 3D Body estimator with optional components.

    Args:
        hf_repo_id: HuggingFace repository ID for the model
        detector_name: Name of detector to use (default: "vitdet")
        segmentor_name: Name of segmentor to use (default: "sam2")
        fov_name: Name of FOV estimator to use (default: "moge2")
        detector_path: URL or path for human detector model
        segmentor_path: Path to human segmentor model (optional)
        fov_path: path for FOV estimator
        device: Device to use (default: auto-detect cuda/cpu)

    Returns:
        estimator: SAM3DBodyEstimator instance ready for inference
    """
    print(f"Loading SAM 3D Body model from {hf_repo_id}...")

    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load core model from HuggingFace
    model, model_cfg = load_sam_3d_body_hf(hf_repo_id, device=device)

    # Initialize optional components
    human_detector, human_segmentor, fov_estimator = None, None, None

    if detector_name:
        print(f"Loading human detector from {detector_name}...")
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(name=detector_name, device=device)

    if segmentor_path:
        print(f"Loading human segmentor from {segmentor_path}...")
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=segmentor_name, device=device, path=segmentor_path
        )

    if fov_name:
        print(f"Loading FOV estimator from {fov_name}...")
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=fov_name, device=device)

    # Create estimator wrapper
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    print(f"Setup complete!")
    print(
        f"  Human detector: {'✓' if human_detector else '✗ (will use full image or manual bbox)'}"
    )
    print(
        f"  Human segmentor: {'✓' if human_segmentor else '✗ (mask inference disabled)'}"
    )
    print(f"  FOV estimator: {'✓' if fov_estimator else '✗ (will use default FOV)'}")

    return estimator


def setup_visualizer():
    """Set up skeleton visualizer with MHR70 pose info"""
    visualizer = SkeletonVisualizer(line_width=2, radius=5)
    visualizer.set_pose_meta(mhr70_pose_info)
    return visualizer


def visualize_2d_results(
    img_cv2: np.ndarray, outputs: List[Dict[str, Any]], visualizer: SkeletonVisualizer
) -> List[np.ndarray]:
    """Visualize 2D keypoints and bounding boxes"""
    results = []

    for pid, person_output in enumerate(outputs):
        img_vis = img_cv2.copy()

        # Draw keypoints
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d_vis = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img_vis = visualizer.draw_skeleton(img_vis, keypoints_2d_vis)

        # Draw bounding box
        bbox = person_output["bbox"]
        img_vis = cv2.rectangle(
            img_vis,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),  # Green color
            2,
        )

        # Add person ID text
        cv2.putText(
            img_vis,
            f"Person {pid}",
            (int(bbox[0]), int(bbox[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        results.append(img_vis)

    return results


def visualize_3d_mesh(
    img_cv2: np.ndarray, outputs: List[Dict[str, Any]], faces: np.ndarray
) -> List[np.ndarray]:
    """Visualize 3D mesh overlaid on image and side view"""
    results = []

    for pid, person_output in enumerate(outputs):
        # Create renderer for this person
        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)

        # 1. Original image
        img_orig = img_cv2.copy()

        # 2. Mesh overlay on original image
        img_mesh_overlay = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                img_cv2.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)

        # 3. Mesh on white background (front view)
        white_img = np.ones_like(img_cv2) * 255
        img_mesh_white = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)

        # 4. Side view
        img_mesh_side = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                side_view=True,
            )
            * 255
        ).astype(np.uint8)

        # Combine all views
        combined = np.concatenate(
            [img_orig, img_mesh_overlay, img_mesh_white, img_mesh_side], axis=1
        )
        results.append(combined)

    return results


def save_mesh_results(
    img_cv2: np.ndarray,
    outputs: List[Dict[str, Any]],
    faces: np.ndarray,
    save_dir: str,
    image_name: str,
) -> List[str]:
    """Save 3D mesh results to files and return PLY file paths"""
    import json

    os.makedirs(save_dir, exist_ok=True)
    ply_files = []

    # Save focal length
    if outputs:
        focal_length_data = {"focal_length": float(outputs[0]["focal_length"])}
        focal_length_path = os.path.join(save_dir, f"{image_name}_focal_length.json")
        with open(focal_length_path, "w") as f:
            json.dump(focal_length_data, f, indent=2)
        print(f"Saved focal length: {focal_length_path}")

    for pid, person_output in enumerate(outputs):
        # Create renderer for this person
        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)

        # Store individual mesh
        tmesh = renderer.vertices_to_trimesh(
            person_output["pred_vertices"], person_output["pred_cam_t"], LIGHT_BLUE
        )
        mesh_filename = f"{image_name}_mesh_{pid:03d}.ply"
        mesh_path = os.path.join(save_dir, mesh_filename)
        tmesh.export(mesh_path)
        ply_files.append(mesh_path)

        # Save individual overlay image
        img_mesh_overlay = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                img_cv2.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)

        overlay_filename = f"{image_name}_overlay_{pid:03d}.png"
        cv2.imwrite(os.path.join(save_dir, overlay_filename), img_mesh_overlay)

        # Save bbox image
        img_bbox = img_cv2.copy()
        bbox = person_output["bbox"]
        img_bbox = cv2.rectangle(
            img_bbox,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),
            4,
        )
        bbox_filename = f"{image_name}_bbox_{pid:03d}.png"
        cv2.imwrite(os.path.join(save_dir, bbox_filename), img_bbox)

        print(f"Saved mesh: {mesh_path}")
        print(f"Saved overlay: {os.path.join(save_dir, overlay_filename)}")
        print(f"Saved bbox: {os.path.join(save_dir, bbox_filename)}")

    return ply_files


def display_results_grid(
    images: List[np.ndarray], titles: List[str], figsize_per_image: tuple = (6, 6)
):
    """Display multiple images in a grid"""
    n_images = len(images)
    if n_images == 0:
        print("No images to display")
        return

    # Calculate grid dimensions
    cols = min(3, n_images)  # Max 3 columns
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(figsize_per_image[0] * cols, figsize_per_image[1] * rows)
    )

    # Handle single image case
    if n_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = axes.flatten()

    for i, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Convert BGR to RGB if needed
            if img.dtype == np.uint8 and np.mean(img) > 1:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
        else:
            img_rgb = img

        axes[i].imshow(img_rgb)
        axes[i].set_title(title)
        axes[i].axis("off")

    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def process_image_with_mask(estimator, image_path: str, mask_path: str):
    """
    Process image with external mask input.

    Note: The refactored code requires bboxes to be provided along with masks.
    This function automatically computes bboxes from the mask.
    """
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask from {mask_path}")

    # Ensure mask is binary (0 or 255)
    mask_binary = (mask > 127).astype(np.uint8) * 255

    print(f"Processing image with external mask: {mask_path}")
    print(f"Mask shape: {mask_binary.shape}, unique values: {np.unique(mask_binary)}")

    # Compute bounding box from mask (required by refactored code)
    # Find all non-zero pixels in the mask
    coords = cv2.findNonZero(mask_binary)
    if coords is None:
        print("Warning: Mask is empty, no objects detected")
        return []

    # Get bounding box from mask contours
    x, y, w, h = cv2.boundingRect(coords)
    bbox = np.array([[x, y, x + w, y + h]], dtype=np.float32)

    print(f"Computed bbox from mask: {bbox[0]}")

    # Process with external mask and computed bbox
    # Note: The mask needs to match the number of bboxes (1 bbox -> 1 mask)
    outputs = estimator.process_one_image(image_path, bboxes=bbox, masks=mask_binary)

    return outputs


# ============================================================================
# VIDEO PROCESSING UTILITIES
# ============================================================================


def process_video(
    estimator,
    video_path: str,
    output_dir: str = None,
    frame_sample_rate: int = 1,
    save_intermediate_frames: bool = True,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Process a video and extract 3D body estimates for all frames.

    Args:
        estimator: SAM3DBodyEstimator instance
        video_path: Path to input video file
        output_dir: Directory to save results (optional)
        frame_sample_rate: Process every Nth frame (default: 1 = all frames)
        save_intermediate_frames: Whether to save annotated frames

    Returns:
        Dictionary mapping frame indices to list of person detections
    """
    import time

    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = f"output/video_{base_name}"
    
    os.makedirs(output_dir, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Resolution: {frame_width}x{frame_height}")
    print(f"  Processing every {frame_sample_rate} frame(s)")
    print(f"  Output directory: {output_dir}")

    # Setup video writer for annotated output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video_path = os.path.join(output_dir, "annotated_video.mp4")
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (frame_width, frame_height))

    frame_results = {}
    frame_idx = 0
    processed_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process only sampled frames
            if frame_idx % frame_sample_rate == 0:
                print(f"Processing frame {frame_idx}/{total_frames}...", end="\r")

                try:
                    # Run inference
                    outputs = estimator.process_one_image(frame)
                    frame_results[frame_idx] = outputs

                    # Visualize results on frame
                    if outputs and save_intermediate_frames:
                        frame_vis = frame.copy()
                        frame_vis = _annotate_frame(frame_vis, outputs, estimator.faces)
                        out.write(frame_vis)
                    else:
                        out.write(frame)

                    processed_count += 1

                except Exception as e:
                    print(f"\nError processing frame {frame_idx}: {e}")
                    out.write(frame)
                    frame_results[frame_idx] = []
            else:
                out.write(frame)

            frame_idx += 1

    finally:
        cap.release()
        out.release()

    elapsed_time = time.time() - start_time
    print(f"\nProcessing complete!")
    print(f"  Processed {processed_count} frames in {elapsed_time:.2f}s")
    print(f"  Average time per frame: {elapsed_time / max(processed_count, 1):.2f}s")
    print(f"  Annotated video saved: {out_video_path}")

    return frame_results


def _annotate_frame(
    frame: np.ndarray, outputs: List[Dict[str, Any]], faces: np.ndarray
) -> np.ndarray:
    """
    Annotate a frame with 2D keypoints, bboxes, and mesh overlays.

    Args:
        frame: Input frame (BGR format)
        outputs: List of person detections
        faces: Mesh face indices

    Returns:
        Annotated frame
    """
    from sam_3d_body.visualization.renderer import Renderer

    frame_annotated = frame.copy()

    for pid, person_output in enumerate(outputs):
        # Draw keypoints
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d_vis = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )

        # Simple skeleton visualization
        # Draw joints as circles
        for kpt in keypoints_2d:
            x, y = int(kpt[0]), int(kpt[1])
            cv2.circle(frame_annotated, (x, y), 3, (0, 255, 0), -1)

        # Draw bounding box
        bbox = person_output["bbox"]
        cv2.rectangle(
            frame_annotated,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),
            2,
        )

        # Add person ID
        cv2.putText(
            frame_annotated,
            f"Person {pid}",
            (int(bbox[0]), int(bbox[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # Optionally draw 3D mesh overlay (slower)
        try:
            renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)
            frame_annotated = (
                renderer(
                    person_output["pred_vertices"],
                    person_output["pred_cam_t"],
                    frame_annotated,
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    alpha=0.4,
                )
                * 255
            ).astype(np.uint8)
        except Exception as e:
            # Skip mesh overlay if there's an issue
            pass

    return frame_annotated


def save_video_meshes(
    frame_results: Dict[int, List[Dict[str, Any]]],
    estimator,
    video_path: str,
    output_dir: str = None,
    sample_frames: bool = True,
    max_meshes_per_person: int = 5,
) -> Dict[str, List[str]]:
    """
    Save 3D mesh files from video frames.

    Args:
        frame_results: Dictionary from process_video()
        estimator: SAM3DBodyEstimator instance
        video_path: Path to the video (for naming)
        output_dir: Output directory for meshes
        sample_frames: If True, save only a sample of frames
        max_meshes_per_person: Max meshes to save per person

    Returns:
        Dictionary mapping person IDs to lists of saved mesh file paths
    """
    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = f"output/meshes_{base_name}"

    os.makedirs(output_dir, exist_ok=True)

    mesh_files = {}
    frame_indices = sorted(frame_results.keys())

    # Sample frames if requested
    if sample_frames and len(frame_indices) > max_meshes_per_person:
        frame_indices = frame_indices[
            :: len(frame_indices) // max_meshes_per_person
        ][:max_meshes_per_person]

    for frame_idx in frame_indices:
        outputs = frame_results[frame_idx]

        for pid, person_output in enumerate(outputs):
            person_key = f"person_{pid}"
            if person_key not in mesh_files:
                mesh_files[person_key] = []

            try:
                renderer = Renderer(
                    focal_length=person_output["focal_length"],
                    faces=estimator.faces,
                )

                # Save mesh
                tmesh = renderer.vertices_to_trimesh(
                    person_output["pred_vertices"],
                    person_output["pred_cam_t"],
                    LIGHT_BLUE,
                )

                mesh_filename = f"frame_{frame_idx:06d}_person_{pid:03d}.ply"
                mesh_path = os.path.join(output_dir, mesh_filename)
                tmesh.export(mesh_path)
                mesh_files[person_key].append(mesh_path)

                print(f"Saved: {mesh_path}")

            except Exception as e:
                print(f"Error saving mesh for frame {frame_idx}, person {pid}: {e}")

    print(f"\nTotal meshes saved: {sum(len(v) for v in mesh_files.values())}")
    return mesh_files


def visualize_video_results(
    frame_results: Dict[int, List[Dict[str, Any]]],
    video_path: str,
    sample_frames: bool = True,
    max_frames: int = 5,
):
    """
    Display results from video processing as a grid.

    Args:
        frame_results: Dictionary from process_video()
        video_path: Path to video (for context)
        sample_frames: If True, display only sampled frames
        max_frames: Maximum frames to display
    """
    cap = cv2.VideoCapture(video_path)
    frame_indices = sorted(frame_results.keys())

    # Sample frames if requested
    if sample_frames and len(frame_indices) > max_frames:
        frame_indices = frame_indices[:: len(frame_indices) // max_frames][
            :max_frames
        ]

    frames_to_show = []
    titles = []

    for frame_idx in frame_indices:
        ret = True
        frame = None

        # Seek to frame and read
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret and frame is not None:
            outputs = frame_results[frame_idx]

            if outputs:
                # Convert keypoints to visualization format
                frame_vis = frame.copy()

                for pid, output in enumerate(outputs):
                    keypoints_2d = output["pred_keypoints_2d"]
                    keypoints_2d_vis = np.concatenate(
                        [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))],
                        axis=-1,
                    )

                    # Draw keypoints
                    for kpt in keypoints_2d:
                        x, y = int(kpt[0]), int(kpt[1])
                        cv2.circle(frame_vis, (x, y), 3, (0, 255, 0), -1)

                    # Draw bbox
                    bbox = output["bbox"]
                    cv2.rectangle(
                        frame_vis,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])),
                        (0, 255, 0),
                        2,
                    )

            else:
                frame_vis = frame

            frames_to_show.append(cv2.cvtColor(frame_vis, cv2.COLOR_BGR2RGB))
            titles.append(f"Frame {frame_idx}")

    cap.release()

    if frames_to_show:
        display_results_grid(frames_to_show, titles, figsize_per_image=(8, 6))
    else:
        print("No frames to display")
