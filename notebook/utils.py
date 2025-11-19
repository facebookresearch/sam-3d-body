# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Utility functions for SAM 3D Body demo notebook
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from typing import List, Dict, Any

from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from sam_3d_body.visualization.renderer import Renderer
from sam_3d_body.metadata.atlas70 import pose_info as atlas70_pose_info

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

def setup_visualizer():
    """Set up skeleton visualizer with Atlas70 pose info"""
    visualizer = SkeletonVisualizer(line_width=2, radius=5)
    visualizer.set_pose_meta(atlas70_pose_info)
    return visualizer

def visualize_2d_results(img_cv2: np.ndarray, outputs: List[Dict[str, Any]], 
                        visualizer: SkeletonVisualizer) -> List[np.ndarray]:
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
            2
        )
        
        # Add person ID text
        cv2.putText(img_vis, f'Person {pid}', 
                   (int(bbox[0]), int(bbox[1]-10)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        results.append(img_vis)
    
    return results

def visualize_3d_mesh(img_cv2: np.ndarray, outputs: List[Dict[str, Any]], 
                     faces: np.ndarray) -> List[np.ndarray]:
    """Visualize 3D mesh overlaid on image and side view"""
    results = []
    
    for pid, person_output in enumerate(outputs):
        # Create renderer for this person
        renderer = Renderer(
            focal_length=person_output["focal_length"], 
            faces=faces
        )
        
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
            ) * 255
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
            ) * 255
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
            ) * 255
        ).astype(np.uint8)
        
        # Combine all views
        combined = np.concatenate([img_orig, img_mesh_overlay, img_mesh_white, img_mesh_side], axis=1)
        results.append(combined)
    
    return results

def save_mesh_results(img_cv2: np.ndarray, outputs: List[Dict[str, Any]], 
                     faces: np.ndarray, save_dir: str, image_name: str) -> List[str]:
    """Save 3D mesh results to files and return PLY file paths"""
    os.makedirs(save_dir, exist_ok=True)
    ply_files = []
    
    for pid, person_output in enumerate(outputs):
        # Create renderer for this person
        renderer = Renderer(
            focal_length=person_output["focal_length"], 
            faces=faces
        )
        
        # Store individual mesh
        tmesh = renderer.vertices_to_trimesh(
            person_output['pred_vertices'], 
            person_output['pred_cam_t'], 
            LIGHT_BLUE
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
            ) * 255
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

def display_results_grid(images: List[np.ndarray], titles: List[str], 
                        figsize_per_image: tuple = (6, 6)):
    """Display multiple images in a grid"""
    n_images = len(images)
    if n_images == 0:
        print("No images to display")
        return
        
    # Calculate grid dimensions
    cols = min(3, n_images)  # Max 3 columns
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per_image[0]*cols, figsize_per_image[1]*rows))
    
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
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def process_image_with_mask(model, image_path: str, mask_path: str):
    """Process image with external mask input"""
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask from {mask_path}")
    
    # Ensure mask is binary (0 or 255)
    mask = (mask > 127).astype(np.uint8) * 255
    
    print(f"Processing image with external mask: {mask_path}")
    print(f"Mask shape: {mask.shape}, unique values: {np.unique(mask)}")
    
    # Process with external mask
    outputs = model.process_one_image(image_path, masks=mask)
    
    return outputs
