import copy
import os
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import cv2

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate, LazyConfig
import detectron2.data.transforms as T

from core.data.utils.io import load_image, resize_image
from core.models.meta_arch import load_sam3d_body

from core.data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)
from core.utils import recursive_to
from torch.utils.data import default_collate
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_tensor


try:
    has_moge_env = True
    from moge.model.v2 import MoGeModel
except:
    has_moge_env = False
    print("Conda env. does not have MoGe installed!")

# Load the model from local.
def get_moge_model():
    device = torch.device("cuda")
    ckpt_path = "/checkpoint/3po/nugrinovic/data/external/moge/Ruicheng/moge-2-vitl-normal/model.pt"
    moge_model = MoGeModel.from_pretrained(ckpt_path).to(device)
    return moge_model

class NoCollate:
    def __init__(self, data):
        self.data = data

def get_cam_intrinsics(model, batch):
    model_name = model.__class__.__name__
    if model_name == "CameraHMR":
        batch_size = batch["img_full"].shape[0]

        # Initialize camera intrinsics
        img_h, img_w = batch["ori_img_size"][:, 0, 1], batch["ori_img_size"][:, 0, 0]
        cam_intrinsics = torch.zeros((batch_size, 3, 3)).to(batch["img"])
        cam_intrinsics[:, 0, 2] = img_w / 2
        cam_intrinsics[:, 1, 2] = img_h / 2
        cam_intrinsics[:, 2, 2] = 1

        # Get Camera intrinsics using HumanFoV Model
        img_full_resized = model.normalize_img(batch["img_full"])
        model.cam_model.eval()
        with torch.no_grad():
            estimated_fov, _ = model.cam_model(img_full_resized)
        vfov = estimated_fov[:, 1]
        fl_h = img_h / (2 * torch.tan(vfov / 2))
        cam_intrinsics[:, 0, 0] = fl_h
        cam_intrinsics[:, 1, 1] = fl_h

    elif model_name == "MoGeModel":
        # NOTE: for now only to be used with demo!
        # We expect the image to be RGB already
        input_image = batch['img_ori'][0].data
        H, W, _ = input_image.shape

        input_image = torch.tensor(input_image / 255, dtype=torch.float32,
                                   device=batch["img"].device).permute(2, 0, 1)
        # Infer w/ MoGe2
        model.eval()
        moge_data = model.infer(input_image)
        # get intrinsics
        intrinsics = denormalize_f(moge_data['intrinsics'].cpu().numpy(), H, W)
        v_focal = intrinsics[1, 1]
        # override hfov with v_focal ?
        intrinsics[0, 0] = v_focal
        intrinsics = (intrinsics).to(batch["img"])
        # add batch dim
        cam_intrinsics = intrinsics[None]
    else:
        raise NotImplementedError

    return cam_intrinsics


class SAM3DBodyEstimator:
    def __init__(
        self,
        checkpoint_path: str = "",
        proto_path: str = "",
        bbox_threshold: float = 0.5,
        detector_path: str = "",
        sam_path: str = "",
        use_mask: bool = False,
        use_triplet: bool = False,
        scale_factor = None,
        just_left_hand = False,
        use_face = False,
        fov_estimator = "camerahmr",
    ):
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.use_mask = use_mask
        self.scale_factor = scale_factor
        self.just_left_hand = just_left_hand

        # Initialize human detector
        self.use_detector = False
        if len(detector_path):
            self.detector = self.init_detector(detector_path, bbox_threshold)
            self.detector = self.detector.to(self.device)
            self.detector.eval()
            self.use_detector = True

        # Initialize SAM2 if needed
        if self.use_mask:
            self.sam_predictor = self.init_sam(sam_path)

        # Build SAM3D-Body model
        self.model, self.cfg = load_sam3d_body(checkpoint_path, proto_path,use_triplet=True, 
                                               use_twostage_for_hands=True, use_face=use_face)
        self.faces = self.model.head_pose.atlas.faces.numpy()
        self.model = self.model.to(self.device)
        self.model.eval()

        # is not using moge env. default to camerahmr
        self.fov_estimator = fov_estimator if has_moge_env else "camerahmr"
        if self.fov_estimator == "moge":
            self.moge_model = get_moge_model()
        elif self.fov_estimator == "camerahmr":
            self.camerahmr = CameraHMR()
        else:
            assert False, "not supported fov estimator {}".format(self.fov_estimator)

    def init_detector(self, detector_path, threshold):
        DETECTRON_CFG = os.path.join(detector_path, "cascade_mask_rcnn_vitdet_h_75ep.py")
        DETECTRON_CKPT = os.path.join(detector_path, "model_final_f05665.pkl")

        detectron2_cfg = LazyConfig.load(str(DETECTRON_CFG))
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = threshold
        detector = instantiate(detectron2_cfg.model)
        checkpointer = DetectionCheckpointer(detector)
        checkpointer.load(DETECTRON_CKPT)

        return detector
    
    def init_sam(self, sam_path):
        checkpoint = f"{sam_path}/checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        import sys
        sys.path.append(sam_path)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=self.device))
        return predictor
    
    def run_human_detection(
        self,
        img,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        default_to_full_image: bool = True,
    ):
        height, width = img.shape[:2]

        IMAGE_SIZE = 1024
        transforms = T.ResizeShortestEdge(short_edge_length=IMAGE_SIZE, max_size=IMAGE_SIZE)
        img_transformed = transforms(T.AugInput(img)).apply_image(img)
        img_transformed = torch.as_tensor(
            img_transformed.astype("float32").transpose(2, 0, 1)
        )
        inputs = {"image": img_transformed, "height": height, "width": width}

        with torch.no_grad():
            det_out = self.detector([inputs])

        det_instances = det_out[0]["instances"]
        valid_idx = (det_instances.pred_classes == det_cat_id) & (
            det_instances.scores > bbox_thr
        )
        if valid_idx.sum() == 0 and default_to_full_image:
            boxes = np.array([0, 0, width, height]).reshape(1, 4)
        else:
            boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

        # Sort boxes to keep a consistent output order
        sorted_indices = np.lexsort(
            (boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0])
        )  # shape: [len(boxes),]
        boxes = boxes[sorted_indices]
        return boxes
    
    def run_sam(self, img, boxes):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.sam_predictor.set_image(img)
            all_masks, all_scores = [], []
            for i in range(boxes.shape[0]):
                # First prediction: bbox only
                masks, scores, logits = self.sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=boxes[[i]],
                    multimask_output=True,
                )
                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind]
                scores = scores[sorted_ind]
                logits = logits[sorted_ind]

                mask_1 = masks[0]
                score_1 = scores[0]
                all_masks.append(mask_1)
                all_scores.append(score_1)

                # cv2.imwrite(os.path.join(save_dir, f"{os.path.basename(image_path)[:-4]}_mask_{i}.jpg"), (mask_1 * 255).astype(np.uint8))
            all_masks = np.stack(all_masks)
            all_scores = np.stack(all_scores)
        
        return all_masks, all_scores

    def process_one_image(
        self,
        img: Union[str, np.ndarray],
        bboxes: Optional[np.ndarray] = None,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
    ):
        """
        Perform model prediction in top-down format: assuming input is a full image.
        """

        # clear all cached results
        self.batch = None
        self.image_embeddings = None
        self.output = None
        self.prev_prompt = []
        torch.cuda.empty_cache()

        if type(img) == str:
            img = load_image(img, backend="cv2", image_format="bgr")
            image_format = "bgr"
        else:
            print ("####### Please make sure the input image is in RGB format")
            image_format = "rgb"
        height, width = img.shape[:2]

        if bboxes is not None:
            boxes = bboxes.reshape(-1, 4)
            self.is_crop = True
        elif self.use_detector:
            # boxes = np.array([0, 0, width, height]).reshape(1, 4)
            if image_format == "rgb":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                image_format = "bgr"
            print("Running object detector...")
            boxes = self.run_human_detection(
                img,
                det_cat_id,
                bbox_thr,
                nms_thr,
                default_to_full_image=False,
            )
            self.is_crop = True
        else:
            boxes = np.array([0, 0, width, height]).reshape(1, 4)

        # If there are no detected humans, don't run prediction
        if len(boxes) == 0:
            return []

        # The following models expect RGB images instead of BGR
        if image_format == "bgr":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get SAM2 mask if needed
        masks, masks_score = None, None
        if self.use_mask:
            masks, scores = self.run_sam(img, boxes)
            # Stress test demo --> use the same bbox for all instances
            boxes = np.concatenate([boxes[:, :2].min(axis=0), boxes[:, 2:].max(axis=0)], axis=0)[None, :].repeat(boxes.shape[0], axis=0)
    
        #################### Run model inference on an image ####################

        transform = Compose(
            [
                GetBBoxCenterScale(),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )
        # construct batch data samples
        data_list = []
        for idx in range(boxes.shape[0]):
            data_info = dict(img=img)
            data_info["bbox"] = boxes[idx]  # shape (4,)
            data_info["bbox_format"] = "xyxy"

            if masks is not None:
                data_info["mask"] = masks[idx].copy()
                if masks_score is not None:
                    data_info["mask_score"] = masks_score[idx]
                else:
                    data_info["mask_score"] = np.array(1.0, dtype=np.float32)
            else:
                data_info["mask"] = np.zeros((height, width, 1), dtype=np.uint8)
                data_info["mask_score"] = np.array(0.0, dtype=np.float32)

            data_list.append(transform(data_info))

        batch = default_collate(data_list)

        ############# Convert to full-image format #################
        full_size = 256
        _, img_full, center_full, scale_full = resize_image(
            img,
            full_size,
            batch["bbox_center"],
            batch["bbox_scale"],
        )
        bbox_min = center_full - scale_full * 0.5  # pyre-ignore
        bbox_max = center_full + scale_full * 0.5  # pyre-ignore
        bbox_full = np.concatenate([bbox_min, bbox_max], axis=1)
        bbox_full = np.clip(bbox_full, 0, full_size)

        batch_full = dict(
            img_full=to_tensor(img_full),
            full_bbox_center=center_full,
            full_bbox_scale=scale_full,
            full_bbox=bbox_full,
        )
        batch_full = default_collate([batch_full])
        batch.update(batch_full)

        max_num_person = batch["img"].shape[0]
        for key in [
            "img",
            "img_size",
            "ori_img_size",
            "bbox_center",
            "bbox_scale",
            "bbox",
            "affine_trans",
            "mask",
            "mask_score",
        ]:
            batch[key] = batch[key].unsqueeze(0).float()
        batch["mask"] = batch["mask"].unsqueeze(2)
        batch["person_valid"] = torch.ones((1, max_num_person))

        # FIXME: Use default camera intrinsics for now
        # batch["cam_int"] = torch.tensor(
        #     [[[(height ** 2 + width ** 2) ** 0.5, 0, width / 2.],
        #     [0, (height ** 2 + width ** 2) ** 0.5, height / 2.],
        #     [0, 0, 1]]],
        # ).to(batch["img"])

        #############################################################

        if self.cfg.MODEL.NAME == "promptable_threepo_triplet":
            batch['lhand_img'] = torch.zeros_like(batch['img'][:, :, :, :self.cfg.MODEL.HAND_IMAGE_SIZE[0], :self.cfg.MODEL.HAND_IMAGE_SIZE[1]])
            batch['rhand_img'] = torch.zeros_like(batch['img'][:, :, :, :self.cfg.MODEL.HAND_IMAGE_SIZE[0], :self.cfg.MODEL.HAND_IMAGE_SIZE[1]])
            batch['img_ori'] = [NoCollate(img)]

        batch = recursive_to(batch, "cuda")
        if self.cfg.MODEL.NAME == "promptable_threepo":
            self.model._initialize_batch(batch)
            with torch.no_grad():
                batch["cam_int"] = get_cam_intrinsics(self.camerahmr, batch).to(
                    batch["img"]
                )
                pose_output, full_output = self.model.forward_step(batch)
        else:
            # triplet
            assert self.model.use_twostage_for_hands
            self.model._initialize_batch(batch)
            with torch.no_grad():
                if self.fov_estimator == "moge":
                    assert hasattr(self, "moge_model"), "MoGe model not found!"
                    fov_model = self.moge_model
                else:
                    fov_model = self.camerahmr

                batch["cam_int"] = get_cam_intrinsics(fov_model, batch).to(
                    batch["img"]
                )

                pose_output_ab, full_output_ab = self.model.forward_step(batch)
                if self.just_left_hand:
                    print("Hack to only use left hand")
                    batch['left_center'] = batch["bbox_center"].cpu().numpy().squeeze(0)
                    batch['left_scale'] = batch["bbox_scale"].cpu().numpy().squeeze(0)
                    batch['right_center'] = np.array([[0, 0]], dtype=np.float32)
                    batch['right_scale'] = np.array([[0.5, 0.5]], dtype=np.float32)
                    # updated_batch['lhand_img'] = F.interpolate(
                    #     updated_batch['img'].flatten(0, 1),
                    #     size=updated_batch['lhand_img'].shape[-2:],
                    #     mode='bilinear'
                    # ).unflatten(0, updated_batch['img'].shape[:2])
                    # updated_batch['rhand_img'][...] = 0
                    updated_batch = self.model.get_hand_box(pose_output_ab, batch, scale_factor=self.scale_factor)
                    updated_batch['rhand_img'][...] = 0
                else:
                    updated_batch = self.model.get_hand_box(pose_output_ab, batch, scale_factor=self.scale_factor)
                pose_output, full_output = self.model.forward_step(updated_batch, return_feature_only=False)


        # Cache information for future prompting
        self.batch = copy.deepcopy(batch)
        self.pose_output = copy.deepcopy(pose_output)
        self.full_output = copy.deepcopy(full_output)

        out = pose_output["atlas"]
        print(out["face"])
        out = recursive_to(out, "cpu")
        out = recursive_to(out, "numpy")
        all_out = []
        for idx in range(max_num_person):
            all_out.append(
                {
                    "bbox": batch["bbox"][0, idx].cpu().numpy(),
                    "focal_length": out["focal_length"][idx],
                    "pred_keypoints_3d": out["pred_keypoints_3d"][idx],
                    "pred_vertices": out["pred_vertices"][idx],
                    "pred_cam_t": out["pred_cam_t"][idx],
                    "pred_keypoints_2d": out["pred_keypoints_2d"][idx],
                    "pred_pose_raw": out["pred_pose_raw"][idx],
                    "global_rot": out["global_rot"][idx],
                    "body_pose_params": out["body_pose"][idx],
                    "hand_pose_params": out["hand"][idx],
                    "scale_params": out["scale"][idx],
                    "shape_params": out["shape"][idx],
                    "expr_params": out["face"][idx],
                    "mask": masks[idx] if masks is not None else None,
                    "pred_joint_coords": out["pred_joint_coords"][idx],
                }
            )

            if self.cfg.MODEL.NAME == "promptable_threepo_triplet":
                all_out[-1]["lhand_bbox"] = np.array([
                    (updated_batch['left_center'][idx][0] - updated_batch['left_scale'][idx][0] / 2).item(),
                    (updated_batch['left_center'][idx][1] - updated_batch['left_scale'][idx][1] / 2).item(),
                    (updated_batch['left_center'][idx][0] + updated_batch['left_scale'][idx][0] / 2).item(),
                    (updated_batch['left_center'][idx][1] + updated_batch['left_scale'][idx][1] / 2).item(),
                ])
                all_out[-1]["rhand_bbox"] = np.array([
                    (updated_batch['right_center'][idx][0] - updated_batch['right_scale'][idx][0] / 2).item(),
                    (updated_batch['right_center'][idx][1] - updated_batch['right_scale'][idx][1] / 2).item(),
                    (updated_batch['right_center'][idx][0] + updated_batch['right_scale'][idx][0] / 2).item(),
                    (updated_batch['right_center'][idx][1] + updated_batch['right_scale'][idx][1] / 2).item(),
                ])

        return all_out
