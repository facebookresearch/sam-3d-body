# Temporary code to test model loading

from sam_3d_body import build_sam_3d_body_hf

model = build_sam_3d_body_hf("facebook/sam-3d-body")
outputs = model.process_one_image("image.jpg")