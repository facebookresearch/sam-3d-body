import torch
import numpy as np

from MHR.mhr.mhr import MHR

class MHR3POWrapper(MHR):
    def forward(
        self,
        identity_coeffs: torch.Tensor,
        model_parameters: torch.Tensor,
        face_expr_coeffs: torch.Tensor | None,
        apply_correctives: bool = True,
        return_joint_params: bool = True,
        return_skel_state: bool = True,
    ) -> torch.Tensor:
        """Compute vertices given input parameters."""

        assert (
            len(identity_coeffs.shape) == 2
        ), f"Expected batched (n_rows >= 1) identity coeffs with {self.identity_model.blend_shapes.shape[0]} columns, got {identity_coeffs.shape}"
        apply_face_expressions = (
            self.face_expressions_model is not None and face_expr_coeffs is not None
        )
        apply_correctives = (
            apply_correctives and self.pose_correctives_model is not None
        )

        if apply_face_expressions:
            assert (
                len(face_expr_coeffs.shape) == 2
            ), f"Expected batched (n_rows >= 1) face expressions coeffs with {self.face_expressions_model.blend_shapes.shape[0]} columns, got {face_expr_coeffs.shape}"

        # Compute identity vertices in rest pose
        identity_rest_pose = self.identity_model.forward(identity_coeffs)

        # Compute joint parameters (local) and skeleton state (global)
        joint_parameters = self.character_gpu.model_parameters_to_joint_parameters(
            model_parameters
        )
        skel_state = self.character_gpu.joint_parameters_to_skeleton_state(
            joint_parameters
        )

        # Apply face expressions
        linear_model_unposed = None
        if apply_face_expressions:
            face_expressions = self.face_expressions_model.forward(face_expr_coeffs)
            linear_model_unposed = identity_rest_pose + face_expressions

        # Apply pose correctives
        if apply_correctives:
            linear_model_pose_correctives = self.pose_correctives_model.forward(
                joint_parameters=joint_parameters
            )
            linear_model_unposed = (
                identity_rest_pose + linear_model_pose_correctives
                if linear_model_unposed is None
                else linear_model_unposed + linear_model_pose_correctives
            )

        if linear_model_unposed is None:
            # i.e. (not apply_face_expressions) and (not apply_correctives):
            linear_model_unposed = identity_rest_pose.expand(
                skel_state.shape[0], -1, -1
            )

        # Compute vertices
        verts = self.character_gpu.skin_points(
            skel_state=skel_state, rest_vertex_positions=linear_model_unposed
        )

        # Set up returns
        breakpoint()
        to_return = [verts]
        if return_joint_params:
            to_return.append(joint_parameters)
        if return_skel_state:
            to_return.append(skel_state)

        if isinstance(to_return, list) and len(to_return) == 1:
            return to_return[0]
        else:
            return tuple(to_return)