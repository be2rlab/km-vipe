# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from einops import rearrange

from vipe.ext.lietorch import SE3
from vipe.utils.cameras import CameraType

from ..maths import geom
from ..maths.matrix import SparseBlockMatrix, SparseDenseBlockMatrix, SparseMDiagonalBlockMatrix
from ..maths.vector import SparseBlockVector
from .kernel import RobustKernel


class TermEvalReturn(ABC):
    @abstractmethod
    def jtwj(self, group_name_row: str, group_name_col: str) -> SparseBlockMatrix: ...

    @abstractmethod
    def nwjtr(self, group_name: str) -> SparseBlockVector: ...

    @abstractmethod
    def remove_jcol_inds(self, group_name: str, col_inds: torch.Tensor): ...

    @abstractmethod
    def residual(self) -> torch.Tensor: ...

    def apply_robust_kernel(self, kernel: RobustKernel):
        raise NotImplementedError


@dataclass(kw_only=True)
class ConcreteTermEvalReturn(TermEvalReturn):
    J: dict[str, SparseBlockMatrix]  # group_name -> (n_occ, res_dim, manifold_dim)
    w: torch.Tensor  # (n_terms, res_dim, )
    r: torch.Tensor  # (n_terms, res_dim, )

    # n_occ = number of occurrences of this group_name in all the terms.
    # i.e. the number of blocks in the sparse Jacobian matrix with size n_terms x n_vars

    def jtwj(self, group_name_row: str, group_name_col: str) -> SparseBlockMatrix:
        wJ = self.J[group_name_col].scale_w_left(self.w)
        try:
            return self.J[group_name_row].tmult_mat(wJ).coalesce()
        except NotImplementedError:
            return wJ.tmult_mat(self.J[group_name_row]).transpose().coalesce()

    def nwjtr(self, group_name: str) -> SparseBlockVector:
        return self.J[group_name].tmult_vec(-self.w * self.r).coalesce()

    def remove_jcol_inds(self, group_name: str, col_inds: torch.Tensor):
        j_group = self.J[group_name]
        keep_mask = torch.isin(j_group.j_inds, col_inds, invert=True)
        self.J[group_name] = j_group.subset(keep_mask)

    def apply_robust_kernel(self, kernel: RobustKernel):
        robust_weight = kernel.apply(self.r)
        self.w = self.w * robust_weight

    def residual(self) -> torch.Tensor:
        return torch.sum(self.r * self.r * self.w, dim=1)


class SolverTerm(ABC):
    @abstractmethod
    def forward(self, variables: dict[str, Any], jacobian: bool = True) -> TermEvalReturn: ...

    @abstractmethod
    def group_names(self) -> set[str]: ...

    def update(self, solver):
        # Default implementation do nothing.
        pass


class DenseDepthFlowTerm(SolverTerm):
    """
    E(pose_pi, pose_pj, dense_disp_di, intr_qi, intr_qj) = \
        proj(rig_j.inv() * pose_j * pose_i.inv() * rig_i, dense_disp_di) - target_[ij di]

        Pose is the world2cam transform.
        Rig is the cam2world(central cam) transform.
        target_[ij di] is the target projected location.
    res_dim = H*W*2
    """

    def __init__(
        self,
        pose_i_inds: torch.Tensor,
        pose_j_inds: torch.Tensor,
        rig_i_inds: torch.Tensor,
        rig_j_inds: torch.Tensor,
        dense_disp_i_inds: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        intrinsics: torch.Tensor | None,
        intrinsics_factor: float,
        rig: SE3 | None,
        image_size: tuple[int, int],
        camera_type: CameraType,
    ) -> None:
        super().__init__()

        self.n_terms = pose_i_inds.shape[0]
        assert pose_i_inds.shape == (self.n_terms,)
        assert pose_j_inds.shape == (self.n_terms,)
        assert rig_i_inds.shape == (self.n_terms,)
        assert rig_j_inds.shape == (self.n_terms,)
        assert dense_disp_i_inds.shape == (self.n_terms,)

        self.pose_i_inds = pose_i_inds
        self.pose_j_inds = pose_j_inds
        self.rig_i_inds = rig_i_inds
        self.rig_j_inds = rig_j_inds
        self.dense_disp_i_inds = dense_disp_i_inds
        self.image_size = image_size
        self.camera_type = camera_type

        n_pixels = image_size[0] * image_size[1]

        self.target = target.reshape(self.n_terms, n_pixels, 2)  # (n_terms, H*W, 2)
        self.weight = weight.reshape(self.n_terms, n_pixels, 2)  # (n_terms, H*W, 2)
        self.intrinsics = intrinsics.reshape(-1, 4) if intrinsics is not None else None  # (Q, 4)
        self.intrinsics_factor = intrinsics_factor
        self.rig = rig

    def group_names(self) -> set[str]:
        names = {"pose", "dense_disp"}
        if self.intrinsics is None:
            names.add("intrinsics")
        if self.rig is None:
            names.add("rig")
        return names

    def forward(self, variables: dict[str, Any], jacobian: bool = True) -> TermEvalReturn:
        """
        variables contain:
            - pose: (n_var, ) SE3 of poses
            - dense_disp: (n_var, H*W) tensor of disparities
            - intrinsics: (Q, 4) tensor of intrinsics (optional)

        # TODO: To accelerate, you can return a PrecomputedTermEvalReturn with kernels from Droid-SLAM.
        """
        pose, dense_disp = variables["pose"], variables["dense_disp"]
        if optimize_intrinsics := self.intrinsics is None:
            intrinsics = variables["intrinsics"]
        else:
            intrinsics = self.intrinsics
        if optimize_rig := self.rig is None:
            rig = variables["rig"]
        else:
            rig = self.rig

        assert isinstance(pose, SE3) and isinstance(dense_disp, torch.Tensor)
        assert dense_disp.shape[1] == self.image_size[0] * self.image_size[1]
        assert intrinsics.shape[0] == rig.shape[0]

        camera_model_cls = self.camera_type.camera_model_cls()

        coords, valid, (Ji, Jj, Jz), (Jfi, Jfj), (Jri, Jrj) = geom.iproj_i_proj_j_disp(
            pose,
            dense_disp.view(-1, self.image_size[0], self.image_size[1]),
            None,
            (camera_model_cls(intrinsics).scaled(1.0 / self.intrinsics_factor).intrinsics),
            self.camera_type,
            rig,
            self.pose_i_inds,
            self.pose_j_inds,
            self.rig_i_inds,
            self.rig_j_inds,
            self.dense_disp_i_inds,
            jacobian_p_d=jacobian,
            jacobian_f=jacobian and optimize_intrinsics,
            jacobian_r=jacobian and optimize_rig,
        )
        coords = rearrange(coords, "n h w c -> n (h w) c", c=2)
        weight = rearrange(valid, "n h w 1 -> n (h w) 1") * self.weight  # (n_terms, H*W, 2)
        weight = rearrange(weight, "n hw c -> n (hw c)", c=2)

        J_dict = {}
        if jacobian:
            assert Ji is not None and Jj is not None and Jz is not None
            Ji = rearrange(Ji, "n h w c d -> n (h w c) d", c=2, d=6)
            Jj = rearrange(Jj, "n h w c d -> n (h w c) d", c=2, d=6)
            Jz = rearrange(Jz, "n h w c d -> n (h w) (c d)", c=2, d=1)
            term_inds = torch.arange(self.n_terms).to(pose.device)
            J_dict = {
                "pose": SparseDenseBlockMatrix(
                    i_inds=torch.cat([term_inds, term_inds]),
                    j_inds=torch.cat([self.pose_i_inds, self.pose_j_inds]),
                    data=torch.cat([Ji, Jj], dim=0),
                ),
                "dense_disp": SparseMDiagonalBlockMatrix(
                    i_inds=term_inds,
                    j_inds=self.dense_disp_i_inds,
                    data=Jz,
                ),
            }
            if optimize_intrinsics:
                assert Jfi is not None and Jfj is not None
                Jfi = rearrange(Jfi, "n h w c d -> n (h w c) d", c=2)
                Jfj = rearrange(Jfj, "n h w c d -> n (h w c) d", c=2)
                J_dict["intrinsics"] = SparseDenseBlockMatrix(
                    i_inds=torch.cat([term_inds, term_inds]),
                    j_inds=torch.cat([self.rig_i_inds, self.rig_j_inds]),
                    data=camera_model_cls.J_scale(
                        1.0 / self.intrinsics_factor,
                        torch.cat([Jfi, Jfj], dim=0),
                    ),
                )
            if optimize_rig:
                assert Jri is not None and Jrj is not None
                Jri = rearrange(Jri, "n h w c d -> n (h w c) d", c=2, d=6)
                Jrj = rearrange(Jrj, "n h w c d -> n (h w c) d", c=2, d=6)
                J_dict["rig"] = SparseDenseBlockMatrix(
                    i_inds=torch.cat([term_inds, term_inds]),
                    j_inds=torch.cat([self.rig_i_inds, self.rig_j_inds]),
                    data=torch.cat([Jri, Jrj], dim=0),
                )
        return ConcreteTermEvalReturn(
            J=J_dict,
            w=weight,
            r=rearrange(coords - self.target, "n hw c -> n (hw c)", c=2),
        )


class EmbeddingSimilarityTerm(SolverTerm):
    """Tightly couples pose and disparity updates with embedding similarity."""

    def __init__(
        self,
        *,
        pose_i_inds: torch.Tensor,
        pose_j_inds: torch.Tensor,
        rig_i_inds: torch.Tensor,
        rig_j_inds: torch.Tensor,
        dense_disp_i_inds: torch.Tensor,
        dense_disp_j_inds: torch.Tensor,
        embeddings: torch.Tensor,
        weight: float,
        image_size: tuple[int, int],
        camera_type: CameraType,
        embedding_valid_mask: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        intrinsics_factor: float = 8.0,
        rig: SE3 | None = None,
        chunk_size: int = 4,  # Process in chunks to save memory
    ) -> None:
        super().__init__()

        self.n_terms = pose_i_inds.shape[0]
        assert pose_i_inds.shape == (self.n_terms,)
        assert pose_j_inds.shape == (self.n_terms,)
        assert rig_i_inds.shape == (self.n_terms,)
        assert rig_j_inds.shape == (self.n_terms,)
        assert dense_disp_i_inds.shape == (self.n_terms,)
        assert dense_disp_j_inds.shape == (self.n_terms,)

        self.pose_i_inds = pose_i_inds
        self.pose_j_inds = pose_j_inds
        self.rig_i_inds = rig_i_inds
        self.rig_j_inds = rig_j_inds
        self.dense_disp_i_inds = dense_disp_i_inds
        self.dense_disp_j_inds = dense_disp_j_inds
        self.camera_type = camera_type

        self.embeddings = embeddings
        self.embedding_valid_mask = embedding_valid_mask
        self.embedding_weight = weight
        self.chunk_size = chunk_size

        self.image_size = image_size
        self.height, self.width = image_size

        assert embeddings.dim() == 4, "Embeddings must be flattened as (NV, C, H, W)"
        assert embeddings.shape[2] == self.height and embeddings.shape[3] == self.width, (
            "Embeddings resolution must match BA image size"
        )
        assert embedding_valid_mask is None or embedding_valid_mask.shape[1:] == image_size

        self.intrinsics = intrinsics.reshape(-1, 4) if intrinsics is not None else None
        self.intrinsics_factor = intrinsics_factor
        self.rig = rig

    def group_names(self) -> set[str]:
        names = {"pose", "dense_disp"}
        if self.intrinsics is None:
            names.add("intrinsics")
        if self.rig is None:
            names.add("rig")
        return names

    @staticmethod
    def bilinear_sample_with_grad(
        features: torch.Tensor, coords: torch.Tensor, return_grad: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Optimized bilinear sampling with analytical gradients.
        Uses efficient indexing and vectorized operations.

        Args:
            features: (N, C, H, W)
            coords: (N, H, W, 2) in (x, y) format in pixel coordinates
            return_grad: whether to compute gradients

        Returns:
            samples: (N, C, H, W)
            grad_u: (N, C, H, W) if return_grad else None
            grad_v: (N, C, H, W) if return_grad else None
        """
        N, C, H, W = features.shape

        u = coords[..., 0]
        v = coords[..., 1]

        # Clamp coordinates to valid range
        u_clamped = u.clamp(0.0, W - 1.0)
        v_clamped = v.clamp(0.0, H - 1.0)

        # Floor coordinates
        x0 = torch.floor(u_clamped).long()
        y0 = torch.floor(v_clamped).long()
        x1 = torch.clamp(x0 + 1, max=W - 1)
        y1 = torch.clamp(y0 + 1, max=H - 1)

        # Interpolation weights
        du = u_clamped - x0.float()
        dv = v_clamped - y0.float()

        # Expand for broadcasting
        du_exp = du.unsqueeze(1)  # (N, 1, H, W)
        dv_exp = dv.unsqueeze(1)

        # Gather corner values efficiently using gather
        flat_features = features.view(N, C, -1)
        idx_00 = (y0 * W + x0).view(N, 1, H, W).expand(-1, C, -1, -1)
        idx_10 = (y0 * W + x1).view(N, 1, H, W).expand(-1, C, -1, -1)
        idx_01 = (y1 * W + x0).view(N, 1, H, W).expand(-1, C, -1, -1)
        idx_11 = (y1 * W + x1).view(N, 1, H, W).expand(-1, C, -1, -1)

        F00 = torch.gather(flat_features, 2, idx_00.view(N, C, -1)).view(N, C, H, W)
        F10 = torch.gather(flat_features, 2, idx_10.view(N, C, -1)).view(N, C, H, W)
        F01 = torch.gather(flat_features, 2, idx_01.view(N, C, -1)).view(N, C, H, W)
        F11 = torch.gather(flat_features, 2, idx_11.view(N, C, -1)).view(N, C, H, W)

        # Bilinear interpolation weights
        w00 = (1 - du_exp) * (1 - dv_exp)
        w10 = du_exp * (1 - dv_exp)
        w01 = (1 - du_exp) * dv_exp
        w11 = du_exp * dv_exp

        samples = F00 * w00 + F10 * w10 + F01 * w01 + F11 * w11

        if not return_grad:
            return samples, None, None

        # Analytical gradients w.r.t. u and v
        grad_u = (F10 - F00) * (1 - dv_exp) + (F11 - F01) * dv_exp
        grad_v = (F01 - F00) * (1 - du_exp) + (F11 - F10) * du_exp

        return samples, grad_u, grad_v

    def forward(self, variables: dict[str, Any], jacobian: bool = True) -> TermEvalReturn:
        pose, dense_disp = variables["pose"], variables["dense_disp"]
        if optimize_intrinsics := self.intrinsics is None:
            intrinsics = variables["intrinsics"]
        else:
            intrinsics = self.intrinsics
        if optimize_rig := self.rig is None:
            rig = variables["rig"]
        else:
            rig = self.rig

        assert isinstance(pose, SE3) and isinstance(dense_disp, torch.Tensor)
        assert dense_disp.shape[1] == self.height * self.width
        assert intrinsics.shape[0] == rig.shape[0]

        camera_model_cls = self.camera_type.camera_model_cls()

        coords, valid, (Ji, Jj, Jz), (Jfi, Jfj), (Jri, Jrj) = geom.iproj_i_proj_j_disp(
            pose,
            dense_disp.view(-1, self.height, self.width),
            None,
            (camera_model_cls(intrinsics).scaled(1.0 / self.intrinsics_factor).intrinsics),
            self.camera_type,
            rig,
            self.pose_i_inds,
            self.pose_j_inds,
            self.rig_i_inds,
            self.rig_j_inds,
            self.dense_disp_i_inds,
            jacobian_p_d=jacobian,
            jacobian_f=jacobian and optimize_intrinsics,
            jacobian_r=jacobian and optimize_rig,
        )

        coords = coords.view(self.n_terms, self.height, self.width, 2)
        valid = valid.view(self.n_terms, self.height, self.width, 1)

        # Process in chunks to avoid OOM
        residual_chunks = []
        weight_chunks = []
        if jacobian:
            J_pose_i_chunks = []
            J_pose_j_chunks = []
            J_disp_chunks = []
            if optimize_intrinsics:
                J_intr_chunks = []
            if optimize_rig:
                J_rig_chunks = []

        pixel_count = self.height * self.width

        for chunk_start in range(0, self.n_terms, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, self.n_terms)
            chunk_len = chunk_end - chunk_start
            chunk_slice = slice(chunk_start, chunk_end)

            # Get chunk data
            src_idx = self.dense_disp_i_inds[chunk_slice]
            tgt_idx = self.dense_disp_j_inds[chunk_slice]

            # Keep embeddings in original dtype (likely float16)
            source_embeddings = self.embeddings[src_idx]
            target_embeddings = self.embeddings[tgt_idx]

            coords_chunk = coords[chunk_slice]
            valid_chunk = valid[chunk_slice]

            # Apply embedding valid mask
            if self.embedding_valid_mask is not None:
                src_valid = self.embedding_valid_mask[src_idx].unsqueeze(-1).to(valid_chunk.dtype)
                tgt_valid = self.embedding_valid_mask[tgt_idx].unsqueeze(-1).to(valid_chunk.dtype)
                valid_chunk = valid_chunk * src_valid * tgt_valid

            valid_map = valid_chunk.permute(0, 3, 1, 2)

            weight_chunk = (self.embedding_weight * valid_map.squeeze(1)).view(chunk_len, pixel_count)
            weight_chunks.append(weight_chunk)

            # Skip heavy computation for edges that have no valid overlap.
            valid_counts = valid_map.view(chunk_len, -1).sum(dim=1)
            active_mask = valid_counts > 0.0

            if not torch.any(active_mask):
                residual_chunks.append(weight_chunk.new_zeros(chunk_len, pixel_count))

                if jacobian:
                    zero_pose = coords_chunk.new_zeros(chunk_len, pixel_count, 6)
                    J_pose_i_chunks.append(zero_pose)
                    J_pose_j_chunks.append(zero_pose.clone())
                    J_disp_chunks.append(coords_chunk.new_zeros(chunk_len, pixel_count, 1))

                    if optimize_intrinsics:
                        intr_dim = Jfi.shape[-1]
                        zero_intr = coords_chunk.new_zeros(chunk_len, pixel_count, intr_dim)
                        J_intr_chunks.append(torch.cat([zero_intr, zero_intr.clone()], dim=0))

                    if optimize_rig:
                        zero_rig = coords_chunk.new_zeros(chunk_len, pixel_count, 6)
                        J_rig_chunks.append(torch.cat([zero_rig, zero_rig.clone()], dim=0))

                continue

            active_inds = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)

            source_embeddings_f32 = source_embeddings[active_inds].float()
            target_embeddings_f32 = target_embeddings[active_inds].float()

            coords_active = coords_chunk[active_inds]
            valid_chunk_active = valid_chunk[active_inds]
            valid_map_active = valid_map[active_inds]

            # Bilinear sampling with gradients
            target_samples, grad_u, grad_v = self.bilinear_sample_with_grad(
                target_embeddings_f32, coords_active, return_grad=jacobian
            )

            # Apply valid mask before normalization
            valid_mask_expanded = valid_map_active.expand_as(target_samples)
            target_samples = target_samples * valid_mask_expanded

            # Normalize embeddings
            source_norm = F.normalize(source_embeddings_f32, dim=1, eps=1e-8)
            target_norm = F.normalize(target_samples, dim=1, eps=1e-8)

            # Cosine similarity: dot product of normalized vectors
            cos_sim = (source_norm * target_norm).sum(dim=1)  # (N, H, W)

            # Residual: we want to minimize (1 - cos_sim), so residual = 1 - cos_sim
            residual_map = (1.0 - cos_sim) * valid_map_active.squeeze(1)

            residual_chunk = weight_chunk.new_zeros(chunk_len, pixel_count)
            residual_chunk[active_inds] = residual_map.view(-1, pixel_count)
            residual_chunks.append(residual_chunk)

            if jacobian:
                # Compute gradient of loss w.r.t. unnormalized target samples
                # Loss = 1 - cos_sim = 1 - (S_norm · T_norm)
                # where S_norm = S/|S|, T_norm = T/|T|
                #
                # For normalized vectors, the gradient w.r.t. unnormalized T is:
                # d/dT [1 - (S_norm · T_norm)] = -d/dT [(S_norm · T_norm)]
                #
                # Using chain rule through normalization:
                # d/dT [S_norm · T_norm] = (I - T_norm @ T_norm^T) / |T| @ S_norm
                #                        = (S_norm - (S_norm · T_norm) * T_norm) / |T|
                #                        = (S_norm - cos_sim * T_norm) / |T|
                #
                # So: d/dT [1 - cos_sim] = -(S_norm - cos_sim * T_norm) / |T|
                norm_t = target_samples.norm(dim=1, keepdim=True).clamp(min=1e-8)
                grad_t = -(source_norm - cos_sim.unsqueeze(1) * target_norm) / norm_t
                grad_t = grad_t * valid_mask_expanded

                # Chain rule: gradient w.r.t. coordinates
                # d/dcoords = d/dT * dT/dcoords
                grad_u_weighted = (grad_t * grad_u).sum(dim=1) * valid_map_active.squeeze(1)
                grad_v_weighted = (grad_t * grad_v).sum(dim=1) * valid_map_active.squeeze(1)
                grad_coords = torch.stack([grad_u_weighted, grad_v_weighted], dim=-1)
                grad_coords_flat = grad_coords.view(-1, pixel_count, 2)

                # Jacobians
                Ji_active = Ji[chunk_slice][active_inds].view(-1, pixel_count, 2, 6)
                Jj_active = Jj[chunk_slice][active_inds].view(-1, pixel_count, 2, 6)
                Jz_active = Jz[chunk_slice][active_inds].view(-1, pixel_count, 2, 1)

                chunk_J_pose_i = coords_chunk.new_zeros(chunk_len, pixel_count, 6)
                chunk_J_pose_j = coords_chunk.new_zeros(chunk_len, pixel_count, 6)
                chunk_J_disp = coords_chunk.new_zeros(chunk_len, pixel_count, 1)

                chunk_J_pose_i[active_inds] = torch.einsum("nkd,nkdf->nkf", grad_coords_flat, Ji_active)
                chunk_J_pose_j[active_inds] = torch.einsum("nkd,nkdf->nkf", grad_coords_flat, Jj_active)
                chunk_J_disp[active_inds] = torch.einsum("nkd,nkdc->nkc", grad_coords_flat, Jz_active)

                J_pose_i_chunks.append(chunk_J_pose_i)
                J_pose_j_chunks.append(chunk_J_pose_j)
                J_disp_chunks.append(chunk_J_disp)

                if optimize_intrinsics:
                    intr_dim = Jfi.shape[-1]
                    Jfi_active = Jfi[chunk_slice][active_inds].view(-1, pixel_count, 2, intr_dim)
                    Jfj_active = Jfj[chunk_slice][active_inds].view(-1, pixel_count, 2, intr_dim)

                    chunk_J_intr_i = coords_chunk.new_zeros(chunk_len, pixel_count, intr_dim)
                    chunk_J_intr_j = coords_chunk.new_zeros(chunk_len, pixel_count, intr_dim)

                    chunk_J_intr_i[active_inds] = torch.einsum("nkd,nkdf->nkf", grad_coords_flat, Jfi_active)
                    chunk_J_intr_j[active_inds] = torch.einsum("nkd,nkdf->nkf", grad_coords_flat, Jfj_active)

                    J_intr_chunks.append(torch.cat([chunk_J_intr_i, chunk_J_intr_j], dim=0))

                if optimize_rig:
                    Jri_active = Jri[chunk_slice][active_inds].view(-1, pixel_count, 2, 6)
                    Jrj_active = Jrj[chunk_slice][active_inds].view(-1, pixel_count, 2, 6)

                    chunk_J_rig_i = coords_chunk.new_zeros(chunk_len, pixel_count, 6)
                    chunk_J_rig_j = coords_chunk.new_zeros(chunk_len, pixel_count, 6)

                    chunk_J_rig_i[active_inds] = torch.einsum("nkd,nkdf->nkf", grad_coords_flat, Jri_active)
                    chunk_J_rig_j[active_inds] = torch.einsum("nkd,nkdf->nkf", grad_coords_flat, Jrj_active)

                    J_rig_chunks.append(torch.cat([chunk_J_rig_i, chunk_J_rig_j], dim=0))

        # Concatenate results
        residual = torch.cat(residual_chunks, dim=0)
        weight = torch.cat(weight_chunks, dim=0)

        J_dict = {}
        if jacobian:
            term_inds = torch.arange(self.n_terms, device=pose.device)

            J_pose_i = torch.cat(J_pose_i_chunks, dim=0)
            J_pose_j = torch.cat(J_pose_j_chunks, dim=0)
            J_disp = torch.cat(J_disp_chunks, dim=0)

            J_dict = {
                "pose": SparseDenseBlockMatrix(
                    i_inds=torch.cat([term_inds, term_inds]),
                    j_inds=torch.cat([self.pose_i_inds, self.pose_j_inds]),
                    data=torch.cat(
                        [
                            J_pose_i.view(self.n_terms, -1, 6),
                            J_pose_j.view(self.n_terms, -1, 6),
                        ],
                        dim=0,
                    ),
                ),
                "dense_disp": SparseMDiagonalBlockMatrix(
                    i_inds=term_inds,
                    j_inds=self.dense_disp_i_inds,
                    data=J_disp.view(self.n_terms, -1, 1),
                ),
            }

            if optimize_intrinsics:
                J_intr = torch.cat(J_intr_chunks, dim=0)
                J_dict["intrinsics"] = SparseDenseBlockMatrix(
                    i_inds=torch.cat([term_inds, term_inds]),
                    j_inds=torch.cat([self.rig_i_inds, self.rig_j_inds]),
                    data=camera_model_cls.J_scale(
                        1.0 / self.intrinsics_factor,
                        J_intr,
                    ),
                )

            if optimize_rig:
                J_rig = torch.cat(J_rig_chunks, dim=0)
                J_dict["rig"] = SparseDenseBlockMatrix(
                    i_inds=torch.cat([term_inds, term_inds]),
                    j_inds=torch.cat([self.rig_i_inds, self.rig_j_inds]),
                    data=J_rig,
                )

        return ConcreteTermEvalReturn(J=J_dict, w=weight, r=residual)


class DispSensRegularizationTerm(SolverTerm):
    """
    E(dense_disp_i) = dense_disp_i - dense_disps_sens_i
    res_dim = H*W
    """

    @dataclass(kw_only=True)
    class ThisTermEvalReturn(TermEvalReturn):
        alpha: float
        i_inds: torch.Tensor
        disps_sens_res: torch.Tensor

        def jtwj(self, group_name_row: str, group_name_col: str) -> SparseBlockMatrix:
            assert group_name_row == group_name_col == "dense_disp"
            return SparseMDiagonalBlockMatrix(
                i_inds=self.i_inds,
                j_inds=self.i_inds,
                data=torch.full_like(self.disps_sens_res, self.alpha).unsqueeze(-1),
            )

        def nwjtr(self, group_name: str) -> SparseBlockVector:
            assert group_name == "dense_disp"
            return SparseBlockVector(inds=self.i_inds, data=-self.alpha * self.disps_sens_res)

        def remove_jcol_inds(self, group_name: str, col_inds: torch.Tensor):
            assert group_name == "dense_disp"
            keep_mask = torch.isin(self.i_inds, col_inds, invert=True)
            self.i_inds = self.i_inds[keep_mask]
            self.disps_sens_res = self.disps_sens_res[keep_mask]

        def residual(self) -> torch.Tensor:
            return self.alpha * (self.disps_sens_res**2).sum(dim=1)

    def __init__(self, i_inds: torch.Tensor, alpha: float, disps_sens: torch.Tensor) -> None:
        super().__init__()

        self.i_inds = i_inds
        self.alpha = alpha
        self.disps_sens = disps_sens

    def group_names(self) -> set[str]:
        return {"dense_disp"}

    def forward(self, variables: dict[str, Any], jacobian: bool = True) -> TermEvalReturn:
        """
        variables contain:
            - dense_disp: (n_var, H*W) tensor of disparities
        """
        dense_disp = variables["dense_disp"]

        assert isinstance(dense_disp, torch.Tensor)
        assert dense_disp.shape == self.disps_sens.shape

        return self.ThisTermEvalReturn(
            alpha=self.alpha,
            i_inds=self.i_inds,
            disps_sens_res=dense_disp[self.i_inds] - self.disps_sens[self.i_inds],
        )


class TracksFlowTerm(SolverTerm):
    """
    E (pose_pi, pose_pj, tracks_di, intr_qi, intr_qj) = \
        proj(rig_j.inv() * pose_j * pose_i.inv() * rig_i, tracks_di) - target_[ij di]

        Pose is the world2cam transform.
        Rig is the cam2world(central cam) transform.
        target_[ij di] is the target projected location.
    res_dim = n_tracks*2
    """

    def __init__(
        self,
        pose_i_inds: torch.Tensor,
        pose_j_inds: torch.Tensor,
        rig_i_inds: torch.Tensor,
        rig_j_inds: torch.Tensor,
        tracks_i_inds: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        tracks_uv: torch.Tensor,
        intrinsics: torch.Tensor | None,
        rig: SE3,
        camera_type: CameraType,
    ) -> None:
        super().__init__()

        self.n_terms = pose_i_inds.shape[0]
        assert pose_i_inds.shape == (self.n_terms,)
        assert pose_j_inds.shape == (self.n_terms,)
        assert rig_i_inds.shape == (self.n_terms,)
        assert rig_j_inds.shape == (self.n_terms,)
        assert tracks_i_inds.shape == (self.n_terms,)

        self.pose_i_inds = pose_i_inds
        self.pose_j_inds = pose_j_inds
        self.rig_i_inds = rig_i_inds
        self.rig_j_inds = rig_j_inds
        self.tracks_i_inds = tracks_i_inds
        self.camera_type = camera_type

        self.target = target.reshape(self.n_terms, -1, 2)  # (n_terms, n_tracks, 2)
        self.weight = weight.reshape(self.n_terms, -1, 2)  # (n_terms, n_tracks, 2)
        self.tracks_uv = tracks_uv

        self.n_tracks = self.target.shape[1]
        assert self.target.shape[1] == self.n_tracks
        assert self.weight.shape[1] == self.n_tracks
        assert self.tracks_uv.shape[1] == self.n_tracks

        self.intrinsics = intrinsics.reshape(-1, 4) if intrinsics is not None else None
        self.rig = rig

    def group_names(self) -> set[str]:
        names = {"pose", "tracks_disp"}
        if self.intrinsics is None:
            names.add("intrinsics")
        return names

    def forward(self, variables: dict[str, Any], jacobian: bool = True) -> TermEvalReturn:
        """
        variables contain:
            - pose: (n_var, ) SE3 of poses
            - tracks_disp: (n_var, n_tracks) tensor of disparities
            - intrinsics: (Q, 4) tensor of intrinsics (optional)
        """
        pose, tracks_disp = variables["pose"], variables["tracks_disp"]
        if optimize_intrinsics := self.intrinsics is None:
            intrinsics = variables["intrinsics"]
        else:
            intrinsics = self.intrinsics

        assert isinstance(pose, SE3) and isinstance(tracks_disp, torch.Tensor)
        assert tracks_disp.shape[1] == self.n_tracks
        assert intrinsics.shape[0] == self.rig.shape[0]

        coords, valid, (Ji, Jj, Jz), (Jfi, Jfj), _ = geom.iproj_i_proj_j_disp(
            pose,
            tracks_disp,
            self.tracks_uv,
            intrinsics,
            self.camera_type,
            self.rig,
            self.pose_i_inds,
            self.pose_j_inds,
            self.rig_i_inds,
            self.rig_j_inds,
            self.tracks_i_inds,
            jacobian_p_d=jacobian,
            jacobian_f=jacobian and optimize_intrinsics,
            jacobian_r=False,
        )
        weight = self.weight * valid

        J_dict = {}
        if jacobian:
            assert Ji is not None and Jj is not None and Jz is not None
            Ji = rearrange(Ji, "n t c d -> n (t c) d", c=2, d=6)
            Jj = rearrange(Jj, "n t c d -> n (t c) d", c=2, d=6)
            Jz = rearrange(Jz, "n t c d -> n (t) (c d)", c=2, d=1)
            term_inds = torch.arange(self.n_terms).to(pose.device)
            J_dict = {
                "pose": SparseDenseBlockMatrix(
                    i_inds=torch.cat([term_inds, term_inds]),
                    j_inds=torch.cat([self.pose_i_inds, self.pose_j_inds]),
                    data=torch.cat([Ji, Jj], dim=0),
                ),
                "tracks_disp": SparseMDiagonalBlockMatrix(
                    i_inds=term_inds,
                    j_inds=self.tracks_i_inds,
                    data=Jz,
                ),
            }
            if optimize_intrinsics:
                assert Jfi is not None and Jfj is not None
                Jfi = rearrange(Jfi, "n t c d -> n (t c) d", c=2)
                Jfj = rearrange(Jfj, "n t c d -> n (t c) d", c=2)
                J_dict["intrinsics"] = SparseDenseBlockMatrix(
                    i_inds=torch.cat([term_inds, term_inds]),
                    j_inds=torch.cat([self.rig_i_inds, self.rig_j_inds]),
                    data=torch.cat([Jfi, Jfj], dim=0),
                )

        return ConcreteTermEvalReturn(
            J=J_dict,
            w=weight.view(self.n_terms, -1),
            r=rearrange(coords - self.target, "n t c -> n (t c)", c=2),
        )
