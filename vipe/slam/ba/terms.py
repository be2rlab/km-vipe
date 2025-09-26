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

        self.image_size = image_size
        self.height, self.width = image_size

        assert embeddings.dim() == 4, "Embeddings must be flattened as (NV, C, H, W)"
        assert (
            embeddings.shape[2] == self.height and embeddings.shape[3] == self.width
        ), "Embeddings resolution must match BA image size"
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

        source_embeddings = self.embeddings[self.dense_disp_i_inds].float()
        target_embeddings = self.embeddings[self.dense_disp_j_inds].float()

        if self.embedding_valid_mask is not None:
            src_valid = self.embedding_valid_mask[self.dense_disp_i_inds].unsqueeze(-1).to(valid.dtype)
            tgt_valid = self.embedding_valid_mask[self.dense_disp_j_inds].unsqueeze(-1).to(valid.dtype)
            valid = valid * src_valid * tgt_valid

        valid_map = valid.permute(0, 3, 1, 2).float()

        # Bilinear interpolation weights
        u = coords[..., 0]
        v = coords[..., 1]
        u_clamped = u.clamp(0.0, self.width - 1.0)
        v_clamped = v.clamp(0.0, self.height - 1.0)
        x0 = torch.floor(u_clamped)
        y0 = torch.floor(v_clamped)
        x1 = torch.clamp(x0 + 1, max=self.width - 1)
        y1 = torch.clamp(y0 + 1, max=self.height - 1)

        du = (u_clamped - x0).unsqueeze(1)
        dv = (v_clamped - y0).unsqueeze(1)
        one_minus_du = 1.0 - du
        one_minus_dv = 1.0 - dv

        x0 = x0.long()
        y0 = y0.long()
        x1 = x1.long()
        y1 = y1.long()

        def gather(feat: torch.Tensor, yy: torch.Tensor, xx: torch.Tensor) -> torch.Tensor:
            flat = feat.view(self.n_terms, feat.shape[1], -1)
            idx = (yy * self.width + xx).view(self.n_terms, 1, -1)
            gathered = torch.gather(flat, 2, idx.expand(-1, feat.shape[1], -1))
            return gathered.view(self.n_terms, feat.shape[1], self.height, self.width)

        F00 = gather(target_embeddings, y0, x0)
        F10 = gather(target_embeddings, y0, x1)
        F01 = gather(target_embeddings, y1, x0)
        F11 = gather(target_embeddings, y1, x1)

        target_samples = (
            F00 * one_minus_du * one_minus_dv
            + F10 * du * one_minus_dv
            + F01 * one_minus_du * dv
            + F11 * du * dv
        )

        target_samples = target_samples * valid_map

        source_norm = F.normalize(source_embeddings, dim=1, eps=1e-6)
        target_norm = F.normalize(target_samples, dim=1, eps=1e-6)
        cos_sim = (source_norm * target_norm).sum(dim=1)

        residual_map = (1.0 - cos_sim) * valid_map.squeeze(1)

        norm_t = target_samples.norm(dim=1, keepdim=True).clamp(min=1e-6)
        grad_t = -(source_norm - cos_sim.unsqueeze(1) * target_norm) / norm_t
        grad_t = grad_t * valid_map

        dT_du = (F10 - F00) * one_minus_dv + (F11 - F01) * dv
        dT_dv = (F01 - F00) * one_minus_du + (F11 - F10) * du

        grad_u = (grad_t * dT_du).sum(dim=1)
        grad_v = (grad_t * dT_dv).sum(dim=1)
        grad_u = grad_u * valid_map.squeeze(1)
        grad_v = grad_v * valid_map.squeeze(1)

        grad_coords = torch.stack([grad_u, grad_v], dim=-1)

        residual = residual_map.view(self.n_terms, -1)
        weight = (self.embedding_weight * valid_map.squeeze(1)).view(self.n_terms, -1)

        J_dict = {}
        if jacobian:
            assert Ji is not None and Jj is not None and Jz is not None
            grad_coords_flat = grad_coords.view(self.n_terms, -1, 2)

            Ji_flat = Ji.view(self.n_terms, -1, 2, 6)
            Jj_flat = Jj.view(self.n_terms, -1, 2, 6)
            Jz_flat = Jz.view(self.n_terms, -1, 2, 1)

            J_pose_i = torch.einsum("nkd,nkdf->nkf", grad_coords_flat, Ji_flat)
            J_pose_j = torch.einsum("nkd,nkdf->nkf", grad_coords_flat, Jj_flat)
            J_disp = torch.einsum("nkd,nkdc->nkc", grad_coords_flat, Jz_flat)

            term_inds = torch.arange(self.n_terms, device=pose.device)
            J_dict = {
                "pose": SparseDenseBlockMatrix(
                    i_inds=torch.cat([term_inds, term_inds]),
                    j_inds=torch.cat([self.pose_i_inds, self.pose_j_inds]),
                    data=torch.cat([
                        J_pose_i.view(self.n_terms, -1, 6),
                        J_pose_j.view(self.n_terms, -1, 6),
                    ], dim=0),
                ),
                "dense_disp": SparseMDiagonalBlockMatrix(
                    i_inds=term_inds,
                    j_inds=self.dense_disp_i_inds,
                    data=J_disp.view(self.n_terms, -1, 1),
                ),
            }

            if optimize_intrinsics:
                assert Jfi is not None and Jfj is not None
                Jfi_flat = Jfi.view(self.n_terms, -1, 2, Jfi.shape[-1])
                Jfj_flat = Jfj.view(self.n_terms, -1, 2, Jfj.shape[-1])
                J_intr = torch.cat([
                    torch.einsum("nkd,nkdf->nkf", grad_coords_flat, Jfi_flat),
                    torch.einsum("nkd,nkdf->nkf", grad_coords_flat, Jfj_flat),
                ], dim=0)
                J_dict["intrinsics"] = SparseDenseBlockMatrix(
                    i_inds=torch.cat([term_inds, term_inds]),
                    j_inds=torch.cat([self.rig_i_inds, self.rig_j_inds]),
                    data=camera_model_cls.J_scale(
                        1.0 / self.intrinsics_factor,
                        J_intr,
                    ),
                )

            if optimize_rig:
                assert Jri is not None and Jrj is not None
                Jri_flat = Jri.view(self.n_terms, -1, 2, 6)
                Jrj_flat = Jrj.view(self.n_terms, -1, 2, 6)
                J_rig = torch.cat([
                    torch.einsum("nkd,nkdf->nkf", grad_coords_flat, Jri_flat),
                    torch.einsum("nkd,nkdf->nkf", grad_coords_flat, Jrj_flat),
                ], dim=0)
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
