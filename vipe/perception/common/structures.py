from __future__ import annotations
from typing import List, Any, Optional, TypeAlias
from dataclasses import dataclass


@dataclass
class Timestamp:
    data: Any


@dataclass
class Pose:
    data: Any


@dataclass
class Image:
    data: Any


@dataclass
class Depth:
    data: Any


@dataclass
class RGBD:
    image: Image
    depth: Depth


@dataclass
class Intrinsics:
    data: Any


# @dataclass
# class KnowledgeFrame:
#     timestamp: Timestamp
#     pose: Pose
#     rgbd: RGBD
#     intrinsics: Intrinsics


@dataclass
class Embedding:
    data: Any
    image: Any
    text: Any


@dataclass
class Object:
    instance_id: Optional[int] = None
    # class_id: Optional[int] = None
    class_label: Optional[str] = None
    bbox2d: Optional[List[int]] = None
    bbox2d_confidence: Optional[List[float]] = None
    crop: Optional[Any] = None
    mask: Optional[Any] = None
    mask_image: Optional[Any] = None
    embedding: Optional[Embedding] = None


Objects: TypeAlias = List[Object]


@dataclass
class LocalKnowledge:
    objects: Optional[List[Object]] = None
