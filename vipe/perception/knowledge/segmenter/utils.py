import io
import torch
from torchvision.transforms.functional import resize
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from typing import List, Union, Tuple, Optional
import math
import rerun as rr
from PIL import Image


class SamResize:
    """
    Resize image to target size
    """

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        h, w, _ = image.shape
        long_side = max(h, w)
        if long_side != self.size:
            return self.apply_image(image)
        else:
            return image.permute(2, 0, 1)

    def apply_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects a torch tensor with shape HxWxC in float format.
        """

        target_size = self.get_preprocess_shape(
            image.shape[0], image.shape[1], self.size
        )
        return resize(image.permute(2, 0, 1), target_size)

    @staticmethod
    def get_preprocess_shape(
        oldh: int, oldw: int, long_side_length: int
    ) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size})"


def resize_longest_image_size(
    input_image_size: torch.Tensor, longest_side: int
) -> torch.Tensor:
    input_image_size = input_image_size.to(torch.float32)
    scale = longest_side / torch.max(input_image_size)
    transformed_size = scale * input_image_size
    transformed_size = torch.floor(transformed_size + 0.5).to(torch.int64)
    return transformed_size


def get_preprocess_shape(
    oldh: int, oldw: int, long_side_length: int
) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def apply_coords(coords, original_size, new_size):
    old_h, old_w = original_size
    new_h, new_w = new_size
    coords = deepcopy(coords).astype(float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords


def apply_boxes(boxes, original_size, new_size):
    boxes = apply_coords(boxes.reshape(-1, 2, 2), original_size, new_size)
    return boxes


def process_point_prompts(points: np.array, origin_image_size, input_size):
    batch_size = len(points)
    labels = np.ones((points.shape[0],), dtype=np.float32)
    point_coords = apply_coords(points, origin_image_size, input_size).astype(
        np.float32
    )
    point_coords_batched = point_coords.reshape(batch_size, 1, 2)  # (B,1,2)
    point_labels_batched = labels.reshape(batch_size, 1)  # (B,1)

    point_coords_batched = torch.from_numpy(point_coords_batched).to("cuda")
    point_labels_batched = torch.from_numpy(point_labels_batched).to("cuda")


def calculate_IoU(mask1, mask2):
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    iou = intersection / union.clamp(min=1e-6)
    return iou.item()


def calculate_IoU_batched(masks1: torch.Tensor, masks2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of masks in a batched manner.

    Args:
        masks1: Tensor of shape (N, H, W) or (N, 1, H, W), where N is the number of masks.
        masks2: Tensor of shape (M, H, W) or (M, 1, H, W), where M is the number of masks.

    Returns:
        IoU matrix of shape (N, M), where each element (i, j) is the IoU between masks1[i] and masks2[j].
    """
    # Ensure masks are 4D (N, 1, H, W) for consistency
    if masks1.dim() == 3:
        masks1 = masks1.unsqueeze(1)  # (N, 1, H, W)
    if masks2.dim() == 3:
        masks2 = masks2.unsqueeze(1)  # (M, 1, H, W)

    # Flatten masks to (N, H*W) and (M, H*W)
    masks1_flat = masks1.view(masks1.size(0), -1).float()  # (N, H*W)
    masks2_flat = masks2.view(masks2.size(0), -1).float()  # (M, H*W)

    # Compute intersection: (N, H*W) @ (H*W, M) -> (N, M)
    intersection = torch.mm(masks1_flat, masks2_flat.T)  # (N, M)

    # Compute union: (N, H*W) sum + (M, H*W) sum - intersection
    union = (
        masks1_flat.sum(dim=1, keepdim=True)
        + masks2_flat.sum(dim=1, keepdim=True).T
        - intersection
    )  # (N, M)

    # Compute IoU
    iou = intersection / union.clamp(min=1e-6)  # (N, M)

    return iou


def get_color(idx):
    cmap = plt.get_cmap("tab20")
    color = cmap(idx % 20)
    color = np.array(color)
    return color


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def split_array_into_batches(array, batch_size):
    n = array.shape[0]
    num_batches = n // batch_size + (1 if n % batch_size != 0 else 0)

    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n)
        batch = array[start_idx:end_idx]
        batches.append(batch)

    return batches


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def resize_and_pad_images(
    images: Union[np.ndarray, torch.Tensor, List[np.ndarray]], max_size: int = 256
) -> torch.Tensor:
    """
    Resize images while maintaining aspect ratio and pad to square with white color.

    Args:
        images: Can be one of:
            - np.ndarray of shape [N, 3, H, W]
            - torch.Tensor of shape [N, 3, H, W]
            - List of np.ndarray, each of shape [3, H, W]
        max_size (int, optional): Target size for both height and width. Defaults to 256.

    Returns:
        torch.Tensor: Resized and padded images of shape [N, 3, max_size, max_size]
    """

    def process_single_image(img: torch.Tensor) -> torch.Tensor:
        """Helper function to process a single image."""
        # Ensure image is in the correct format [C, H, W]
        if img.dim() == 2:
            img = img.unsqueeze(0)
        elif img.dim() == 3 and img.shape[0] != 3:
            img = img.permute(2, 0, 1)

        h, w = img.shape[1:]

        # Calculate scaling factor to maintain aspect ratio
        scale = min(max_size / h, max_size / w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize image
        resized = F.interpolate(
            img.unsqueeze(0),  # Add batch dimension
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # Remove batch dimension

        # Create white canvas
        result = torch.ones((3, max_size, max_size), dtype=torch.float32)

        # Calculate padding
        pad_h = (max_size - new_h) // 2
        pad_w = (max_size - new_w) // 2

        # Place the resized image in the center
        result[:, pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        return result

    # Handle list input
    if isinstance(images, list):
        processed_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).to(torch.float32)
            processed_images.append(process_single_image(img))
        return torch.stack(processed_images)

    # Handle batch input
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).to(torch.float32)

    # Process each image in the batch
    processed_images = []
    for i in range(len(images)):
        processed_images.append(process_single_image(images[i]))

    return torch.stack(processed_images)


def create_image_grid(
    images: torch.Tensor,
    nrow: Optional[int] = None,
    padding: int = 2,
    normalize: bool = True,
    pad_value: float = 1.0,
) -> torch.Tensor:
    """
    Create a grid of images from a batch of images.

    Args:
        images (torch.Tensor): Input images of shape [N, 3, H, W]
        nrow (Optional[int]): Number of images per row. If None, tries to make a square grid
        padding (int): Padding between images
        normalize (bool): Whether to normalize the input images to [0, 1]
        pad_value (float): Value for padding (default is 1.0 for white)

    Returns:
        torch.Tensor: A single image containing all input images in a grid [3, H_grid, W_grid]
    """
    # Input validation
    if not torch.is_tensor(images):
        raise TypeError("Input should be a torch tensor")
    if len(images.shape) != 4:
        raise ValueError(f"Input should be [N, C, H, W], got {images.shape}")

    # Normalize images if needed
    if normalize:
        images = images.float()
        if images.min() < 0 or images.max() > 1:
            images = (images - images.min()) / (images.max() - images.min())

    # Get dimensions
    N, C, H, W = images.shape

    # Calculate grid dimensions
    if nrow is None:
        nrow = int(math.ceil(math.sqrt(N)))
    ncol = math.ceil(N / nrow)

    # Calculate output dimensions
    grid_H = H * nrow + padding * (nrow - 1)
    grid_W = W * ncol + padding * (ncol - 1)

    # Create output tensor
    grid = torch.full(
        (C, grid_H, grid_W), pad_value, dtype=images.dtype, device=images.device
    )

    # Fill in the images
    for idx in range(N):
        i = idx // ncol
        j = idx % ncol
        h_start = i * (H + padding)
        w_start = j * (W + padding)
        grid[:, h_start : h_start + H, w_start : w_start + W] = images[idx]

    return grid


def show_image_grid(
    images: torch.Tensor,
    nrow: Optional[int] = None,
    title: str = "Image Grid",
    figsize: Tuple[int, int] = None,
    normalize: bool = True,
) -> None:
    """
    Display a grid of images using matplotlib.

    Args:
        images (torch.Tensor): Input images of shape [N, 3, H, W]
        nrow (Optional[int]): Number of images per row
        title (str): Title for the plot
        figsize (Tuple[int, int]): Figure size (width, height) in inches
        normalize (bool): Whether to normalize the input images to [0, 1]
    """
    import matplotlib.pyplot as plt

    # Move to CPU if needed
    if images.is_cuda:
        images = images.cpu()

    # Create the grid
    grid = create_image_grid(images, nrow, normalize=normalize)

    # Convert to numpy and transpose for plotting
    grid_np = grid.numpy()
    grid_np = np.transpose(grid_np, (1, 2, 0))

    # Ensure values are in valid range
    grid_np = np.clip(grid_np, 0, 1)

    # Calculate default figure size if not provided
    if figsize is None:
        aspect_ratio = grid_np.shape[1] / grid_np.shape[0]
        figsize = (min(15, 10 * aspect_ratio), min(10, 10 / aspect_ratio))

    # Create the plot
    plt.figure(figsize=figsize)
    plt.imshow(grid_np)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = np.array(Image.open(buf))
    rr.log("mapper/grid_objects", rr.Image(img))


def search_query(text_query_ft, objects):
    similarities = objects.compute_similarities(text_query_ft)

    # Find the indices of the top object with the highest scores
    top_k_indices = torch.topk(similarities, k=1, dim=-1).indices

    idx = top_k_indices[0].item()
    return idx
