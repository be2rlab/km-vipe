import numpy as np
import json
import rerun as rr
from pathlib import Path
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def load_point_cloud(pcd_path):
    """Load point cloud from .pcd file using Open3D"""
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    points = np.asarray(pcd.points)
    
    # Try to load RGB colors if available
    colors = None
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        # Convert from [0,1] to [0,255] if needed
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)
    
    return points, colors

def load_semantic_labels(npy_path):
    """Load semantic labels from .npy file"""
    semantic_ids = np.load(npy_path)
    return semantic_ids

def load_annotations(json_path):
    """Load semantic class annotations from JSON file"""
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    return annotations

def generate_colors_for_classes(num_classes):
    """Generate distinct colors for each semantic class"""
    # Use a colormap to generate distinct colors
    cmap = plt.cm.tab20  # Good for up to 20 classes
    if num_classes > 20:
        cmap = plt.cm.hsv
    
    colors = []
    for i in range(num_classes):
        color = cmap(i / max(num_classes - 1, 1))
        colors.append([int(c * 255) for c in color[:3]])  # Convert to RGB 0-255
    
    return colors

def visualize_semantic_pointcloud(pcd_path, semantic_path, annotations_path, app_id="semantic_pointcloud"):
    """
    Visualize semantic segmented point cloud using Rerun with RGB toggle capability
    
    Args:
        pcd_path: Path to .pcd file
        semantic_path: Path to semantic.npy file
        annotations_path: Path to annotations.json file
        app_id: Application ID for Rerun
    """
    
    # Initialize Rerun
    rr.init(app_id, spawn=True)
    
    # Load data
    print("Loading point cloud...")
    points, rgb_colors = load_point_cloud(pcd_path)
    print(rgb_colors)
    
    print("Loading semantic labels...")
    semantic_ids = load_semantic_labels(semantic_path)
    
    print("Loading annotations...")
    annotations = load_annotations(annotations_path)
    
    # Validate data consistency
    if len(points) != len(semantic_ids):
        raise ValueError(f"Point cloud has {len(points)} points but semantic labels has {len(semantic_ids)} labels")
    
    # Convert string keys to integers and create class mapping
    class_mapping = {int(k): v for k, v in annotations.items()}
    num_classes = len(class_mapping)
    
    print(f"Found {num_classes} semantic classes:")
    for class_id, class_name in class_mapping.items():
        count = np.sum(semantic_ids == class_id)
        print(f"  {class_id}: {class_name} ({count} points)")
    
    # Check if RGB colors are available
    has_rgb = rgb_colors is not None
    if has_rgb:
        print(f"RGB colors loaded successfully ({rgb_colors.shape})")
    else:
        print("No RGB colors found in point cloud")
    
    # Generate colors for each semantic class
    class_colors = generate_colors_for_classes(num_classes)
    
    # Create semantic color array for points
    semantic_point_colors = np.zeros((len(points), 3), dtype=np.uint8)
    
    for class_id, class_name in class_mapping.items():
        mask = semantic_ids == class_id
        if class_id < len(class_colors):
            semantic_point_colors[mask] = class_colors[class_id]
        else:
            # Fallback color if we have more classes than predefined colors
            semantic_point_colors[mask] = [128, 128, 128]  # Gray
    
    # Log the semantic colored point cloud
    rr.log(
        "pointcloud/semantic_colored",
        rr.Points3D(
            positions=points,
            colors=semantic_point_colors,
            radii=0.01
        )
    )
    
    # Log RGB colored point cloud if available
    if has_rgb:
        rr.log(
            "pointcloud/rgb_colored", 
            rr.Points3D(
                positions=points,
                colors=rgb_colors,
                radii=0.01
            )
        )
        
        # Log combined view with both RGB and semantic (semantic with transparency)
        rr.log(
            "pointcloud/rgb_with_semantic_overlay",
            rr.Points3D(
                positions=points,
                colors=rgb_colors,
                radii=0.008  # Slightly smaller for the base layer
            )
        )
        
        # Add semantic overlay with slight transparency effect by using different radii
        for class_id, class_name in class_mapping.items():
            mask = semantic_ids == class_id
            if np.any(mask):
                class_points = points[mask]
                class_color = class_colors[class_id] if class_id < len(class_colors) else [128, 128, 128]
                
                rr.log(
                    f"pointcloud/rgb_with_semantic_overlay/semantic_{class_name}",
                    rr.Points3D(
                        positions=class_points,
                        colors=[class_color] * len(class_points),
                        radii=0.012  # Slightly larger for overlay effect
                    )
                )
    
    # Log individual semantic classes as separate entities
    for class_id, class_name in class_mapping.items():
        mask = semantic_ids == class_id
        if np.any(mask):
            class_points = points[mask]
            class_color = class_colors[class_id] if class_id < len(class_colors) else [128, 128, 128]
            
            # Semantic colored version
            rr.log(
                f"pointcloud/classes_semantic/{class_name}",
                rr.Points3D(
                    positions=class_points,
                    colors=[class_color] * len(class_points),
                    radii=0.01
                )
            )
            
            # RGB colored version if available
            if has_rgb:
                class_rgb_colors = rgb_colors[mask]
                rr.log(
                    f"pointcloud/classes_rgb/{class_name}",
                    rr.Points3D(
                        positions=class_points,
                        colors=class_rgb_colors,
                        radii=0.01
                    )
                )
    
    # Log class information as text
    class_info = "## Semantic Classes\n\n"
    for class_id, class_name in class_mapping.items():
        count = np.sum(semantic_ids == class_id)
        percentage = (count / len(semantic_ids)) * 100
        color = class_colors[class_id] if class_id < len(class_colors) else [128, 128, 128]
        class_info += f"**{class_name}** (ID: {class_id}): {count} points ({percentage:.1f}%)\n\n"
    
    # Add RGB information
    if has_rgb:
        class_info += "\n## Visualization Options\n\n"
        class_info += "- **semantic_colored**: Points colored by semantic class\n"
        class_info += "- **rgb_colored**: Original RGB colors from point cloud\n"
        class_info += "- **rgb_with_semantic_overlay**: RGB base with semantic overlay\n"
        class_info += "- **classes_semantic/**: Individual classes with semantic colors\n"
        class_info += "- **classes_rgb/**: Individual classes with original RGB colors\n"
    else:
        class_info += "\n*Note: No RGB colors available in point cloud*\n"
    
    rr.log("info/visualization_guide", rr.TextDocument(class_info, media_type=rr.MediaType.MARKDOWN))
    
    # Set up the 3D view
    rr.log("", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    
    print(f"\nVisualization complete! Rerun viewer should open automatically.")
    print(f"Total points: {len(points)}")
    
    if has_rgb:
        print("\n=== Available Visualizations ===")
        print("1. pointcloud/semantic_colored - Semantic class colors")
        print("2. pointcloud/rgb_colored - Original RGB colors") 
        print("3. pointcloud/rgb_with_semantic_overlay - RGB with semantic overlay")
        print("4. pointcloud/classes_semantic/ - Individual semantic classes")
        print("5. pointcloud/classes_rgb/ - Individual classes with RGB colors")
        print("\nToggle different views on/off in the Rerun viewer sidebar!")
    else:
        print("Only semantic coloring available (no RGB data in point cloud)")
        print("Toggle individual classes on/off in the Rerun viewer sidebar.")

def main():
    """Example usage"""
    # Update these paths to match your file locations
    pcd_path = "results/visuals/osma-bench/conceptgraphs/replica_cad/baseline/apt_0/pointcloud.pcd"
    semantic_path = "results/visuals/osma-bench/conceptgraphs/replica_cad/baseline/apt_0/semantic.npy"
    annotations_path = "results/visuals/osma-bench/conceptgraphs/replica_cad/baseline/apt_0/annotations.json"

    # Check if files exist
    for path, name in [(pcd_path, "Point cloud"), (semantic_path, "Semantic labels"), (annotations_path, "Annotations")]:
        if not Path(path).exists():
            print(f"Error: {name} file not found at {path}")
            return
    
    try:
        visualize_semantic_pointcloud(pcd_path, semantic_path, annotations_path)
    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == "__main__":
    main()