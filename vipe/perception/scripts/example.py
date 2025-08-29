import sys
import cv2 as cv
import torch
import os
import numpy as np

import random
from tqdm import tqdm

from vipe.perception.knowledge.processor import Grounder  # noqa: E402


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    seed_torch(42)
    processor = Grounder()
    img_paths = ["/home/jaafar/dev/be2r/data/000000.jpg"]
    imgs = [cv.imread(img_path) for img_path in img_paths]
    imgs = [cv.cvtColor(img, cv.COLOR_BGR2RGB) for img in imgs]
    # img = torch.from_numpy(img)

    for i in tqdm(range(1000)):
        processor.process_image(imgs[0], visualize=True)
