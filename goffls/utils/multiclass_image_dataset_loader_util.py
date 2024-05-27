from numpy import empty
from pathlib import Path
from PIL import Image


def _get_images_count(dataset_folder: Path) -> int:
    return sum(1 for sub_path in dataset_folder.rglob("*") if sub_path.suffix in {".gif", ".jpg", ".jpeg", ".png"})


def _get_images_attributes(dataset_folder: Path) -> tuple:
    first_folder = [sub_path for sub_path in dataset_folder.iterdir() if sub_path.is_dir()][0]
    first_image_from_folder = [sub_path for sub_path in first_folder.rglob("*")
                               if sub_path.is_file() and sub_path.suffix in {".gif", ".jpg", ".jpeg", ".png"}][0]
    image = Image.open(fp=first_image_from_folder)
    width, height = image.size
    depth = len(image.getbands())
    return width, height, depth


def load_x_y_for_multiclass_image_dataset(dataset_root_folder: Path,
                                          phase: str) -> tuple:
    dataset_folder = dataset_root_folder.joinpath(phase)
    images_count = _get_images_count(dataset_folder)
    width, height, depth = _get_images_attributes(dataset_folder)
    derived_x_shape = (images_count, height, width, depth)
    derived_y_shape = (images_count, 1)
    x = empty(shape=derived_x_shape, dtype="uint8")
    y = empty(shape=derived_y_shape, dtype="uint8")
    index = 0
    for sub_path in dataset_folder.iterdir():
        if sub_path.is_dir():
            for inner_sub_path in sub_path.rglob("*"):
                if inner_sub_path.is_file() and inner_sub_path.suffix in {".gif", ".jpg", ".jpeg", ".png"}:
                    x[index] = Image.open(fp=inner_sub_path)
                    y[index] = int(sub_path.stem)
                    index += 1
    return x, y
