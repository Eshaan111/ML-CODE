"""Build the frozen ResNet-34 spatial feature bank used by M1.ipynb."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, ResNetForImageClassification


class ImageDataset(Dataset):
    def __init__(self, image_dir, image_names, processor):
        self.image_dir = Path(image_dir)
        self.image_names = list(image_names)
        self.processor = processor

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        with Image.open(self.image_dir / image_name) as image:
            pixel_values = self.processor(
                images=image.convert("RGB"),
                return_tensors="pt",
            ).pixel_values.squeeze(0)
        return image_name, pixel_values


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build the feature cache")

    image_dir = Path("../datasets/Captioning/Images")
    cache_dir = Path("image_feature_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "resnet34_spatial_fp16.pt"

    if cache_path.exists():
        print(f"Cache already exists: {cache_path}")
        return

    image_names = sorted(path.name for path in image_dir.glob("*.jpg"))
    processor = AutoFeatureExtractor.from_pretrained(
        "microsoft/resnet-34",
        local_files_only=True,
    )
    dataset = ImageDataset(image_dir, image_names, processor)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    device = torch.device("cuda:0")
    resnet = ResNetForImageClassification.from_pretrained(
        "microsoft/resnet-34",
        local_files_only=True,
    ).resnet.to(device).eval()

    saved_names = []
    feature_batches = []

    with torch.inference_mode():
        for batch_number, (names, pixels) in enumerate(loader, start=1):
            pixels = pixels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features = resnet(pixels).last_hidden_state

            saved_names.extend(names)
            feature_batches.append(features.cpu().half())

            if batch_number % 20 == 0 or batch_number == len(loader):
                print(f"Cached {len(saved_names)}/{len(dataset)} images", flush=True)

    feature_bank = torch.cat(feature_batches)
    torch.save(
        {
            "model_name": "microsoft/resnet-34",
            "image_names": saved_names,
            "features": feature_bank,
        },
        cache_path,
    )
    print(f"Saved {tuple(feature_bank.shape)} to {cache_path}")


if __name__ == "__main__":
    main()
