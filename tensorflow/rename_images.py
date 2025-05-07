#!/usr/bin/env python3
import os
import argparse

def rename_images_in_dataset(root_dir: str, prefix: str = ""):
    """
    Traverse root_dir/{train,test}/{parasitized,uninfected}, renaming images
    to prefix + {train|test}_{parasitized|uninfected}{index}.{ext}
    """
    valid_exts = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
    sets = ['train', 'test']
    classes = ['parasitized', 'uninfected']

    for subset in sets:
        for cls in classes:
            dir_path = os.path.join(root_dir, subset, cls)
            if not os.path.isdir(dir_path):
                print(f"Skipping missing folder: {dir_path}")
                continue

            # Gather and sort image files
            files = [
                f for f in os.listdir(dir_path)
                if os.path.isfile(os.path.join(dir_path, f))
                and os.path.splitext(f)[1].lower() in valid_exts
            ]
            files.sort()

            # Rename each file
            for idx, filename in enumerate(files, start=1):
                ext = os.path.splitext(filename)[1].lower()
                new_name = f"{prefix}{subset}_{cls}{idx}{ext}"
                src = os.path.join(dir_path, filename)
                dst = os.path.join(dir_path, new_name)
                if os.path.exists(dst):
                    print(f"  → Skipping {filename}: {new_name} already exists")
                    continue
                os.rename(src, dst)
                print(f"  Renamed: {subset}/{cls}/{filename} → {new_name}")

def main():
    parser = argparse.ArgumentParser(
        description="Rename all images in train/test parasitized/uninfected folders"
    )
    parser.add_argument(
        "root",
        help="Path to the dataset root (which contains 'train' and 'test')"
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional prefix for every filename (default: none)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        parser.error(f"Root folder not found: {args.root}")

    rename_images_in_dataset(args.root, args.prefix)

if __name__ == "__main__":
    main()
