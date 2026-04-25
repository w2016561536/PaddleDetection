import os
import random
from pathlib import Path

# ===== 可改参数 =====
root_dir = Path("./dataset/newdata/")          # 数据集根目录
images_dir = root_dir / "images"      # 图片目录
annotations_dir = root_dir / "annotations"  # XML目录

train_txt = root_dir / "train.txt"
val_txt = root_dir / "val.txt"

val_ratio = 0.2       # 验证集比例
random_seed = 2026    # 固定随机种子，保证每次结果可复现
# ===================

# 常见图片后缀
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

def find_image_by_stem(stem: str, images_dir: Path):
    """
    根据 xml 文件名（不带后缀）寻找对应图片
    优先按常见后缀匹配
    """
    for ext in IMG_EXTS:
        img_path = images_dir / f"{stem}{ext}"
        if img_path.exists():
            return img_path
    return None

def main():
    if not annotations_dir.exists():
        raise FileNotFoundError(f"annotations 目录不存在: {annotations_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"images 目录不存在: {images_dir}")

    xml_files = sorted(annotations_dir.glob("*.xml"))
    if not xml_files:
        raise RuntimeError(f"未在 {annotations_dir} 中找到 xml 文件")

    pairs = []
    missing_images = []

    for xml_path in xml_files:
        stem = xml_path.stem
        img_path = find_image_by_stem(stem, images_dir)

        if img_path is None:
            missing_images.append(xml_path.name)
            continue

        # 输出相对路径，格式类似：
        # ./images/704zhengchang.jpg ./annotations/704zhengchang.xml
        img_rel = f"./images/{img_path.name}"
        xml_rel = f"./annotations/{xml_path.name}"
        pairs.append((img_rel, xml_rel))

    if not pairs:
        raise RuntimeError("没有成功匹配到任何 图片-XML 对")

    # 乱序
    random.seed(random_seed)
    random.shuffle(pairs)

    # 划分 train / val
    val_count = int(len(pairs) * val_ratio)
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    # 写入 train.txt
    with open(train_txt, "w", encoding="utf-8") as f:
        for img_rel, xml_rel in train_pairs:
            f.write(f"{img_rel} {xml_rel}\n")

    # 写入 val.txt
    with open(val_txt, "w", encoding="utf-8") as f:
        for img_rel, xml_rel in val_pairs:
            f.write(f"{img_rel} {xml_rel}\n")

    print(f"总匹配样本数: {len(pairs)}")
    print(f"train 数量: {len(train_pairs)} -> {train_txt}")
    print(f"val   数量: {len(val_pairs)} -> {val_txt}")

    if missing_images:
        print("\n以下 XML 没有找到对应图片：")
        for name in missing_images:
            print(name)

if __name__ == "__main__":
    main()
