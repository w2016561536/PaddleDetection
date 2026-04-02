from pathlib import Path
import shutil

# 你的 txt 文件路径
txt_file = Path("./val.txt")   # 改成你的 TXT 文件名

images_test_dir = Path("./images/val")
labels_test_dir = Path("./labels/val")

images_test_dir.mkdir(parents=True, exist_ok=True)
labels_test_dir.mkdir(parents=True, exist_ok=True)

with txt_file.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 1:
            continue

        img_path = Path(parts[0])   # 第一列是图片路径
        stem = img_path.stem        # 例如 704zhengchang

        yolo_label = Path("./YOLO") / f"{stem}.txt"

        # 复制图片
        if img_path.exists():
            shutil.copy2(img_path, images_test_dir / img_path.name)
            print(f"复制图片: {img_path} -> {images_test_dir / img_path.name}")
        else:
            print(f"图片不存在: {img_path}")

        # 复制 YOLO 标注
        if yolo_label.exists():
            shutil.copy2(yolo_label, labels_test_dir / yolo_label.name)
            print(f"复制标注: {yolo_label} -> {labels_test_dir / yolo_label.name}")
        else:
            print(f"YOLO标注不存在: {yolo_label}")
