import os
import json
import argparse
from collections import defaultdict


def load_coco_gt(gt_json_path):
    with open(gt_json_path, 'r', encoding='utf-8') as f:
        gt = json.load(f)

    images = gt['images']
    categories = gt.get('categories', [])

    file_name_to_image = {}
    image_id_to_info = {}

    for img in images:
        file_name = os.path.basename(img['file_name'])
        file_name_to_image[file_name] = img
        image_id_to_info[img['id']] = img

    return gt, file_name_to_image, image_id_to_info, categories


def yolo_xywhn_to_coco_xywh(xc, yc, w, h, img_w, img_h):
    bw = w * img_w
    bh = h * img_h
    x = xc * img_w - bw / 2.0
    y = yc * img_h - bh / 2.0
    return [float(x), float(y), float(bw), float(bh)]


def convert_yolo_txt_dir_to_coco_json(
    gt_json_path,
    pred_txt_dir,
    output_json_path,
    class_offset=0,
    score_thresh=0.0
):
    _, file_name_to_image, _, _ = load_coco_gt(gt_json_path)

    coco_results = []
    txt_files = [f for f in os.listdir(pred_txt_dir) if f.endswith('.txt')]
    txt_files.sort()

    missing_images = []
    total_boxes = 0

    for txt_name in txt_files:
        stem = os.path.splitext(txt_name)[0]

        matched_img = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            candidate = stem + ext
            if candidate in file_name_to_image:
                matched_img = file_name_to_image[candidate]
                break

        if matched_img is None:
            # 再尝试直接匹配完整 file_name 去掉目录后的 basename
            if txt_name in file_name_to_image:
                matched_img = file_name_to_image[txt_name]

        if matched_img is None:
            missing_images.append(txt_name)
            continue

        image_id = matched_img['id']
        img_w = matched_img['width']
        img_h = matched_img['height']

        txt_path = os.path.join(pred_txt_dir, txt_name)
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) < 6:
                continue

            cls_id = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            conf = float(parts[5])

            if conf < score_thresh:
                continue

            bbox = yolo_xywhn_to_coco_xywh(xc, yc, w, h, img_w, img_h)

            coco_results.append({
                "image_id": int(image_id),
                "category_id": int(cls_id + class_offset),
                "bbox": bbox,
                "score": float(conf)
            })
            total_boxes += 1

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True) if os.path.dirname(output_json_path) else None
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_results, f)

    print('=' * 60)
    print(f'GT json        : {gt_json_path}')
    print(f'Pred txt dir   : {pred_txt_dir}')
    print(f'Output json    : {output_json_path}')
    print(f'Total boxes    : {total_boxes}')
    print(f'Missing txt map: {len(missing_images)}')
    if missing_images[:10]:
        print('Examples of unmatched txt files:', missing_images[:10])
    print('=' * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, required=True, help='COCO GT annotation json')
    parser.add_argument('--pred_txt_dir', type=str, required=True, help='Directory containing YOLO prediction txt files')
    parser.add_argument('--output', type=str, required=True, help='Output COCO prediction json')
    parser.add_argument('--class_offset', type=int, default=0, help='category_id = cls_id + class_offset')
    parser.add_argument('--score_thresh', type=float, default=0.0, help='filter low-score predictions before conversion')
    args = parser.parse_args()

    convert_yolo_txt_dir_to_coco_json(
        gt_json_path=args.gt,
        pred_txt_dir=args.pred_txt_dir,
        output_json_path=args.output,
        class_offset=args.class_offset,
        score_thresh=args.score_thresh
    )


if __name__ == '__main__':
    main()
