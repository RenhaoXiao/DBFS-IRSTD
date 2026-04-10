import os
import sys
import json
import argparse

import torch
import torchvision.transforms as T
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig

# python tools/inference/export_coco_pred_plain.py -c configs/deimv2_IRSTD/deimv2_hgnetv2_n_IRSTD.yml -r outputs/deimv2_n_NUDT-SIRST/best_stg2.pth -o results/NUDT-SIRST/deimv2_n.json -d cuda --score_thresh 0.0 --label_offset 0

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def load_checkpoint_state(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'ema' in checkpoint and checkpoint['ema'] is not None:
        print('[INFO] Using EMA weights.')
        return checkpoint['ema']['module']
    print('[INFO] Using model weights.')
    return checkpoint['model']


def get_eval_image_size(cfg):
    # try to read eval_spatial_size from yaml config
    if 'eval_spatial_size' in cfg.yaml_cfg:
        size = cfg.yaml_cfg['eval_spatial_size']
        if isinstance(size, (list, tuple)) and len(size) == 2:
            return tuple(size)
    # fallback
    return (640, 640)


def build_transform(cfg):
    size = get_eval_image_size(cfg)
    vit_backbone = cfg.yaml_cfg.get('DINOv3STAs', False)

    if vit_backbone:
        transform = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = T.Compose([
            T.Resize(size),
            T.ToTensor(),
        ])
    return transform, size


def find_val_dataset_info(cfg):
    """
    Read image folder and annotation file from cfg.val_dataloader.dataset
    """
    dataset = cfg.val_dataloader.dataset
    img_folder = dataset.img_folder
    ann_file = dataset.ann_file
    return img_folder, ann_file


@torch.no_grad()
def export_predictions_plain(cfg, checkpoint_path, output_json, device='cuda',
                             score_thresh=0.0, label_offset=0):
    # disable pretrained loading
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    state = load_checkpoint_state(checkpoint_path)

    cfg.model.load_state_dict(state)
    model = cfg.model.to(device)
    model.eval()

    postprocessor = cfg.postprocessor.to(device)
    postprocessor.eval()

    transform, resize_size = build_transform(cfg)
    img_folder, ann_file = find_val_dataset_info(cfg)

    print(f"[INFO] Image folder : {img_folder}")
    print(f"[INFO] Annotation   : {ann_file}")
    print(f"[INFO] Resize size  : {resize_size}")

    with open(ann_file, 'r', encoding='utf-8') as f:
        coco_gt = json.load(f)

    images = coco_gt["images"]
    coco_results = []

    for idx, img_info in enumerate(images):
        image_id = img_info["id"]
        file_name = img_info["file_name"]
        img_path = os.path.join(img_folder, file_name)

        if not os.path.exists(img_path):
            print(f"[WARNING] Image not found: {img_path}")
            continue

        img_pil = Image.open(img_path).convert("RGB")
        w, h = img_pil.size

        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        orig_target_sizes = torch.tensor([[w, h]], dtype=torch.float32, device=device)

        outputs = model(img_tensor)
        results = postprocessor(outputs, orig_target_sizes)
        result = results[0]

        boxes = result["boxes"].detach().cpu()
        labels = result["labels"].detach().cpu()
        scores = result["scores"].detach().cpu()

        keep = scores >= score_thresh
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]

        for box, label, score in zip(boxes, labels, scores):
            coco_results.append({
                "image_id": int(image_id),
                "category_id": int(label.item()) + label_offset,
                "bbox": xyxy_to_xywh(box.tolist()),
                "score": float(score.item())
            })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(images):
            print(f"[INFO] Processed {idx + 1}/{len(images)} images.")

    output_dir = os.path.dirname(output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(coco_results, f)

    print(f"[INFO] Export finished. Saved to: {output_json}")
    print(f"[INFO] Total predictions: {len(coco_results)}")


def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume)

    export_predictions_plain(
        cfg=cfg,
        checkpoint_path=args.resume,
        output_json=args.output,
        device=args.device,
        score_thresh=args.score_thresh,
        label_offset=args.label_offset,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config yaml')
    parser.add_argument('-r', '--resume', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output COCO result json')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--score_thresh', type=float, default=0.0, help='Filter predictions by score before export')
    parser.add_argument('--label_offset', type=int, default=0, help='Offset added to predicted labels')
    args = parser.parse_args()
    main(args)
