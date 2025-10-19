# evaluate_yolo.py
import argparse, json, csv
from pathlib import Path
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, required=True, help='Path to trained weights, e.g., runs/detect/train7/weights/best.pt')
    ap.add_argument('--data', type=str, required=True, help='Dataset YAML, e.g., coco_yolo_exact.yaml')
    ap.add_argument('--split', type=str, default='val', choices=['val','test'], help='Dataset split to evaluate')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--iou', type=float, default=0.65)
    ap.add_argument('--device', type=str, default='0')
    ap.add_argument('--batch', type=int, default=16)
    args = ap.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        batch=args.batch,
        plots=True,
        save_json=True,
        verbose=True
    )

    out_dir = Path(metrics.save_dir)
    names = metrics.names
    per_class_ap = metrics.box.maps  # list of AP@[.5:.95] per class
    overall = {
        'map50-95': float(metrics.box.map),
        'map50': float(metrics.box.map50),
        'map75': float(metrics.box.map75),
        'precision': float(metrics.box.p),
        'recall': float(metrics.box.r)
    }
    (out_dir / 'overall_metrics.json').write_text(json.dumps(overall, indent=2))

    with open(out_dir / 'per_class_ap.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class_id', 'class_name', 'AP50-95'])
        for cid, ap in enumerate(per_class_ap):
            w.writerow([cid, names.get(cid, str(cid)), 0.0 if ap is None else float(ap)])

    print(f'Done, results saved to: {out_dir}')

if __name__ == "__main__":
    main()
