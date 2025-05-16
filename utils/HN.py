import torch
import math
import json
import pickle
import random
import pandas as pd
import numpy as np

from pathlib import Path

from .inference_POLO import plot_img_predictions, load_img_gt, match_predictions_loc
from ultralytics.utils.ops import generate_radii_t
from ultralytics.utils.metrics import ConfusionMatrix, loc_dor_pw


def read_detections_HN(dets_file: str, cls_name2id: dict, imgs_dir: str, dor_thresh: float, radii: dict, class_ids: list, output_dir: str, 
                       ann_file: str = None, ann_format: str = None, box_dims: dict = None, vis_prob: float = -1.0, vis_density: int = math.inf) -> None:
    
    det_dir = f"{output_dir}/detections"
    Path(det_dir).mkdir(exist_ok=True)
    vis_dir = f"{output_dir}/vis"
    Path(vis_dir).mkdir(exist_ok=True)

    hn_dets = pd.read_csv(dets_file)
    img_names = hn_dets["images"].unique()

     # read file if annotations are provided
    if ann_file:
        with open(ann_file, "r") as f:
            ann_dict = json.load(f)

    # for accumulating total counts across images
    counts_sum = {cls_id: 0 for cls_id in class_ids}    

    for fn in img_names:
        img_dets = hn_dets[hn_dets["images"] == fn]
        coords = torch.Tensor(img_dets[["x", "y"]].values)
        coords = coords[~torch.any(coords.isnan(),dim=1)]
        coords = coords * 2

        conf = torch.Tensor(img_dets["scores"].values).reshape(-1, 1)
        conf = conf[~torch.any(conf.isnan(),dim=1)]

        cls_names = [sn.capitalize() for sn in img_dets["species"].values]
        cls_ids = [cls_name2id[name] for name in cls_names if isinstance(name, str)]
        cls = torch.Tensor(cls_ids).reshape(-1, 1)

        visualize = (random.randint(0, 1000) / 1000 <= vis_prob) or (coords.shape[0] >= vis_density)
        if visualize:
            plot_img_predictions(img_fn=f"{imgs_dir}/{fn}", coords=coords, cls=cls, output_dir=vis_dir)

        # combine coordinates, confidence and class into one tensor
        preds_img_final = torch.hstack((coords, conf, cls))
 
        
        # If annotations are available, collect evaluation metrics at the image level 
        if ann_file:
            boxes_in = "BX" in ann_format
            gt_coords, gt_cls = load_img_gt(annotations=ann_dict[Path(fn).stem], boxes_in=boxes_in, boxes_out=False, ann_format=ann_format, 
                                            device=coords.device, box_dims=box_dims)
            radii_gt_t = generate_radii_t(radii=radii, cls=gt_cls)


            # Make Confusion matrix
            cfm_img = ConfusionMatrix(nc=len(class_ids), task="locate", dor_thresh=dor_thresh)
            cfm_img.process_batch_loc(localizations=preds_img_final, gt_locs=gt_coords, gt_cls=gt_cls, radii=radii_gt_t)
            # write confusion matrix to file
            with open(f"{det_dir}/{Path(fn).stem}_cfm.npy", "wb") as f:
                np.save(f, cfm_img.matrix)


            # make stats dict
            npr = preds_img_final.size(dim=0)
            stat = dict(
                conf=torch.zeros(0, device=preds_img_final.device),
                pred_cls=torch.zeros(0, device=preds_img_final.device),
                tp=torch.zeros(npr, 10, dtype=torch.bool, device=preds_img_final.device),
            )
            nl = gt_cls.size(dim=0)
            stat["target_cls"] = gt_cls

            if npr != 0: 
                stat["conf"] = preds_img_final[:, 2]
                stat["pred_cls"] = preds_img_final[:, 3]
                # Evaluate
                if nl:
                    dor = loc_dor_pw(loc1=gt_coords, loc2=preds_img_final[:, :2], radii=radii_gt_t)
                    stat["tp"] = match_predictions_loc(pred_classes=preds_img_final[:, 3], true_classes=gt_cls, dor=dor)    
            else:
                if not nl: 
                    raise ValueError("No predictions and no labels case is skipped in the original code, but I need a matching file, so I'm adding zero-stats.\n" \
                                     "This is going to wrongly reduce the performance metrics. Normally this shouldn't happen as there shouldn't be empty images.\n" \
                                     "This error being raised, however, means that there are!!")

            #write to pickle file:
            with open(f"{det_dir}/{Path(fn).stem}_stats.pickle", "wb") as f:
                pickle.dump(stat, f)

        cls_idx, counts = torch.unique(preds_img_final[:, -1], return_counts=True)
        counts_dict = {}
        for j in range(cls_idx.shape[0]):
            counts_dict[int(cls_idx[j].item())] = int(counts[j].item())

        with open(f"{det_dir}/{Path(fn).stem}.json", "w") as f:
            json.dump(counts_dict, f, indent=1)
        
        # add to count sum
        for class_idx, n in counts_dict.items():
            counts_sum[class_idx] += n

    # save counts
    with open(f"{Path(det_dir).parent}/counts_total.json", "w") as f:
        json.dump(counts_sum, f, indent=1)
