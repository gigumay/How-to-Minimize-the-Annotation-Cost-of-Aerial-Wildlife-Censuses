import math 
from utils.inference_YOLO import *

IMGS_DIR = "" #TODO: Insert Path to image directory.
OUTPUT_DIR = "" #TODO: Insert path to output directory 

PATCH_DIMS = {"width": -1, "height": -1}  #TODO: Insert patch width and height; (640, 640) in the paper
OVRLP = -1.0    #TODO: define patch overlap as a fraction of the patch dimensions (e.g. 128 pixel overlap = 0.2 of 640x640 patches)
DOR_THRESH = 0.3

VIS_PROB = 0.0
VIS_DENSITY = math.inf

MODEL_PATH = "/home/giacomo/projects/ennedi_herdcount_POLO/models/"

MODEL_NAME = "polov8n_300e_300pat_03DoR_dataID_3_NO_PT"

RADII = {0: 80, 1: 50, 2: 25}
BOX_DIMS = {"width": 50, "height": 50}

if __name__ == "__main__":

    ann_file = f"{IMGS_DIR}/test_annotations.json"
    tiling_dir = f"{IMGS_DIR}/tiles"
    
    random.seed(0)

    thresh_str = "".join(str(DOR_THRESH).split("."))
    task = "locate"

    results_dir = f"{OUTPUT_DIR}/{MODEL_NAME}_{thresh_str}thresh_{MODEL_CHKPT}"
    Path(results_dir).mkdir(parents=False, exist_ok=True)

    with open(f"{results_dir}/radii.json", "w") as f:
        json.dump(RADII, f)

    run_tiled_inference_POLO(model=f"{MODEL_PATH}/{MODEL_NAME}/weights/{MODEL_CHKPT}.pt", 
                             class_ids=list(RADII.keys()),
                             imgs_dir=IMGS_DIR, 
                             img_files_ext="JPG",
                             patch_dims=PATCH_DIMS, 
                             patch_overlap=OVRLP, 
                             output_dir=results_dir,
                             dor_thresh=DOR_THRESH,
                             radii=RADII,
                             ann_file=ANN_FILE,
                             ann_format="PT_DEFAULT", 
                             vis_prob=VIS_PROB,
                             vis_density=VIS_DENSITY)
        

    