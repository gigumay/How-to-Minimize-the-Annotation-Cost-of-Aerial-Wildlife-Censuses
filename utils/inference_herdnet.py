import argparse
import torch
import os
import pandas
import warnings
import PIL
import logging

import albumentations as A

from torch.utils.data import DataLoader

from tqdm import tqdm

from animaloc.data.transforms import DownSample, Rotate90
from animaloc.models import LossWrapper, HerdNet
from animaloc.eval import HerdNetStitcher, HerdNetLMDS
from animaloc.datasets import CSVDataset
from animaloc.utils.useful_funcs import mkdir, current_date

warnings.filterwarnings('ignore')
PIL.Image.MAX_IMAGE_PIXELS = None


parser = argparse.ArgumentParser(
    prog='inference', 
    description='Collects the detections of HerdNet model on a set of images '
    )

parser.add_argument('root', type=str,
    help='path to the JPG images folder (str)')
parser.add_argument('pth', type=str,
    help='path to PTH file containing your model parameters (str)')  
parser.add_argument('--size', type=int, default=512,
    help='patch size for stitching. Defaults to 512.')
parser.add_argument('--over', type=int, default=160,
    help='overlap for stitching. Defaults to 160.')
parser.add_argument('--ats', type=float, default=0.2,
    help='adaptive threshold for LMDS. Defaults to 0.2.')
parser.add_argument('--device', type=str, default='cuda',
    help='device on which model and images will be allocated (str). \
        Possible values are \'cpu\' or \'cuda\'. Defaults to \'cuda\'.')
parser.add_argument('--rot', type=int, default=0,
    help='number of times to rotate by 90 degrees. Defaults to 0.')

args = parser.parse_args()

def main():

    # Create destination folder
    curr_date = current_date()
    dest = os.path.join(args.root, f"{curr_date}_HerdNet_results")
    mkdir(dest)

    # Configure CSV logging
    csv_logger = logging.getLogger("detection_logger")
    csv_logger.setLevel("INFO")
    csv_file = os.path.join(dest, f"{curr_date}_HerdNet_detections.csv")
    csv_handler = logging.FileHandler(csv_file, mode="a", encoding="utf-8")
    csv_formatter = logging.Formatter("{message}", style="{")
    csv_handler.setFormatter(csv_formatter)
    csv_logger.addHandler(csv_handler)

    csv_logger.info("images,x,y,species,scores,dscores")

    # Read info from PTH file
    map_location = torch.device('cpu')
    if torch.cuda.is_available():
        map_location = torch.device('cuda')

    checkpoint = torch.load(args.pth, map_location=map_location)
    classes = checkpoint['classes']
    num_classes = len(classes) + 1
    img_mean = checkpoint['mean']
    img_std = checkpoint['std']
    
    # Prepare dataset and dataloader
    img_names = [i for i in os.listdir(args.root) 
            if i.endswith(('.JPG','.jpg','.JPEG','.jpeg'))]
    n = len(img_names)
    empty_df = pandas.DataFrame(data={'images': img_names, 'x': [0]*n, 'y': [0]*n, 'labels': [1]*n})
    
    end_transforms = []
    if args.rot != 0:
        end_transforms.append(Rotate90(k=args.rot))
    end_transforms.append(DownSample(down_ratio = 2, anno_type = 'point'))
    
    albu_transforms = [A.Normalize(mean=img_mean, std=img_std)]
    
    dataset = CSVDataset(
        csv_file = empty_df,
        root_dir = args.root,
        albu_transforms = albu_transforms,
        end_transforms = end_transforms
        )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
        sampler=torch.utils.data.SequentialSampler(dataset))
    
    # Build the trained model
    print('Building the model ...')
    device = torch.device(args.device)
    model = HerdNet(num_classes=num_classes, pretrained=False)
    model = LossWrapper(model, [])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Start inference
    print('Starting inference ...')
    lmds_kwargs = dict(kernel_size=(3,3), adapt_ts=args.ats, neg_ts=0.1)
    stitcher = HerdNetStitcher(
            model = model,
            size = (args.size,args.size),
            overlap = args.over,
            down_ratio = 2,
            up = True, 
            reduction = 'mean',
            device_name = device
            ) 
    
    model.eval()

    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(dataloader)):
            
            image_name = targets["image_name"][0][0]

            images = images.to(device)

            if stitcher is not None:
                output = stitcher(images[0])
                output = output[:,:1,:,:], output[:,1:,:,:]
            else: 
                output, _ = model(images)

            lmds = HerdNetLMDS(up=False, **lmds_kwargs)
            counts, locs, labels, scores, dscores = lmds(output)
            n = len(labels[0])
            preds = dict(
                images = [image_name] * n,
                x = [x[1] for x in locs[0]],
                y = [x[0] for x in locs[0]],
                species = [classes[x] for x in labels[0]],
                scores = scores[0],
                dscores = dscores[0]
                )

            for i in range(n):
                one_row = []
                for k in preds.keys():
                    one_row.append(str(preds[k][i]))
                csv_logger.info(",".join(one_row))

    print('Inference finished!')

if __name__ == '__main__':
    main()