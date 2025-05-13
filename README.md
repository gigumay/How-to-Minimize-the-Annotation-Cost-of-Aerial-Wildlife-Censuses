# How-to-Minimize-the-Annotation-Cost-of-Aerial-Wildlife-Censuses
GitHub repo for the paper "How to Minimize the Annotation  Cost of Aerial Wildlife Censuses". In our paper we compare three deep learning architectures (one localization model and two object detection models; cf. below picture) with respect to their ability to detect animals from aerial imagery. Our experiments are conducted on four diverse datasets containing various animal species, environments, degrees of animal density, camera angles, and taken at different altitudes. We look at the effect of label complexity (i.e., hand-crafted bounding boxes vs. auto-generaterd pseudo-boxes vs. point labels) on detection accuracy and counting performance. We find:
- Wildlife counting accuracy is largely maintained at reduced label complexity
- Counting accuracy is robust towards variations in the size of pseudo box labels 
- Current detection models perform better on oblique images than previous architectures



![image](./YOLO_POLO.drawio.pdf "Our detection models.")
![image](./HN.drawio.pdf "Our localization model.")

This repo is intended to help users reproduce our results, and to provide tools for running inference with 
- dependencies (setup and install env)
- repo structure
- only YOLO/POLO. Refer to Herdnet repo for 
- cannot make Ennedi dataset publicly available 



- Ask alex if he wants to make a notebook for HN inference, if so I'll then merge it with the evluation (and adjust the readme)