# LMSeg: A deep graph message-passing network for efficient and accurate semantic segmentation of large-scale 3D landscape meshes

## Abstract

Semantic segmentation of large-scale 3D landscape meshes is pivotal for various geospatial applications, including spatial analysis, automatic mapping and localization of target objects, and urban planning and development. This requires an efficient and accurate 3D perception system to understand and analyze real-world environments. However, traditional mesh segmentation methods face challenges in accurately segmenting small objects and maintaining computational efficiency due to the complexity and large size of 3D landscape mesh datasets. This paper presents an end-to-end deep graph message-passing network, LMSeg, designed to efficiently and accurately perform semantic segmentation on large-scale 3D landscape meshes. The proposed approach takes the barycentric dual graph of meshes as inputs and applies deep message-passing neural networks to hierarchically capture the geometric and spatial features from the barycentric graph structures and learn intricate semantic information from textured meshes. The hierarchical and local pooling of the barycentric graph, along with the effective geometry aggregation modules of LMSeg, enable fast inference and accurate segmentation of small-sized and irregular mesh objects in various complex landscapes. Extensive experiments on two benchmark datasets (natural and urban landscapes) demonstrate that LMSeg significantly outperforms existing learning-based segmentation methods in terms of object segmentation accuracy and computational efficiency. Furthermore, our method exhibits strong generalization capabilities across diverse landscapes and demonstrates robust resilience against varying mesh densities and landscape topologies.

## Architecture

![alt text](figs/architecture.png)
Overall architecture of LMSeg. (a). Input mesh is converted into barycentric dual graph with mesh texture and face normal features. (b). LMSeg encoder consists of random node sub-sampling, HGA+, edge pooling and LGA+ modules for hierarchical and local feature learning. A residual MLP takes concatenated LGA+ and HGA+ features as inputs and updates graph node features. (c). LMSeg decoder consists of feature propagation layers, which progressively up-sample the size of deep encoder features back to the original input size. $\texttt{N}$ denotes the number of input nodes of barycentric dual graph, and $\texttt{C}$ refers to the input node feature dimensions.

## **Installation**

Conda environment to run the code has exported to **requirements.yaml**.

## **Dataset**

```text
data
├── BudjBimWall
│   ├── mesh
│   │   ├── area1
│   │   ├── area2
│   │   ├── area3
│   │   ├── area4
│   │   ├── area5
│   │   ├── area6
│   │   └── processed
│   │       ├── area1
│   │       ├── area2
│   │       ├── area3
│   │       ├── area4
│   │       ├── area5
│   │       └── area6
│   └── pcd
│       ├── area1
│       ├── area2
│       ├── area3
│       ├── area4
│       ├── area5
│       └── area6
├── SUM
    ├── processed
    │   ├── test
    │   ├── train
    │   └── validate
    └── raw
        ├── test
        ├── train
        └── validate
```

## **Model Training**

```bash
python3 train/train_lmseg_sum.py --cfg=cfg/sum/sum_lmseg_feature.yaml
```

or

```bash
python3 train/train_lmseg.py --cfg=cfg/bbw/bbw_lmseg_feature.yaml
```

## **Evaluation**

Pre-trained models are located at:

```bash
save/sum/sum_lmseg_feature/ckpt/epoch{}
```

or

```bash
save/bbw/bbw_lmseg_feature/ckpt/epoch{}
```

## Results

![alt text](figs/sum_preds.png)
<p align="center"> Qualitative performance of LMSeg on SUM dataset.</p>

![alt text](figs/bbw_preds.png)
<p align="center"> Qualitative performance of LMSeg on Budj Bim Wall dataset.</p>

## Citation

```text
@misc{huang2024lmsegdeepgraphmessagepassing,
      title={LMSeg: A deep graph message-passing network for efficient and accurate semantic segmentation of large-scale 3D landscape meshes}, 
      author={Zexian Huang and Kourosh Khoshelham and Gunditj Mirring Traditional Owners Corporation and Martin Tomko},
      year={2024},
      eprint={2407.04326},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.04326}, 
}
```
