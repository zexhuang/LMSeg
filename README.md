# LMSeg: An end-to-end geometric message-passing network on barycentric dual graphs for large-scale landscape mesh segmentation

## Abstract

Semantic segmentation of large-scale 3D landscape meshes is critical for geospatial analysis in complex environments, yet existing approaches face persistent challenges of scalability, end-to-end trainability, and accurate segmentation of small and irregular objects. To address these issues, we introduce the BudjBim Wall (BBW) dataset, a large-scale annotated mesh dataset derived from high-resolution LiDAR scans of the UNESCO World Heritage-listed Budj Bim cultural landscape in Victoria, Australia. The BBW dataset captures historic dry-stone wall structures that are difficult to detect under vegetation occlusion, supporting research in underrepresented cultural heritage contexts. Building on this dataset, we propose LMSeg, a deep graph message-passing network for semantic segmentation of large-scale meshes. LMSeg employs a barycentric dual graph representation of mesh faces and introduces the Geometry Aggregation+ (GA+) module, a learnable softmax-based operator that adaptively combines neighborhood features and captures high-frequency geometric variations. A hierarchical–local dual pooling integrates hierarchical and local geometric aggregation to balance global context with fine-detail preservation. Experiments on two large-scale benchmarks—SUM (urban) and BBW (natural)—show that LMSeg achieves 75.1\% mIoU on SUM and 62.4\% mIoU on BBW with only 2.4M parameters, outperforming strong point- and graph-based baselines. In particular, LMSeg excels on small-object classes (e.g., vehicles, high vegetation) and successfully detects dry-stone walls in dense natural environments. Together, the BBW dataset and LMSeg provide a practical and extensible method for advancing 3D mesh segmentation in cultural heritage, environmental monitoring, and urban applications.

## Architecture

![alt text](figs/architecture.png)
Overall architecture of LMSeg. (a). Input mesh is converted into barycentric dual graph with mesh texture and face normal features. (b). LMSeg encoder consists of random node sub-sampling, HGA+, edge pooling and LGA+ modules for hierarchical and local feature learning. A residual MLP takes concatenated LGA+ and HGA+ features as inputs and updates graph node features. (c). LMSeg decoder consists of feature propagation layers, which progressively up-sample the size of deep encoder features back to the original input size. $\texttt{N}$ denotes the number of input nodes of barycentric dual graph, and $\texttt{C}$ refers to the input node feature dimensions.

## **Installation**

Conda environment to run the code has exported to **requirements.yaml**.

## **Dataset**

```text
data
├── BBW
│   ├── mesh
│   │   ├── area1
│   │   ├── area2
│   │   ├── area3
│   │   ├── area4
│   │   ├── area5
│   │   └── area6
│   ├── pcd
│   │   ├── area1
│   │   ├── area2
│   │   ├── area3
│   │   ├── area4
│   │   ├── area5
│   │   └── area6
│   └── processed
│       ├── area1
│       ├── area2
│       ├── area3
│       ├── area4
│       ├── area5
│       └── area6
└── SUM
    ├── processed
    │   ├── test
    │   ├── train
    │   └── validate
    └── raw
        ├── test
        ├── train
        └── validate
```

## **Budj Bim Wall Dataset**

Budj Bim Wall (BBW) dataset is a lidar-scanned point-cloud dataset of the UNESCO World Heritage cultural landscape covered by the Budj Bim National Park in southwest Victoria, Australia. This is one of the areas with the highest density of European historic dry-stone walls in Australia. The dataset was collected in 2020 by the Department of Environment, Land, Water and Planning in Victoria, Australia for the Gunditj Mirring Traditional Owners Corporation.

![alt text](figs/budjbim_bev.jpeg)
<p align="center"> The entire Budj Bim landscape (301 km^2 area) collected by aerial lidar point clouds, containing 33 billion points.</p>

The BBW dataset is a subset of the full dataset, capturing the northern part of the data. It is spatially divided into six equal, rectangular areas. Each tile in BBW dataset is a textured landscape mesh of 400$m^2$ map area (with face density of ~45 faces/m$^2$) semi-manually annotated into binary semantic labels (wall vs. other terrain). We adopt a six-fold *leave-one-area-out* cross-validation, testing on one held-out geographic area per fold while training and validating (80/20 split) on the remaining five. This ensures coverage of diverse terrain types and provides a statistically robust estimate of model performance in unseen spatial regions.

![alt text](figs/bbw_map.png)
<p align="center"> Spatial partitioning of the *BudjBimArea* dataset. Orange lines indicate annotated European historic dry-stone walls near Tae Rak (Lake Condah), Victoria, Australia. The number of data samples per area is as follows: Area 1 - 107, Area 2 - 647, Area 3 - 625, Area 4 - 716, Area 5 - 893, and Area 6 - 1008.</p>

The textured mesh tiles of BBW dataset are constructed from raw lidar tiles using a reproducible pipeline. Point clouds are clipped to area boundaries, subdivided into processing grids, and filtered with a cloth simulation filter (CSF; resolution 0.05m, rigidness 1, slope smoothing enabled) to isolate ground points. Ground surfaces are then triangulated in 2D, elevations restored, and meshes simplified to reduce redundancy. Subsequent cleaning removes non-manifold edges, degenerate faces, and unreferenced vertices; merges vertices within tolerance; fills small holes; and reorients normals. These steps explicitly address mesh noise and missing-face artefacts, ensuring geometric integrity for downstream segmentation. Finally, RGB texture and binary wall/terrain masks are mapped to mesh faces by sampling centroids from orthophotos and raster masks, producing enriched meshes with per-face color and labels.

For GT labels, given the large area and high point density of the Budj Bim landscape, the [Semi-Automatic Classification Plugin](https://plugins.qgis.org/plugins/SemiAutomaticClassificationPlugin/) from QGIS, developed for the annotations of remote sensing images, are adopted for data anootation. This tool enables the creation of binary masks (0: Terrain, 1: Stone wal) on the xy-axis of stone wall stcutres. Additionally, for height-based stone wall annotation on the z-axis, a 2-meter height-above-ground constraint was applied to exclude points exceeding this height threshold.

![alt text](figs/budjbim_annotated_points.jpeg)
<p align="center"> The annotated stone walls (in lidar point clouds) of BBW dataset.</p>

## **Model Training**

```bash
python3 train/train_lmseg_sum.py --cfg=cfg/sum/lmseg_feature.yaml
```

or

```bash
python3 train/train_lmseg.py --cfg=cfg/bbw/lmseg_feature.yaml
```

## **Evaluation**

Pre-trained models are located at:

```bash
save/sum/lmseg_feature/ckpt/epoch{}.pth
```

or

```bash
save/bbw/bbw_lmseg_feature/ckpt/epoch{}.pth
```

## Results

![alt text](figs/sum_preds.png)
<p align="center"> Qualitative performance of LMSeg on SUM dataset.</p>

![alt text](figs/bbw_preds.png)
<p align="center"> Qualitative performance of LMSeg on Budj Bim Wall dataset.</p>

## License

The data in this repository is licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

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
