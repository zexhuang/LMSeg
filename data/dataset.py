import os
from typing import Optional, Union
from pathlib import Path

import numpy as np
import trimesh
import pymeshlab as pml 

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import (coalesce, remove_self_loops,
                                   to_undirected, is_undirected)


def rgb_to_hsv(rgb: np.ndarray):
    """ convert RGB to HSV color space

    :param rgb: np.ndarray
    :return: np.ndarray
    """
        
    assert rgb.shape[-1] == 3
    
    rgb = rgb.astype(np.float32)
    maxv = np.amax(rgb, axis=-1)
    maxc = np.argmax(rgb, axis=-1)
    minv = np.amin(rgb, axis=-1)
    minc = np.argmin(rgb, axis=-1)

    hsv = np.zeros(rgb.shape, dtype=np.float32)
    hsv[maxc == minc, 0] = np.zeros(hsv[maxc == minc, 0].shape)
    hsv[maxc == 0, 0] = (((rgb[..., 1] - rgb[..., 2]) * 60.0 / (maxv - minv + np.spacing(1))) % 360.0)[maxc == 0]
    hsv[maxc == 1, 0] = (((rgb[..., 2] - rgb[..., 0]) * 60.0 / (maxv - minv + np.spacing(1))) + 120.0)[maxc == 1]
    hsv[maxc == 2, 0] = (((rgb[..., 0] - rgb[..., 1]) * 60.0 / (maxv - minv + np.spacing(1))) + 240.0)[maxc == 2]
    hsv[maxv == 0, 1] = np.zeros(hsv[maxv == 0, 1].shape)
    hsv[maxv != 0, 1] = (1 - minv / (maxv + np.spacing(1)))[maxv != 0]
    hsv[..., 2] = maxv

    return hsv
    
    
class BudjBimWallMeshDataset(Dataset):
    def __init__(self, 
                 root: Union[Path, str],
                 split: str = 'train', 
                 load_feature: Optional[str] = 'all',
                 transform = None, 
                 pre_transform = None):
        self.root = Path(root) / 'mesh'       
        
        self.areas = {'train': ['area1', 'area3', 'area5', 'area6'],
                      'val': ['area4'],
                      'test': ['area2']}
        
        super().__init__(self.root, transform, pre_transform)
        
        assert split in ['train', 'val', 'test']
        self.split = split        
        
        self.load_feature = load_feature
         
        self.mesh_list = []
        for area in self.areas[split]:
            self.mesh_list += sorted(list((self.root / area).glob('*.ply')))
        self.mesh_list = sorted(self.mesh_list)
            
        self.data_list = []
        for area in self.areas[split]:
            self.data_list += sorted(list((Path(self.processed_dir) / area).glob('*.pt')))
        
        if transform:
            self.transform=transform
        elif split == 'train': 
            self.transform = self.get_transform['train']
        elif split == 'test' or 'val': 
            self.transform = self.get_transform['test']
        
    @property
    def get_transform(self) -> dict:
        return {
                'train': T.Compose([T.NormalizeScale(),
                                    T.RandomRotate(1, axis=0),
                                    T.RandomRotate(1, axis=1),
                                    T.RandomRotate(180, axis=2),
                                    T.RandomJitter(0.001)]),
                'test': T.Compose([T.NormalizeScale()])
                }
        
    @property
    def processed_file_names(self):
        return ['area1', 'area2', 'area3', 'area4', 'area5', 'area6']
            
    def process(self): 
        for area_id, area in enumerate(self.processed_file_names):
            Path(self.processed_paths[area_id]).mkdir(parents=True, exist_ok=True)
            
            raw_ply_files = list(self.root / area / f for f in os.listdir(self.root / area))
                
            for f in raw_ply_files:
                plydata = trimesh.load(f, force="mesh")
                plydata.fix_normals()
                plydata.invert()
                
                face = torch.from_numpy(np.asarray(plydata.faces, dtype=np.int64)).t()
                face_rgba = torch.from_numpy(np.asarray(plydata.visual.face_colors, dtype=np.float32))
                pos = torch.from_numpy(np.asarray(plydata.triangles_center, dtype=np.float32)) 
                mask = torch.from_numpy(np.asarray(plydata.metadata['_ply_raw']['face']['data']['mask'], dtype=np.float32))
                if mask.ndim == 1: mask = mask.unsqueeze(1)
                
                edge_index = torch.from_numpy(np.asarray(plydata.face_adjacency.copy(), dtype=np.int64)).t()
                if not is_undirected(edge_index, num_nodes=pos.shape[0]):
                    edge_index = to_undirected(edge_index, num_nodes=pos.shape[0])
                edge_index = coalesce(edge_index, num_nodes=pos.shape[0])
                edge_index, _ = remove_self_loops(edge_index)
                
                data = Data(face=face, rgba=face_rgba / 255.0, pos=pos, edge_index=edge_index, y=mask) 
                # mesh normals 
                data.normals = torch.from_numpy(np.asarray(plydata.face_normals, dtype=np.float32))  
                # mesh (HSV) color  
                vertex_id = data.face.t().numpy().reshape(-1) 
                v_rgba = np.asarray(plydata.visual.vertex_colors, dtype=np.float32)[vertex_id].reshape(-1, 3, 4)
                f_rgba = np.asarray(plydata.visual.face_colors, dtype=np.float32)
                
                hsv_i, hsv_j, hsv_k, hsv_f = rgb_to_hsv(v_rgba[:, 0, 0:3]), \
                                             rgb_to_hsv(v_rgba[:, 1, 0:3]), \
                                             rgb_to_hsv(v_rgba[:, 2, 0:3]), \
                                             rgb_to_hsv(f_rgba[:, 0:3])
                                            
                hsv_max = np.array([360.0, 1.0, 255.0], dtype=np.float32)                                                 
                hsv = torch.from_numpy(np.concatenate([hsv_i / hsv_max, 
                                                       hsv_j / hsv_max, 
                                                       hsv_k / hsv_max, 
                                                       hsv_f / hsv_max], axis=1))    
                data.hsv = hsv
                    
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                    
                torch.save(data, os.path.join(self.processed_paths[area_id], f'{f.stem}.pt'))
        
    def len(self):
        return len(self.data_list) 
    
    def get(self, idx):        
        data = torch.load(self.data_list[idx])            
        
        if self.load_feature == 'all':
            data.x = torch.cat([data.normals, data.hsv], dim=-1)
        elif self.load_feature == 'hsv':
            data.x = data.hsv
        elif self.load_feature == 'normals':
            data.x = data.normals
        elif self.load_feature is None:
            data.x = None
        
        if self.transform:
            data = self.transform(data)
            
        return data


class SUMDataset(Dataset):
    def __init__(self, 
                 root: Union[Path, str], 
                 split: str = 'train', 
                 load_feature: Optional[str] = 'all',
                 num_faces: Optional[int] = 430000,
                 qualitythr: float = 0.3,
                 transform = None, 
                 pre_transform = None):
        self.root = Path(root)       
        self.root.mkdir(parents=True, exist_ok=True)
        
        self.SUM_URLs = {'train': "https://3d.bk.tudelft.nl/opendata/sum/1.0/SUM_Helsinki_C6_mesh/train/", 
                         'validate': "https://3d.bk.tudelft.nl/opendata/sum/1.0/SUM_Helsinki_C6_mesh/validate/",
                         'test': "https://3d.bk.tudelft.nl/opendata/sum/1.0/SUM_Helsinki_C6_mesh/test/"}
        
        self.num_faces = num_faces
        self.qualitythr = qualitythr
        
        super().__init__(root, transform, pre_transform)
        
        assert split in ['train', 'validate', 'test']
        self.split = split
        
        self.load_feature = load_feature
        
        self.raw_mesh_list = sorted(list((Path(self.raw_dir) / split).glob('*.ply')))
        self.processed_mesh_list = sorted(list((Path(self.processed_dir) / split).glob('*.ply')))
        self.data_list = sorted(list((Path(self.processed_dir) / split).glob('*.pt')))
        
        if transform:
            self.transform=transform
        elif split == 'train': 
            self.transform = self.get_transform['train']
        elif split == 'test' or 'validate': 
            self.transform = self.get_transform['test']
            
    @property
    def raw_file_names(self):
        return ['train', 'validate', 'test']
    
    @property
    def processed_file_names(self):
        return ['train', 'validate', 'test']
    
    @property
    def mask_dict(self):
        return {
            '[0.0, 0.0, 0.0, 255.0]': 0, # unclassified, black
            '[170.0, 85.0, 0.0, 255.0]': 1, # terrain, brown
            '[170.0, 84.0, 0.0, 255.0]': 1, # terrain, brown
            '[0.0, 255.0, 0.0, 255.0]': 2, # high vegetation, green
            '[255.0, 255.0, 0.0, 255.0]': 3, # building, yellow
            '[0.0, 255.0, 255.0, 255.0]': 4, # water, lightblue
            '[255.0, 0.0, 255.0, 255.0]': 5, # car, pink
            '[0.0, 0.0, 153.0, 255.0]': 6, # boat, blue
            }
        
    @property
    def rgba_dict(self):
        return {
            0: [0.0, 0.0, 0.0, 255.0], # unclassified, black
            1: [170.0, 85.0, 0.0, 255.0], # terrain, brown
            2: [0.0, 255.0, 0.0, 255.0], # high vegetation, green
            3: [255.0, 255.0, 0.0, 255.0], # building, yellow
            4: [0.0, 255.0, 255.0, 255.0], # water, lightblue
            5: [255.0, 0.0, 255.0, 255.0], # car, pink
            6: [0.0, 0.0, 153.0, 255.0] # boat, blue
            }
    
    def download(self):
        for folder, url in self.SUM_URLs.items(): 
            os.system(f"wget -P {os.path.join(self.raw_dir, folder)} -r -np -nd {url}")
                    
    def process(self):        
        for id, folder in enumerate(self.processed_file_names):
            Path(self.processed_paths[id]).mkdir(parents=True, exist_ok=True)
            
            raw_ply_files = sorted(list((Path(self.raw_dir) / folder).glob('*.ply')))
            raw_tex_files = sorted(list((Path(self.raw_dir) / folder).glob('*.jpg')))
        
            for f, tex in zip(raw_ply_files, raw_tex_files):
                ms = pml.MeshSet()
                ms.load_new_mesh(str(f))
                ms.set_current_mesh(0)
                ms.set_texture_per_mesh(textname=str(tex))
                ms.transfer_texture_to_color_per_vertex()
                if folder in ['train', 'val', 'test'] and self.num_faces:
                    ms.meshing_decimation_quadric_edge_collapse_with_texture(targetfacenum=self.num_faces,
                                                                             qualitythr=self.qualitythr,
                                                                             preservenormal=True)
                ms.current_mesh().compact()
                
                faces = ms.current_mesh().face_matrix()
                vertices = ms.current_mesh().vertex_matrix()
                normals = ms.current_mesh().face_normal_matrix()
        
                mask_color = ms.current_mesh().face_color_matrix() * 255.0
                mask = np.array([self.mask_dict[str(rgba)] for rgba in mask_color.tolist()], dtype=np.uint8)
              
                ms.compute_color_transfer_vertex_to_face()
                v_color = ms.current_mesh().vertex_color_matrix() * 255.0        
                f_color = ms.current_mesh().face_color_matrix() * 255.0
                
                face_attr = {'mask': mask,
                             'red': mask_color[:,0].astype(np.uint8),
                             'green': mask_color[:,1].astype(np.uint8),
                             'blue': mask_color[:,2].astype(np.uint8),
                             'alpha': mask_color[:,3].astype(np.uint8),
                             'face_red': f_color[:,0].astype(np.uint8),
                             'face_green': f_color[:,1].astype(np.uint8),
                             'face_blue': f_color[:,2].astype(np.uint8),
                             'face_alpha': f_color[:,3].astype(np.uint8)}
                
                plydata = trimesh.Trimesh(vertices=vertices, 
                                          faces=faces, 
                                          face_colors=f_color,
                                          vertex_colors=v_color,
                                          face_normals=normals, 
                                          face_attributes=face_attr,
                                          validate=True, 
                                          process=True,
                                          merge_norm=True,
                                          merge_tex=True)
                plydata.fix_normals()
                plydata.invert()
                plydata.export(os.path.join(self.processed_paths[id], f.name))
                
                face = torch.from_numpy(np.asarray(plydata.faces, dtype=np.int64)).t()
                face_rgba = torch.from_numpy(np.asarray(plydata.visual.face_colors, dtype=np.float32))
                pos = torch.from_numpy(np.asarray(plydata.triangles_center, dtype=np.float32)) 
                mask = torch.from_numpy(plydata.face_attributes['mask'].astype(np.int64))
            
                edge_index = torch.from_numpy(np.asarray(plydata.face_adjacency.copy(), dtype=np.int64)).t()
                if not is_undirected(edge_index, num_nodes=pos.shape[0]):
                    edge_index = to_undirected(edge_index, num_nodes=pos.shape[0])
                edge_index = coalesce(edge_index, num_nodes=pos.shape[0])
                edge_index, _ = remove_self_loops(edge_index)
                    
                data = Data(face=face, rgba=face_rgba / 255.0, pos=pos, edge_index=edge_index, y=mask)                                          
                # mesh normals
                data.normals = torch.from_numpy(np.asarray(plydata.face_normals, dtype=np.float32))  
                # mesh (HSV) color
                vertex_id = data.face.t().numpy().reshape(-1)
                v_rgba = np.asarray(plydata.visual.vertex_colors, dtype=np.float32)[vertex_id].reshape(-1, 3, 4)
                f_rgba = np.asarray(plydata.visual.face_colors, dtype=np.float32)

                hsv_i, hsv_j, hsv_k, hsv_f = rgb_to_hsv(v_rgba[:, 0, 0:3]), \
                                             rgb_to_hsv(v_rgba[:, 1, 0:3]), \
                                             rgb_to_hsv(v_rgba[:, 2, 0:3]), \
                                             rgb_to_hsv(f_rgba[:, 0:3])
                                                
                hsv_max = np.array([360.0, 1.0, 255.0], dtype=np.float32)                                                 
                hsv = torch.from_numpy(np.concatenate([hsv_i / hsv_max, 
                                                       hsv_j / hsv_max, 
                                                       hsv_k / hsv_max, 
                                                       hsv_f / hsv_max], axis=1))
                data.hsv = hsv        
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                    
                torch.save(data, os.path.join(self.processed_paths[id], f'{f.stem}.pt'))
        
    @property
    def get_transform(self) -> dict:
        return {
                'train': T.Compose([T.NormalizeScale(),
                                    T.RandomRotate(1, axis=0),
                                    T.RandomRotate(1, axis=1),
                                    T.RandomRotate(180, axis=2),
                                    T.RandomJitter(0.001)]),
                'test': T.Compose([T.NormalizeScale()])
                }
        
    def len(self):
        return len(self.data_list) 
    
    def get(self, idx):        
        data = torch.load(self.data_list[idx])            
        
        if self.load_feature == 'all':
            data.x = torch.cat([data.normals, data.hsv], dim=-1)
        elif self.load_feature == 'hsv':
            data.x = data.hsv
        elif self.load_feature == 'normals':
            data.x = data.normals
        elif self.load_feature is None:
            data.x = None
        
        if self.transform:
            data = self.transform(data)
            
        return data
    
    
class BBWPointDataset(BudjBimWallMeshDataset):
    def __init__(self, 
                 root: Union[Path, str],
                 split: str = 'train', 
                 load_feature: Optional[str] = 'all',
                 first_subsampling_dl = 0.02,
                 config = None,
                 classification = False,
                 class_choice = None,
                 transform = None, 
                 pre_transform = None):
        super().__init__(root, split, load_feature, transform, pre_transform)
        self.first_subsampling_dl = first_subsampling_dl
        self.config = config
        self.classification = classification
        self.cat2id = {}
        self.seg_classes = {}
        
        # if a subset of classes is specified.
        if class_choice is not None:
            self.cat2id = {k: v for k, v in self.cat2id.items() if k in class_choice}
        self.id2cat = {v: k for k, v in self.cat2id.items()}
        
        self.classes = dict(zip(sorted(self.cat2id), range(len(self.cat2id))))
        
    def __getitem__(self, idx):
        data = self.get(self.indices()[idx])
        return data
    
    def get(self, idx):        
        data = torch.load(self.data_list[idx])   
        
        if self.transform:
            data = self.transform(data)
            
        if self.load_feature == 'all':
            point_features = torch.cat([data.normals, data.hsv], dim=-1)
        elif self.load_feature == 'hsv':
            point_features = data.hsv
        elif self.load_feature == 'normals':
            point_features = data.normals
        elif self.load_feature is None:
            point_features = None
            
        point_set = data.pos.detach().cpu().numpy()
        point_features = point_features.detach().cpu().numpy() if point_features != None else point_features
        seg = data.y.detach().cpu().numpy()
            
        features = np.ones([point_set.shape[0], 1])    
        features = np.concatenate([features, point_set, point_features], axis=1)            
        return point_set, features, seg