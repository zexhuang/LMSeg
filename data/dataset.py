import os
import numpy as np

import trimesh
import pymeshlab as pml 

import torch
import torch_geometric.transforms as T

from typing import Union
from pathlib import Path
from tqdm import tqdm

from torch import nn
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import (coalesce, remove_self_loops,
                                   to_undirected, is_undirected)
from sklearn.model_selection import train_test_split
from scipy.spatial import cKDTree


class ColorJitter(nn.Module):
    def __init__(
        self,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
        color_dropout=0.05,
        noise_std=0.01,
        dropout_type='gray',
    ):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.color_dropout = color_dropout
        self.noise_std = noise_std
        self.dropout_type = dropout_type
        assert dropout_type in ['gray', 'zero']

    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            Data object containing 'rgb' attribute of shape.
        Returns:
            Augmented RGB tensor of same shape
        """
        
        if not hasattr(data, 'rgb'):
            return data
        
        rgb = data.rgb.contiguous()
        rgb = rgb.view(-1, 4, 3).float()
        
        N, P, _ = rgb.shape
        device = rgb.device
        rgb = rgb.clamp(0, 1)

        # Brightness (scale per-sample)
        if self.brightness > 0:
            factors = torch.empty(N, 1, 1, device=device).uniform_(
                1 - self.brightness, 1 + self.brightness)
            rgb = rgb * factors

        # Contrast
        if self.contrast > 0:
            means = rgb.mean(dim=1, keepdim=True)
            factors = torch.empty(N, 1, 1, device=device).uniform_(
                1 - self.contrast, 1 + self.contrast)
            rgb = (rgb - means) * factors + means

        # Saturation
        if self.saturation > 0:
            weights = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 1, 3)
            gray = (rgb * weights).sum(dim=-1, keepdim=True)
            factors = torch.empty(N, 1, 1, device=device).uniform_(
                1 - self.saturation, 1 + self.saturation)
            rgb = (rgb - gray) * factors + gray

        # Hue adjustment (convert RGB -> HSV -> RGB)
        if self.hue > 0:
            hue_shifts = torch.empty(N, device=device).uniform_(-self.hue, self.hue)
            rgb = self.adjust_hue(rgb, hue_shifts)

        # Color dropout
        if self.color_dropout > 0:
            mask = torch.rand(N, device=device) < self.color_dropout
            if mask.any():
                if self.dropout_type == 'gray':
                    gray = rgb.mean(dim=2, keepdim=True)
                    rgb[mask] = gray[mask].expand(-1, P, 3)
                else:  # zero
                    rgb[mask] = 0.0

        # Add Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(rgb) * self.noise_std
            rgb = rgb + noise

        data.rgb = rgb.clamp(0, 1).view_as(data.rgb)
        return data

    def adjust_hue(self, rgb: torch.Tensor, hue_shift: torch.Tensor) -> torch.Tensor:
        """Adjust hue per sample for batch of (N, P, 3) RGB"""
        hsv = self.rgb_to_hsv(rgb)
        hsv[..., 0] = (hsv[..., 0] + hue_shift.view(-1, 1)) % 1.0
        return self.hsv_to_rgb(hsv)

    def rgb_to_hsv(self, rgb):
        r, g, b = rgb.unbind(-1)
        maxc = rgb.max(dim=-1).values
        minc = rgb.min(dim=-1).values
        v = maxc
        deltac = maxc - minc

        s = deltac / (maxc + 1e-8)
        s[maxc == 0] = 0

        h = torch.zeros_like(deltac)
        mask = deltac > 0
        rc = (maxc - r) / (deltac + 1e-8)
        gc = (maxc - g) / (deltac + 1e-8)
        bc = (maxc - b) / (deltac + 1e-8)

        h[(maxc == r) & mask] = (bc - gc)[(maxc == r) & mask]
        h[(maxc == g) & mask] = 2.0 + (rc - bc)[(maxc == g) & mask]
        h[(maxc == b) & mask] = 4.0 + (gc - rc)[(maxc == b) & mask]
        h = (h / 6.0).remainder(1.0)

        return torch.stack((h, s, v), dim=-1)

    def hsv_to_rgb(self, hsv):
        h, s, v = hsv.unbind(-1)
        h = h % 1.0
        i = torch.floor(h * 6).to(torch.int64)
        f = h * 6 - i
        i = i % 6

        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        out = torch.stack([
            torch.stack([v, t, p], dim=-1),
            torch.stack([q, v, p], dim=-1),
            torch.stack([p, v, t], dim=-1),
            torch.stack([p, q, v], dim=-1),
            torch.stack([t, p, v], dim=-1),
            torch.stack([v, p, q], dim=-1),
        ], dim=-2)
    
        i = i[..., None, None].expand(-1, -1, 1, 3)
        rgb = torch.gather(out, dim=2, index=i)  # shape: (n, 4, 1, 3)
        
        return rgb.squeeze(2)  # final shape: (n, 4, 3)
    

# def rgb_to_hsv(rgb: np.ndarray):
#     """ convert RGB to HSV color space

#     :param rgb: np.ndarray
#     :return: np.ndarray
#     """
        
#     assert rgb.shape[-1] == 3
    
#     rgb = rgb.astype(np.float32)
#     maxv = np.amax(rgb, axis=-1)
#     maxc = np.argmax(rgb, axis=-1)
#     minv = np.amin(rgb, axis=-1)
#     minc = np.argmin(rgb, axis=-1)

#     hsv = np.zeros(rgb.shape, dtype=np.float32)
#     hsv[maxc == minc, 0] = np.zeros(hsv[maxc == minc, 0].shape)
#     hsv[maxc == 0, 0] = (((rgb[..., 1] - rgb[..., 2]) * 60.0 / (maxv - minv + np.spacing(1))) % 360.0)[maxc == 0]
#     hsv[maxc == 1, 0] = (((rgb[..., 2] - rgb[..., 0]) * 60.0 / (maxv - minv + np.spacing(1))) + 120.0)[maxc == 1]
#     hsv[maxc == 2, 0] = (((rgb[..., 0] - rgb[..., 1]) * 60.0 / (maxv - minv + np.spacing(1))) + 240.0)[maxc == 2]
#     hsv[maxv == 0, 1] = np.zeros(hsv[maxv == 0, 1].shape)
#     hsv[maxv != 0, 1] = (1 - minv / (maxv + np.spacing(1)))[maxv != 0]
#     hsv[..., 2] = maxv

#     return hsv
    
    
class BudjBimLandscapeMeshDataset(Dataset):
    def __init__(self, 
                 root: Union[Path, str],
                 transform = None, 
                 pre_transform = None):
        self.root = Path(root)
        super().__init__(self.root, transform, pre_transform)
        self.mesh_list = sorted(list((self.root / 'mesh').glob('*.ply')))
        self.data_list = sorted(list((Path(self.processed_paths[0])).glob('*.pt')))
        
        if transform:
            self.transform=transform
        else:
            self.transform = T.Compose([T.NormalizeScale()])

    @property
    def processed_file_names(self):
        return ['processed_mesh']
            
    def process(self): 
        Path(self.processed_paths[0]).mkdir(parents=True, exist_ok=True)
        
        raw_ply_files = list(self.root / 'mesh' / f for f in os.listdir(self.root / 'mesh'))
            
        for f in tqdm(raw_ply_files, desc="Processing meshes"):
            plydata = trimesh.load(f, force="mesh")
            plydata.fix_normals()
            
            face = torch.from_numpy(np.asarray(plydata.faces, dtype=np.int64)).t()
            face_rgba = torch.from_numpy(np.asarray(plydata.visual.face_colors, dtype=np.float32))
            pos = torch.from_numpy(np.asarray(plydata.triangles_center, dtype=np.float32)) 
            
            edge_index = torch.from_numpy(np.asarray(plydata.face_adjacency.copy(), dtype=np.int64)).t()
            if not is_undirected(edge_index, num_nodes=pos.shape[0]):
                edge_index = to_undirected(edge_index, num_nodes=pos.shape[0])
            edge_index = coalesce(edge_index, num_nodes=pos.shape[0])
            edge_index, _ = remove_self_loops(edge_index)
            
            data = Data(face=face, pos=pos, face_rgba=(face_rgba / 255.0).clamp(0, 1), edge_index=edge_index) 
            vertex_id = data.face.t().numpy().reshape(-1) 
            # mesh normals 
            v_normals = np.asarray(plydata.vertex_normals, dtype=np.float32)[vertex_id].reshape(-1, 3, 3)
            f_normals = np.asarray(plydata.face_normals, dtype=np.float32)
            data.normals = torch.from_numpy(np.concatenate([v_normals[:, 0],
                                                            v_normals[:, 1],
                                                            v_normals[:, 2],
                                                            f_normals[:,]], axis=1)).clamp(-1, 1)
            # mesh (RGB) colors  
            v_rgba = np.asarray(plydata.visual.vertex_colors, dtype=np.float32)[vertex_id].reshape(-1, 3, 4)
            f_rgba = np.asarray(plydata.visual.face_colors, dtype=np.float32)
            data.rgb = torch.from_numpy(np.concatenate([(v_rgba[:, 0, 0:3] / 255.0),
                                                        (v_rgba[:, 1, 0:3] / 255.0),
                                                        (v_rgba[:, 2, 0:3] / 255.0),
                                                        (f_rgba[:, 0:3] / 255.0)], axis=1)).clamp(0, 1)
                
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)
                
            torch.save(data, os.path.join(self.processed_paths[0], f'{f.stem}.pt'))
        
    def len(self):
        return len(self.data_list) 
    
    def get(self, idx):        
        data = torch.load(self.data_list[idx], weights_only=False)           
        return data
    
    
class BudjBimWallMeshDataset(Dataset):
    def __init__(self, 
                 root: Union[Path, str],
                 split: str = 'train', 
                 test_area: str = 'area4',
                 train_size: float = 0.8,
                 transform = None, 
                 pre_transform = None):
        self.root = Path(root)       
        self.root.mkdir(parents=True, exist_ok=True)
        
        super().__init__(self.root, transform, pre_transform)  
        
        assert split in ['train', 'val', 'test'], f"Valid data split should be one of {['train', 'val', 'test']}"
        
        valid_areas = ['area1', 'area2', 'area3', 'area4', 'area5', 'area6']
        if test_area not in valid_areas:
            raise ValueError(
                f"Invalid test area: {test_area}. Valid options are {valid_areas}."
            )

        self.train_val_areas = [area for area in valid_areas if area != test_area]
        self.test_area = test_area
        
        if not (0 < train_size <= 1.0):
            raise ValueError("train_size must be in the range (0, 1].")
        self.train_size = train_size
            
        self.mesh_list = sorted(f for area in valid_areas for f in (self.root / 'mesh' / area).glob('*.ply'))
        self.data_list = sorted(f for area in valid_areas for f in (Path(self.processed_dir) / area).glob('*.pt'))
            
        self._prepare_data_split(split)
        
        if transform:
            self.transform=transform
        elif split == 'train': 
            self.transform = self.get_transform['train']
        elif split in ['test', 'val']: 
            self.transform = self.get_transform['test']
            
    def _prepare_data_split(self, split):
        """
        Prepare train, validation, or test data splits.
        """
        random_state = 42
        
        train_val_data_files = []
        for area in self.train_val_areas:
            train_val_data_files += sorted(list((Path(self.processed_dir) / area).glob('*.pt')))
            
        test_files = []
        test_files = sorted(list((Path(self.processed_dir) / self.test_area).glob('*.pt')))
            
        if split == 'train':
            train_files, _ = train_test_split(train_val_data_files, train_size=self.train_size, random_state=random_state)
            setattr(self, "data_files", train_files)
        elif split == 'val':
            _, val_files = train_test_split(train_val_data_files, train_size=self.train_size, random_state=random_state)
            setattr(self, "data_files", val_files)
        else:
            setattr(self, "data_files", test_files)  
        
    @property
    def get_transform(self) -> dict:
        return {
                'train': T.Compose([T.RandomRotate(1, axis=0),
                                    T.RandomRotate(1, axis=1),
                                    T.RandomRotate(180, axis=2),
                                    T.RandomJitter(0.005),
                                    T.NormalizeScale(),
                                    ColorJitter()]),
                'test': T.Compose([T.NormalizeScale()])
                }
        
    @property
    def processed_file_names(self):
        return ['area1', 'area2', 'area3', 'area4', 'area5', 'area6']
            
    def process(self): 
        for area_id, area in enumerate(self.processed_file_names):
            Path(self.processed_paths[area_id]).mkdir(parents=True, exist_ok=True)
            
            raw_ply_files = list(self.root / 'mesh' / area / f for f in os.listdir(self.root / 'mesh' / area))
                
            for f in tqdm(raw_ply_files, desc="Processing meshes"):
                plydata = trimesh.load(f, force="mesh")
                plydata.fix_normals()
                
                face = torch.from_numpy(np.asarray(plydata.faces, dtype=np.int64)).t()
                face_rgba = torch.from_numpy(np.asarray(plydata.visual.face_colors, dtype=np.float32))
                pos = torch.from_numpy(np.asarray(plydata.triangles_center, dtype=np.float32)) 
                mask = torch.from_numpy(np.asarray(plydata.metadata['_ply_raw']['face']['data']['mask'], dtype=np.float32))
                if mask.ndim == 1: 
                    mask = mask.unsqueeze(1)
                
                edge_index = torch.from_numpy(np.asarray(plydata.face_adjacency.copy(), dtype=np.int64)).t()
                if not is_undirected(edge_index, num_nodes=pos.shape[0]):
                    edge_index = to_undirected(edge_index, num_nodes=pos.shape[0])
                edge_index = coalesce(edge_index, num_nodes=pos.shape[0])
                edge_index, _ = remove_self_loops(edge_index)
                
                data = Data(face=face, pos=pos, face_rgba=(face_rgba / 255.0).clamp(0, 1), edge_index=edge_index, y=mask) 
                vertex_id = data.face.t().numpy().reshape(-1) 
                # mesh normals 
                v_normals = np.asarray(plydata.vertex_normals, dtype=np.float32)[vertex_id].reshape(-1, 3, 3)
                f_normals = np.asarray(plydata.face_normals, dtype=np.float32)
                data.normals = torch.from_numpy(np.concatenate([v_normals[:, 0],
                                                                v_normals[:, 1],
                                                                v_normals[:, 2],
                                                                f_normals[:,]], axis=1)).clamp(-1, 1)
                # mesh (RGB) colors  
                v_rgba = np.asarray(plydata.visual.vertex_colors, dtype=np.float32)[vertex_id].reshape(-1, 3, 4)
                f_rgba = np.asarray(plydata.visual.face_colors, dtype=np.float32)
                data.rgb = torch.from_numpy(np.concatenate([(v_rgba[:, 0, 0:3] / 255.0),
                                                            (v_rgba[:, 1, 0:3] / 255.0),
                                                            (v_rgba[:, 2, 0:3] / 255.0),
                                                            (f_rgba[:, 0:3] / 255.0)], axis=1)).clamp(0, 1)
                    
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                    
                torch.save(data, os.path.join(self.processed_paths[area_id], f'{f.stem}.pt'))
        
    def len(self):
        return len(self.data_files) 
    
    def get(self, idx):        
        data = torch.load(self.data_files[idx], weights_only=False)
        return data


class SUMDataset(Dataset):
    def __init__(self, 
                 root: Union[Path, str], 
                 split: str = 'train', 
                 transform = None, 
                 pre_transform = None):
        self.root = Path(root)       
        self.root.mkdir(parents=True, exist_ok=True)
        
        self.SUM_URLs = {'train': "https://3d.bk.tudelft.nl/opendata/sum/1.0/SUM_Helsinki_C6_mesh/train/", 
                         'validate': "https://3d.bk.tudelft.nl/opendata/sum/1.0/SUM_Helsinki_C6_mesh/validate/",
                         'test': "https://3d.bk.tudelft.nl/opendata/sum/1.0/SUM_Helsinki_C6_mesh/test/"}
        
        super().__init__(root, transform, pre_transform)
        
        assert split in ['train', 'validate', 'test']
        self.split = split
        
        self.raw_mesh_list = sorted(list((Path(self.raw_dir) / split).glob('*.ply')))
        self.processed_mesh_list = sorted(list((Path(self.processed_dir) / split).glob('*.ply')))
        self.data_list = sorted(list((Path(self.processed_dir) / split).glob('*.pt')))
        
        if transform:
            self.transform=transform
        elif split == 'train': 
            self.transform = self.get_transform['train']
        elif split in ('test', 'validate'):
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
            '[0.0, 0.0, 0.0, 255.0]': 0,     # unclassified, black
            '[170.0, 85.0, 0.0, 255.0]': 1,  # terrain, brown
            '[170.0, 84.0, 0.0, 255.0]': 1,  # terrain, brown
            '[0.0, 255.0, 0.0, 255.0]': 2,   # high vegetation, green
            '[255.0, 255.0, 0.0, 255.0]': 3, # building, yellow
            '[0.0, 255.0, 255.0, 255.0]': 4, # water, lightblue
            '[255.0, 0.0, 255.0, 255.0]': 5, # car, pink
            '[0.0, 0.0, 153.0, 255.0]': 6,   # boat, blue
            }
        
    @property
    def rgba_dict(self):
        return {
            0: [0.0, 0.0, 0.0, 255.0],     # unclassified, black
            1: [170.0, 85.0, 0.0, 255.0],  # terrain, brown
            2: [0.0, 255.0, 0.0, 255.0],   # high vegetation, green
            3: [255.0, 255.0, 0.0, 255.0], # building, yellow
            4: [0.0, 255.0, 255.0, 255.0], # water, lightblue
            5: [255.0, 0.0, 255.0, 255.0], # car, pink
            6: [0.0, 0.0, 153.0, 255.0]    # boat, blue
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
                ms.current_mesh().compact()
                
                faces = ms.current_mesh().face_matrix()
                vertices = ms.current_mesh().vertex_matrix()
                normals = ms.current_mesh().face_normal_matrix()
        
                mask_color = ms.current_mesh().face_color_matrix() 
                mask = np.array([self.mask_dict[str(rgba)] for rgba in (mask_color * 255.0).tolist()], dtype=np.uint8)
              
                ms.compute_color_transfer_vertex_to_face()
                v_color = ms.current_mesh().vertex_color_matrix()
                v_color = (v_color * 255.0).astype(np.uint8)
                    
                f_color = ms.current_mesh().face_color_matrix()
                f_color = (f_color * 255.0).astype(np.uint8)
                
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
                    
                data = Data(face=face, pos=pos, face_rgba=(face_rgba / 255.0).clamp(0, 1), edge_index=edge_index, y=mask) 
                vertex_id = data.face.t().numpy().reshape(-1) 
                # mesh normals 
                v_normals = np.asarray(plydata.vertex_normals, dtype=np.float32)[vertex_id].reshape(-1, 3, 3)
                f_normals = np.asarray(plydata.face_normals, dtype=np.float32)
                data.normals = torch.from_numpy(np.concatenate([v_normals[:, 0],
                                                                v_normals[:, 1],
                                                                v_normals[:, 2],
                                                                f_normals[:,]], axis=1)).clamp(-1, 1)
                # mesh (RGB) colors  
                v_rgba = np.asarray(plydata.visual.vertex_colors, dtype=np.float32)[vertex_id].reshape(-1, 3, 4)
                f_rgba = np.asarray(plydata.visual.face_colors, dtype=np.float32)
                data.rgb = torch.from_numpy(np.concatenate([(v_rgba[:, 0, 0:3] / 255.0),
                                                            (v_rgba[:, 1, 0:3] / 255.0),
                                                            (v_rgba[:, 2, 0:3] / 255.0),
                                                            (f_rgba[:, 0:3] / 255.0)], axis=1)).clamp(0, 1)      
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                    
                torch.save(data, os.path.join(self.processed_paths[id], f'{f.stem}.pt'))
        
    @property
    def get_transform(self) -> dict:
        return {
                'train': T.Compose([T.RandomRotate(1, axis=0),
                                    T.RandomRotate(1, axis=1),
                                    T.RandomRotate(180, axis=2),
                                    T.RandomJitter(0.005),
                                    T.NormalizeScale(),
                                    ColorJitter()]),
                'test': T.Compose([T.NormalizeScale()])
                }
        
    def len(self):
        return len(self.data_list) 
    
    def get(self, idx):        
        data = torch.load(self.data_list[idx], weights_only=False)           
        return data
    

class H3DDataset(Dataset):
    def __init__(self, 
                 root: Union[Path, str], 
                 epoch: str = "Epoch_March2019",
                 split: str = 'train', 
                 transform = None, 
                 pre_transform = None):
        assert epoch in ['Epoch_March2019', 'Epoch_November2018', 'Epoch_March2018']
        self.root = Path(root) / epoch
        self.root.mkdir(parents=True, exist_ok=True)

        super().__init__(self.root, transform, pre_transform)
        
        assert split in ['train', 'val', 'test']
        self.split = split
        
        self.processed_mesh_list = sorted(list((Path(self.processed_dir) / split).glob('*.ply')))
        self.data_list = sorted(list((Path(self.processed_dir) / split).glob('*.pt')))
        
        if transform:
            self.transform=transform
        elif split == 'train': 
            self.transform = self.get_transform['train']
        elif split in ('test', 'val'):
            self.transform = self.get_transform['test']
            
    @property
    def raw_file_names(self):
        return ['train', 'val', 'test']
    
    @property
    def processed_file_names(self):
        return ['train', 'val', 'test']
    
    @property
    def mask_dict(self):
        return {
            '[178.0, 203.0, 47.0, 255.0]': 0,  # Low Vegetation
            '[177.0, 202.0, 47.0, 255.0]': 0,
            '[183.0, 179.0, 170.0, 255.0]': 1, # Impervious Surface
            '[182.0, 177.0, 170.0, 255.0]': 1,
            '[32.0, 151.0, 163.0, 255.0]': 2,  # Vehicle
            '[31.0, 151.0, 163.0, 255.0]': 2,
            '[168.0, 33.0, 107.0, 255.0]': 3,  # Urban Furniture
            '[255.0, 122.0, 89.0, 255.0]': 4,  # Roof
            '[255.0, 121.0, 89.0, 255.0]': 4,
            '[255.0, 215.0, 136.0, 255.0]': 5, # Facade
            '[255.0, 214.0, 135.0, 255.0]': 5,
            '[89.0, 125.0, 53.0, 255.0]': 6,   # Shrub
            '[89.0, 125.0, 52.0, 255.0]': 6,   
            '[89.0, 124.0, 52.0, 255.0]': 6,  
            '[0.0, 128.0, 65.0, 255.0]': 7,    # Tree
            '[170.0, 85.0, 0.0, 255.0]': 8,    # Soil/Gravel
            '[170.0, 84.0, 0.0, 255.0]': 8,
            '[252.0, 225.0, 5.0, 255.0]': 9,   # Vertical Surface
            '[251.0, 225.0, 5.0, 255.0]': 9,
            '[128.0, 0.0, 0.0, 255.0]': 10,    # Chimney/Antenna
            '[0.0, 0.0, 0.0, 255.0]': 11,      # Unlabeled
        }
        
    @property
    def rgba_dict(self):
        return {
            0:  [178.0, 203.0, 47.0, 255.0],   # Low Vegetation
            1:  [183.0, 179.0, 170.0, 255.0],  # Impervious Surface
            2:  [32.0, 151.0, 163.0, 255.0],   # Vehicle
            3:  [168.0, 33.0, 107.0, 255.0],   # Urban Furniture
            4:  [255.0, 122.0, 89.0, 255.0],   # Roof
            5:  [255.0, 215.0, 136.0, 255.0],  # Facade
            6:  [89.0, 125.0, 53.0, 255.0],    # Shrub
            7:  [0.0, 128.0, 65.0, 255.0],     # Tree
            8:  [170.0, 85.0, 0.0, 255.0],     # Soil/Gravel
            9:  [252.0, 225.0, 5.0, 255.0],    # Vertical Surface
            10: [128.0, 0.0, 0.0, 255.0],      # Chimney/Antenna
            11: [0.0, 0.0, 0.0, 255.0],        # Unlabeled
        }
                    
    def process(self):        
        for id, folder in enumerate(self.processed_file_names):
            Path(self.processed_paths[id]).mkdir(parents=True, exist_ok=True)
            
            base_path = Path(self.root) / "Mesh" / "per_tile" / folder
            labeled_objs = sorted(base_path.rglob("*_labeled.obj"))
            raw_objs = sorted([f for f in base_path.rglob("*.obj") if not f.name.endswith("_labeled.obj")])

            for robj, lobj in zip(raw_objs, labeled_objs):
                rms = pml.MeshSet()
                rms.load_new_mesh(str(robj))
                rms.set_current_mesh(0)
                rms.current_mesh().compact()
                rms.meshing_decimation_quadric_edge_collapse_with_texture(
                    targetperc=0.5, 
                    qualitythr=0.9,
                    extratcoordw=1.0,
                    preserveboundary=True,
                    optimalplacement=True,
                    preservenormal=True,
                    planarquadric=False
                )

                faces = rms.current_mesh().face_matrix()
                vertices = rms.current_mesh().vertex_matrix()
                normals = rms.current_mesh().face_normal_matrix()

                rms.transfer_texture_to_color_per_vertex()
                v_color = rms.current_mesh().vertex_color_matrix()
                v_color = (v_color * 255.0).astype(np.uint8)
                    
                rms.compute_color_transfer_vertex_to_face()
                f_color = rms.current_mesh().face_color_matrix()
                f_color = (f_color * 255.0).astype(np.uint8)

                lms = pml.MeshSet()
                lms.load_new_mesh(str(lobj))
                lms.set_current_mesh(0)
                lms.current_mesh().compact()
                lms_f = lms.current_mesh().face_matrix()
                lms_v = lms.current_mesh().vertex_matrix()
    
                lms_c = np.asarray(trimesh.Trimesh(lms_v, lms_f).triangles_center) 
                rms_c = np.asarray(trimesh.Trimesh(vertices, faces).triangles_center)
                tree = cKDTree(lms_c)
                _, idx = tree.query(rms_c, k=1)

                lms_color = lms.current_mesh().face_color_matrix() * 255.0
                mask_color = lms_color[idx]
                mask = np.array([self.mask_dict[str(rgba)] for rgba in (mask_color).tolist()], dtype=np.uint8)
                
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
                plydata.export(os.path.join(self.processed_paths[id], f'{robj.stem}.ply'))
                
                face = torch.from_numpy(np.asarray(plydata.faces, dtype=np.int64)).t()
                face_rgba = torch.from_numpy(np.asarray(plydata.visual.face_colors, dtype=np.float32))
                pos = torch.from_numpy(np.asarray(plydata.triangles_center, dtype=np.float32)) 
                mask = torch.from_numpy(plydata.face_attributes['mask'].astype(np.int64))
            
                edge_index = torch.from_numpy(np.asarray(plydata.face_adjacency.copy(), dtype=np.int64)).t()
                if not is_undirected(edge_index, num_nodes=pos.shape[0]):
                    edge_index = to_undirected(edge_index, num_nodes=pos.shape[0])
                edge_index = coalesce(edge_index, num_nodes=pos.shape[0])
                edge_index, _ = remove_self_loops(edge_index)
                    
                data = Data(face=face, pos=pos, face_rgba=(face_rgba / 255.0).clamp(0, 1), edge_index=edge_index, y=mask) 
                vertex_id = data.face.t().numpy().reshape(-1) 
                # mesh normals 
                v_normals = np.asarray(plydata.vertex_normals, dtype=np.float32)[vertex_id].reshape(-1, 3, 3)
                f_normals = np.asarray(plydata.face_normals, dtype=np.float32)
                data.normals = torch.from_numpy(np.concatenate([v_normals[:, 0],
                                                                v_normals[:, 1],
                                                                v_normals[:, 2],
                                                                f_normals[:,]], axis=1)).clamp(-1, 1)
                # mesh (RGB) colors  
                v_rgba = np.asarray(plydata.visual.vertex_colors, dtype=np.float32)[vertex_id].reshape(-1, 3, 4)
                f_rgba = np.asarray(plydata.visual.face_colors, dtype=np.float32)
                data.rgb = torch.from_numpy(np.concatenate([(v_rgba[:, 0, 0:3] / 255.0),
                                                            (v_rgba[:, 1, 0:3] / 255.0),
                                                            (v_rgba[:, 2, 0:3] / 255.0),
                                                            (f_rgba[:, 0:3] / 255.0)], axis=1)).clamp(0, 1)      
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                    
                torch.save(data, os.path.join(self.processed_paths[id], f'{robj.stem}.pt'))
        
    @property
    def get_transform(self) -> dict:
        return {
                'train': T.Compose([T.RandomRotate(180, axis=2),
                                    T.RandomJitter(0.001),
                                    T.NormalizeScale()]),
                'test': T.Compose([T.NormalizeScale()])
                }
        
    def len(self):
        return len(self.data_list) 
    
    def get(self, idx):        
        data = torch.load(self.data_list[idx], weights_only=False)           
        return data
    
    
class BBWPointDataset(BudjBimWallMeshDataset):
    def __init__(self, 
                 root: Union[Path, str],
                 split: str = 'train', 
                 test_area: str = 'area4',
                 train_size: float = 0.8,
                 first_subsampling_dl = 0.02,
                 config = None,
                 classification = False,
                 class_choice = None,
                 transform = None, 
                 pre_transform = None):
        super().__init__(root, split, test_area, train_size, transform, pre_transform)
        self.first_subsampling_dl = first_subsampling_dl
        self.config = config
        self.classification = classification
        
    def __getitem__(self, idx):
        data = self.get(self.indices()[idx])
        return data
    
    def get(self, idx):        
        data = torch.load(self.data_files[idx], weights_only=False)   
        
        if self.transform:
            data = self.transform(data)
            
        point_set = data.pos.detach().cpu().numpy()
        point_features = torch.cat([data.rgb, data.normals], dim=-1)
        seg = data.y.detach().cpu().numpy()
            
        features = np.ones([point_set.shape[0], 1])    
        features = np.concatenate([features, point_set, point_features], axis=1)            
        return point_set, features, seg