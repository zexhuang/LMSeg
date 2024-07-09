import os
import argparse

import numpy as np
import geopandas as gpd

import startinpy
import pymeshlab as pml
import fast_simplification as sim

from typing import Optional, List, Union
from shapely.geometry import Polygon
from tqdm import tqdm
from pathlib import Path


def split_areas(url: str, 
                w_factor: int = 2, 
                h_factor: int = 3):
    from shapely.geometry import box
    from rio_tiler.io import Reader
    
    areas = []
    
    with Reader(url) as cog:
        crs = cog.dataset.crs
        bounds = cog.bounds
        
        # Tile bounds
        x_min, y_min, \
        x_max, y_max = bounds[0], bounds[1], \
                       bounds[2], bounds[3]               
                       
        # Tile size
        w = x_max - x_min
        h = y_max - y_min
                       
        area_w = w // w_factor
        area_h = h // h_factor
        
        # Top-left corner
        window_bounds = (x_min, y_max - area_h, x_min + area_w, y_max)        
        
        for _, i in enumerate(tqdm(list(range(0, int(w // area_w))))):
            for j in list(range(0, int(h // area_h))):
                area_bounds = (window_bounds[0] + i * area_w,
                               window_bounds[1] + j * -area_h,
                               window_bounds[2] + i * area_w,
                               window_bounds[3] + j * -area_h)
                areas.append(box(*area_bounds))            
        
        gdf = gpd.GeoDataFrame(geometry=areas, crs=crs.data['init'])
        return gdf


def get_grid_df(url: str, 
                size: int, 
                stride: List[int],
                area: Polygon = None,
                thred: float = 0.02):
    from shapely.geometry import box
    from rio_tiler.io import Reader
    
    grids = []
    
    with Reader(url) as cog:
        crs = cog.dataset.crs
        bounds = cog.bounds if area is None else area.bounds
        
        # Tile bounds
        x_min, y_min, \
        x_max, y_max = bounds[0], bounds[1], \
                       bounds[2], bounds[3]
        # Tile size
        w = x_max - x_min
        h = y_max - y_min
        # Top-left corner
        window_bounds = (x_min, y_max - size, x_min + size, y_max)
    
        for s in stride:
            # Move from top to down, left to right
            for _, i in enumerate(tqdm(list(range(0, int(w // s))))):
                for j in list(range(0, int(h // s))):
                    temp_bounds = (window_bounds[0] + i * s,
                                   window_bounds[1] + j * -s,
                                   window_bounds[2] + i * s,
                                   window_bounds[3] + j * -s)

                    cog_part = cog.part(temp_bounds, dst_crs=crs, bounds_crs=crs)
                    cog_img = np.asarray(cog_part.data_as_image())
                    
                    pos, neg = np.count_nonzero(cog_img == 1), \
                               np.count_nonzero(cog_img == 0)
                    assert (neg + pos) == cog_img.size
                    
                    if (pos / (neg + pos)) > thred: grids.append(box(*temp_bounds))          
                        
    gdf = gpd.GeoDataFrame(geometry=grids, crs=crs.data['init'])
    return gdf


def save_points(fname: Union[str, Path],
                points: np.ndarray, 
                rgb: Optional[np.ndarray] = None, 
                intensity: Optional[np.ndarray] = None,
                mask: Optional[np.ndarray] = None):
    import trimesh
    
    pcd = trimesh.Trimesh(vertices=points, faces=None, 
                          face_normals=None, vertex_normals=None,
                          face_colors=None, vertex_colors=rgb,
                          vertex_attributes={'mask': mask, 'intensity': intensity},
                          process=True, 
                          validate=True,
                          merge_tex=True)
    pcd.export(fname)
    

def save_mesh(fname: Union[str, Path], 
              vertex: np.ndarray, 
              face: np.ndarray, 
              v_normal: Optional[np.ndarray] = None, 
              f_normal: Optional[np.ndarray] = None,
              v_rgb:  Optional[np.ndarray] = None, 
              v_mask: Optional[np.ndarray] = None,
              f_rgb:  Optional[np.ndarray] = None, 
              f_mask: Optional[np.ndarray] = None):
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertex, faces=face, 
                           face_normals=f_normal, vertex_normals=v_normal,
                           face_colors=f_rgb, vertex_colors=v_rgb,
                           face_attributes={'mask': f_mask}, 
                           vertex_attributes={'mask': v_mask},
                           process=True, 
                           validate=True,
                           merge_tex=True)
    mesh.export(fname)
    
    
def from_copc(url: str, 
              bounds: tuple, 
              classification: Optional[List[int]] = None, 
              ground_filt: bool = True,
              csf_res: float = 0.03,
              rigidness: int = 1,
              slope_smooth: bool = True):
    from laspy import Bounds, CopcReader
    with CopcReader.open(url) as copc:
        query = copc.query(Bounds(mins=bounds[:2], maxs=bounds[2:]))     
        
        if classification != None and hasattr(query, 'classification'):
            label_mask = []
            for cls in classification: 
                label_mask.append(query['classification'] == cls)
            label_mask = np.vstack(label_mask).transpose().any(1)
            query = query[label_mask]  
            
        if ground_filt:
            from csf import csf
            xyz = np.vstack((query.x, query.y, query.z)).transpose()
            non_ground_id, ground_id = csf(points=xyz, cloth_resolution=csf_res, rigidness=rigidness, slope_smooth=slope_smooth)
        else:
            non_ground_id, ground_id = None, None
            
    xyz = np.vstack((query.x, query.y, query.z)).astype(np.float32).transpose()
    rgb = np.vstack((query.red, query.green, query.blue)).astype(np.uint).transpose() // 256
    intensity = query.intensity
    return xyz, rgb, intensity, ground_id, non_ground_id


def from_cog(url: str, bounds: tuple, points: np.ndarray):
    from rio_tiler.io import Reader
    
    with Reader(url) as cog:
        crs = cog.dataset.crs
        cog_part = cog.part(bounds, dst_crs=crs, bounds_crs=crs)

        right, left, top, bottom = cog_part.bounds.right, cog_part.bounds.left, \
                                   cog_part.bounds.top, cog_part.bounds.bottom
                                   
        bound_w, bound_h = right - left, top - bottom
                        
        pixel_x = (points[:, 0] - left) / bound_w * (cog_part.width - 1)
        pixel_x = np.floor(pixel_x).astype(int)
            
        pixel_y = (top - points[:, 1]) / bound_h * (cog_part.height - 1)
        pixel_y = np.floor(pixel_y).astype(int)
        
        cog_img = np.asarray(cog_part.data_as_image())
        cog_attr = cog_img[pixel_y, pixel_x]
    return cog_attr


def clean_mesh(vertex:np.ndarray, face: np.ndarray):        
    ms = pml.MeshSet()    
    ms.add_mesh(pml.Mesh(vertex_matrix=vertex, face_matrix=face))
    
    ms.meshing_snap_mismatched_borders()
    
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_duplicate_vertices()
    
    ms.meshing_remove_null_faces()
    ms.meshing_remove_folded_faces()
    
    ms.meshing_remove_unreferenced_vertices()
    
    ms.meshing_repair_non_manifold_edges(method=0)
    ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
    
    ms.compute_normal_per_vertex(weightmode=2)
    ms.compute_normal_per_face()
    
    v, f = ms.current_mesh().vertex_matrix(), \
           ms.current_mesh().face_matrix()
    vn, fn = ms.current_mesh().vertex_normal_matrix(), \
             ms.current_mesh().face_normal_matrix()
    return v, f, vn, fn


def copc_to_poly(copc_url: str, 
                 mask_url: str,
                 rgb_url: str,
                 area_file: str,
                 grid_size: int,
                 grid_stride: List[int],
                 classification: Optional[List[int]] = None,
                 ground_filt: bool = True,
                 csf_res: float = 0.03,
                 rigidness: int = 1,
                 slope_smooth: bool = True,
                 agg: float = 0.0,
                 output: Union[str, None] = None):
    
    data_dir = Path(__file__).parent / output if output else Path(__file__).parent / 'BudjBimWall' 
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if os.path.exists(data_dir / area_file):
        areas = gpd.read_file(data_dir / area_file)
    else:
        areas = split_areas(mask_url)
        areas.to_file(data_dir / area_file, driver="GPKG") 
        
    for area_id, area in areas.iterrows():        
        mesh_fp = data_dir / 'mesh' / f'area{area_id + 1}'
        mesh_fp.mkdir(parents=True, exist_ok=True)
        
        pcd_fp = data_dir / 'pcd' / f'area{area_id + 1}'
        pcd_fp.mkdir(parents=True, exist_ok=True)
        
        grid_df = get_grid_df(mask_url, grid_size, grid_stride, area.geometry)
        grids = [row['geometry'] for _, row in grid_df.iterrows()]
        
        for _, grid in enumerate(tqdm(grids)):          
            fname = f"e{str(int(grid.centroid.x))}_n{str(int(grid.centroid.y))}_{grid_df.crs}.ply".replace(":", "")
            
            points, rgb, intensity, ground_id, _ = from_copc(copc_url, grid.bounds, classification, ground_filt, csf_res, rigidness, slope_smooth)
                    
            bounds = (points[:,0].min(), points[:,1].min(), 
                      points[:,0].max(), points[:,1].max())                        
            
            mask = from_cog(mask_url, bounds, points[:, :2])
            mask = mask.astype(np.float64).reshape(-1)
            
            save_points(pcd_fp / fname, points, rgb, intensity, mask)
            
            dt = startinpy.DT()
            dt.insert(points[ground_id])
            v, f = dt.points, dt.triangles
            v, f = sim.simplify(v.astype(np.float32), f.astype(np.float32), target_reduction=0.99, agg=agg)
            v, f, vn, fn = clean_mesh(v, f)
            
            bounds = (v[:,0].min(), v[:,1].min(), 
                      v[:,0].max(), v[:,1].max())
            
            v_rgb = from_cog(rgb_url, bounds, v[:, :2])
            v_rgb = v_rgb.astype(np.uint8)
            
            v_mask = from_cog(mask_url, bounds, v[:, :2])
            v_mask = v_mask.astype(np.float64).reshape(-1)
            
            import trimesh
            fv = trimesh.Trimesh(v, f).triangles_center
            
            bounds = (fv[:,0].min(), fv[:,1].min(), 
                      fv[:,0].max(), fv[:,1].max())
            
            f_rgb = from_cog(rgb_url, bounds, fv[:, :2])
            f_rgb = f_rgb.astype(np.uint8)

            f_mask = from_cog(mask_url, bounds, fv[:, :2])
            f_mask = f_mask.astype(np.float64).reshape(-1)
            
            save_mesh(mesh_fp / fname, v, f, v_normal=vn,  f_normal=fn, v_rgb=v_rgb, v_mask=v_mask, f_rgb=f_rgb, f_mask=f_mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BudjBim Mesh Dataset based on Hand Annotation Mask')
    parser.add_argument('--copc_url', type=str,  metavar='N',
                        default='https://objects.storage.unimelb.edu.au/4320_budjbimdata/COPC/AHD_10cm.copc.laz',
                        help='copc url')
    parser.add_argument('--mask_url', type=str,  metavar='N',
                        default='https://objects.storage.unimelb.edu.au/4320_budjbimdata/COGs/binary_mask_anno.tif',
                        help='binary image mask url'),
    parser.add_argument('--rgb_url', type=str,  metavar='N',
                        default='https://objects.storage.unimelb.edu.au/4320_budjbimdata/COGs/RGB_10cm.tif',
                        help='rgb image url')
    parser.add_argument('--area_file', type=str, default='areas.gpkg', metavar='N', 
                        help='area splits of BudjBim stone wall')
    parser.add_argument('--size', type=int, default=20, metavar='N', 
                        help='size of grid from top-left corner (default: 20 (meter))')
    parser.add_argument('--stride', type=list[int], default=[20], metavar='N', 
                        help='list of distances defined for grids to move from top to bottm, left to right (default: [20])')
    parser.add_argument('--classification', type=list, default=[2, 3, 4], metavar='N', # ground, low and medium veg class
                        help='point cloud query by ALS classification')
    parser.add_argument('--ground_filt', type=bool,  default=True,
                        help='ground point filtration with CSF')
    parser.add_argument('--csf_res', type=float, default=0.03, metavar='N',
                        help='cloth resolution: the grid size of cloth which is use to cover the terrain (default: 0.03)')
    parser.add_argument('--rigidness', type=int, default=1, metavar='N',
                        help='rigidness of scenes of CSF')
    parser.add_argument('--slope_smooth', type=bool,  default=True,
                        help='indicate whether to enable slope smoothing in CSF, defaults to True.')
    parser.add_argument('--agg', type=float, default=0, metavar='N',
                        help='controls how aggressively to decimate the mesh.')
    parser.add_argument('--output', type=str, default='BudjBimWall', metavar='N',
                        help='output folder')
    args = parser.parse_args()
    
    copc_to_poly(copc_url = args.copc_url, 
                 mask_url = args.mask_url, 
                 rgb_url = args.rgb_url,
                 area_file= args.area_file,
                 grid_size = args.size,
                 grid_stride = args.stride,
                 classification = args.classification, 
                 ground_filt = args.ground_filt,
                 csf_res = args.csf_res,
                 rigidness = args.rigidness,
                 slope_smooth = args.slope_smooth,
                 agg=args.agg,
                 output = args.output)