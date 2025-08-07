import os
import argparse

import numpy as np
import geopandas as gpd
import pyvista as pv
import CSF

from typing import Optional, List, Union, Tuple
from shapely.geometry import Polygon
from tqdm import tqdm
from pathlib import Path


def csf(
    points: np.ndarray, 
    cloth_resolution: float = 0.5, 
    rigidness: int = 1,
    class_threshold: float = 0.5, 
    iterations: int = 500, 
    slope_smooth: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies the CSF (Cloth Simulation Filter) algorithm to filter ground points in a point cloud.
    Source: https://github.com/Yarroudh/segment-lidar/blob/4a565e9df6221f5771ed7140a9ef47a363aa3400/segment_lidar/samlidar.py

    :param points: The input point cloud as a NumPy array, where each row represents a point with x, y, z coordinates.
    :type points: np.ndarray
    :param class_threshold: The threshold value for classifying points as ground/non-ground, defaults to 0.5.
    :type class_threshold: float, optional
    :param cloth_resolution: The resolution value for cloth simulation, defaults to 0.5.
    :type cloth_resolution: float, optional
    :param iterations: The number of iterations for the CSF algorithm, defaults to 500.
    :type iterations: int, optional
    :param slope_smooth: A boolean indicating whether to enable slope smoothing, defaults to True.
    :type slope_smooth: bool, optional
    :return: A tuple containing two arrays: non-ground (filtered) points indinces and ground points indices.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    csf = CSF.CSF()
    csf.params.cloth_resolution = cloth_resolution
    csf.params.rigidness = rigidness 
    csf.params.interations = iterations
    csf.params.class_threshold = class_threshold
    csf.params.bSloopSmooth = slope_smooth
    
    csf.setPointCloud(points[:, :3])
    ground = CSF.VecInt()
    non_ground = CSF.VecInt()
    csf.do_filtering(ground, non_ground)

    os.remove('cloth_nodes.txt')

    return np.asarray(non_ground, dtype=np.int32),\
           np.asarray(ground, dtype=np.int32)
           

def split_areas(
    url: str, 
    w_factor: int = 2, 
    h_factor: int = 3
):
    from shapely.geometry import box
    from rio_tiler.io import Reader
    
    areas = []
    
    with Reader(url) as cog:
        crs = cog.dataset.crs
        # Tile bounds
        x_min, y_min, x_max, y_max = cog.bounds         
        # Tile size
        w, h = x_max - x_min, y_max - y_min
        # Area size
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
        
        gdf = gpd.GeoDataFrame(geometry=areas, crs=crs.to_string())
        return gdf


def get_grid_df(
    url: str, 
    size: int, 
    stride: List[int],
    area: Polygon = None,
    thred: float = 0.02
):
    from shapely.geometry import box
    from rio_tiler.io import Reader
    
    grids = []
    
    with Reader(url) as cog:
        crs = cog.dataset.crs
        bounds = cog.bounds if area is None else area.bounds
        # Tile bounds
        x_min, y_min, x_max, y_max = bounds
        # Tile size
        w, h = x_max - x_min, y_max - y_min
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
                    
                    pos, neg = np.count_nonzero(cog_img >= 1), \
                               np.count_nonzero(cog_img == 0)
                    assert (neg + pos) == cog_img.size
                    
                    if (pos / (neg + pos)) > thred: grids.append(box(*temp_bounds))          
                        
    gdf = gpd.GeoDataFrame(geometry=grids, crs=crs.to_string())
    return gdf


def save_points(
    fname: Union[str, Path],
    points: np.ndarray, 
    rgb: Optional[np.ndarray] = None, 
    intensity: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None
):
    import trimesh
    
    pcd = trimesh.Trimesh(vertices=points, faces=None, 
                          face_normals=None, vertex_normals=None,
                          face_colors=None, vertex_colors=rgb,
                          vertex_attributes={'mask': mask, 'intensity': intensity},
                          process=True, 
                          validate=True,
                          merge_tex=True)
    pcd.export(fname)
    

def save_mesh(
    fname: Union[str, Path], 
    vertex: np.ndarray, 
    face: np.ndarray, 
    v_normal: Optional[np.ndarray] = None, 
    f_normal: Optional[np.ndarray] = None,
    v_rgb: Optional[np.ndarray] = None, 
    v_mask: Optional[np.ndarray] = None,
    f_rgb: Optional[np.ndarray] = None, 
    f_mask: Optional[np.ndarray] = None,
    agg: int = 0
):
    # Wrap masks into dict with key 'mask' if they exist
    vertex_attributes = {'mask': v_mask} if v_mask is not None else None
    face_attributes = {'mask': f_mask} if f_mask is not None else None

    import trimesh
    # Create Trimesh object
    mesh = trimesh.Trimesh(
        vertices=vertex,
        faces=face,
        vertex_normals=v_normal,
        face_normals=f_normal,
        vertex_colors=v_rgb,
        face_colors=f_rgb,
        vertex_attributes=vertex_attributes,
        face_attributes=face_attributes,
        process=True,
        validate=True,
        merge_tex=True
    )

    # Clean up mesh
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    mesh.fill_holes()
    mesh.fix_normals()

    # Simplify mesh (in-place)
    mesh.simplify_quadric_decimation(percent=0.01, aggression=agg)

    # Export mesh to file
    mesh.export(fname)
    
    
def from_copc(
    url: str, 
    bounds: tuple, 
    classification: Optional[List[int]] = None, 
    ground_filt: bool = True,
    csf_res: float = 0.05,
    rigidness: int = 1,
    slope_smooth: bool = True
):
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
            xyz = np.vstack((query.x, query.y, query.z)).transpose()
            non_ground_id, ground_id = csf(points=xyz, cloth_resolution=csf_res, rigidness=rigidness, slope_smooth=slope_smooth)
        else:
            non_ground_id, ground_id = None, None
            
    xyz = np.vstack((query.x, query.y, query.z)).astype(np.float32).transpose()
    rgb = np.vstack((query.red, query.green, query.blue)).astype(np.uint).transpose() // 256  # 16-bit to 8-bit RGB
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


def copc_to_poly_by_area(
    copc_url: str,
    mask_url: str,
    rgb_url: str,
    area_file: str,
    grid_size: int,
    grid_stride: List[int],
    classification: Optional[List[int]] = None,
    ground_filt: bool = True,
    csf_res: float = 0.05,
    rigidness: int = 1,
    slope_smooth: bool = True,
    agg: float = 0.0,
    output: Union[str, None] = None,
):
    # Setup output directory
    data_dir = Path(__file__).parent / output if output else Path(__file__).parent / "BudjBimWall"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load or generate areas GeoDataFrame
    area_path = data_dir / area_file
    if os.path.exists(area_path):
        areas = gpd.read_file(area_path)
    else:
        areas = split_areas(mask_url)
        areas.to_file(data_dir / "areas.gpkg", driver="GPKG")

    # Process each area
    for area_id, area in areas.iterrows():
        mesh_fp = data_dir / "mesh" / f"area{area_id + 1}"
        mesh_fp.mkdir(parents=True, exist_ok=True)

        pcd_fp = data_dir / "pcd" / f"area{area_id + 1}"
        pcd_fp.mkdir(parents=True, exist_ok=True)

        grid_df = get_grid_df(mask_url, grid_size, grid_stride, area.geometry)
        grids = [row["geometry"] for _, row in grid_df.iterrows()]

        # Iterate over grids within area
        for _, grid in enumerate(tqdm(grids)):
            fname = f"e{int(grid.centroid.x)}_n{int(grid.centroid.y)}_{grid_df.crs}.ply".replace(":", "")

            points, rgb, intensity, ground_id, _ = from_copc(
                copc_url, grid.bounds, classification, ground_filt, csf_res, rigidness, slope_smooth
            )

            if points.size == 0: 
                continue      # Ad-hoc fix to skip current iteration for empty point clouds returned  

            bounds = (points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max())
            mask = from_cog(mask_url, bounds, points[:, :2]).astype(np.float64).reshape(-1)
            save_points(pcd_fp / fname, points, rgb, intensity, mask)

            ground_points = points[ground_id].copy()
            ground_points[:, 2] = 0  # Flatten Z for triangulation
            pcd = pv.PolyData(ground_points)

            # Generate surface mesh via 2D Delaunay triangulation
            surface = pcd.delaunay_2d()
            surface.points[:, 2] = points[ground_id][:, 2]  # Restore original Z values
            surface = surface.clean()
            surface.compute_normals(point_normals=True, cell_normals=True, inplace=True)

            # Extract vertices, faces, normals, and colors
            v, f = surface.points, surface.faces.reshape(-1, 4)[:, 1:]
            vn, fn = surface.point_data["Normals"], surface.cell_data["Normals"]
            v_rgb = rgb[ground_id]
            f_rgb = v_rgb[f].mean(axis=1).astype(np.uint8)
            fv = v[f].mean(axis=1)

            # Extract vertex mask from COG
            v_bounds = (v[:, 0].min(), v[:, 1].min(), v[:, 0].max(), v[:, 1].max())
            try:
                v_mask = from_cog(mask_url, v_bounds, v[:, :2]).astype(np.float64).reshape(-1)
            except Exception as e:
                print(f"Unexpected error extracting vertex mask: {e}")
                continue

            # Extract face mask from COG
            f_bounds = (fv[:,0].min(), fv[:,1].min(), fv[:,0].max(), fv[:,1].max())
            try:
                f_mask = from_cog(mask_url, f_bounds, fv[:, :2]).astype(np.float64).reshape(-1)
            except Exception as e:
                print(f"Unexpected error extracting face mask: {e}")
                continue

            # Save mesh with all attributes
            save_mesh(
                mesh_fp / fname,
                v,
                f,
                v_normal=vn,
                f_normal=fn,
                v_rgb=v_rgb,
                v_mask=v_mask,
                f_rgb=f_rgb,
                f_mask=f_mask,
                agg=agg,
            )
            
            
def copc_to_poly(
    copc_url: str, 
    rgb_url: str,
    grid_size: int,
    grid_stride: List[int],
    grid_file: str,
    classification: Optional[List[int]] = None,
    ground_filt: bool = True,
    csf_res: float = 0.05,
    rigidness: int = 1,
    slope_smooth: bool = True,
    agg: float = 0.0,
    output: Union[str, None] = None
):
    
    data_dir = Path(__file__).parent / output if output else Path(__file__).parent / 'BudjBimLandscape' 
    data_dir.mkdir(parents=True, exist_ok=True)
           
    mesh_fp = data_dir / 'mesh'
    mesh_fp.mkdir(parents=True, exist_ok=True)
    
    if os.path.exists(data_dir / grid_file):
        grid_df = gpd.read_file(data_dir / grid_file)
    else:
        grid_df = get_grid_df(rgb_url, grid_size, grid_stride)
        grid_df.to_file(data_dir / 'grids.gpkg', driver="GPKG")
    
    grids = [row['geometry'] for _, row in grid_df.iterrows()]
    
    for _, grid in enumerate(tqdm(grids)):          
        fname = f"e{str(int(grid.centroid.x))}_n{str(int(grid.centroid.y))}_{grid_df.crs}.ply".replace(":", "")
        
        points, rgb, intensity, ground_id, _ = from_copc(copc_url, grid.bounds, classification, ground_filt, csf_res, rigidness, slope_smooth)
        if points.size == 0: 
            continue      # Ad-hoc fix to skip current iteration for empty point clouds returned                
        
        ground_points = points[ground_id].copy()
        ground_points[:, 2] = 0  # Flatten Z for triangulation
        pcd = pv.PolyData(ground_points)

        # Generate surface mesh via 2D Delaunay triangulation
        surface = pcd.delaunay_2d()
        surface.points[:, 2] = points[ground_id][:, 2]  # Restore original Z values
        surface = surface.clean()
        surface.compute_normals(point_normals=True, cell_normals=True, inplace=True)

        # Extract vertices, faces, normals, and colors
        v, f = surface.points, surface.faces.reshape(-1, 4)[:, 1:]
        vn, fn = surface.point_data["Normals"], surface.cell_data["Normals"]
        v_rgb = rgb[ground_id]
        f_rgb = v_rgb[f].mean(axis=1).astype(np.uint8)
        
        save_mesh(mesh_fp / fname, v, f, v_normal=vn, f_normal=fn, v_rgb=v_rgb, f_rgb=f_rgb, agg=agg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BudjBim Mesh Dataset based on Hand Annotation Mask')
    parser.add_argument('--dataset', type=str, default="BudjBimWall", metavar='N', 
                        help='choice of BudjBim dataset')
    parser.add_argument('--copc_url', type=str,  metavar='N',
                        default='https://objects.storage.unimelb.edu.au/4320_budjbimdata/COPC/budj_bim.copc.laz',
                        help='copc url')
    parser.add_argument('--mask_url', type=str,  metavar='N',
                        default='https://objects.storage.unimelb.edu.au/4320_budjbimdata/COGs/binary_mask_anno.tif',
                        help='binary image mask url'),
    parser.add_argument('--rgb_url', type=str,  metavar='N',
                        default='https://objects.storage.unimelb.edu.au/4320_budjbimdata/COGs/RGB_10cm.tif',
                        help='rgb image url')
    parser.add_argument('--area_file', type=str, default='areas.gkpg', metavar='N', 
                        help='area splits of BudjBim stone wall')
    parser.add_argument('--size', type=int, default=40, metavar='N', 
                        help='size of grid from top-left corner (default: 40 (meter))')
    parser.add_argument('--stride', type=list[int], default=[20], metavar='N', 
                        help='list of distances defined for grids to move from top to bottm, left to right (default: [20])')
    parser.add_argument('--grid_file', type=str, default=None, metavar='N', 
                        help='rectangular grid of BudjBim landscape')
    parser.add_argument('--classification', type=list, default=[2, 3, 4], metavar='N', # ground, low and medium veg class
                        help='point cloud query by ALS classification')
    parser.add_argument('--ground_filt', type=bool,  default=True,
                        help='ground point filtration with CSF')
    parser.add_argument('--csf_res', type=float, default=0.1, metavar='N',
                        help='cloth resolution: the grid size of cloth which is use to cover the terrain (default: 0.05)')
    parser.add_argument('--rigidness', type=int, default=1, metavar='N',
                        help='rigidness of scenes of CSF')
    parser.add_argument('--slope_smooth', type=bool,  default=True,
                        help='indicate whether to enable slope smoothing in CSF, defaults to True.')
    parser.add_argument('--agg', type=float, default=0, metavar='N',
                        help='controls how aggressively to decimate the mesh.')
    parser.add_argument('--output', type=str, default='BBW/BudjBimWall', metavar='N',
                        help='output folder')
    args = parser.parse_args()
    
    if args.dataset == "BudjBimWall":
        copc_to_poly_by_area(
            copc_url = args.copc_url, 
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
            output = args.output
        )
    elif args.dataset == "BudjBimLandscape":
        copc_to_poly(
            copc_url = args.copc_url, 
            rgb_url = args.rgb_url,
            grid_size = args.size,
            grid_stride = args.stride,
            grid_file= args.grid_file,
            classification = args.classification, 
            ground_filt = args.ground_filt,
            csf_res = args.csf_res,
            rigidness = args.rigidness,
            slope_smooth = args.slope_smooth,
            agg=args.agg,
            output = args.output
        )