"""Call the unmodified milliEgo projector with explicit, externally verified parameters."""
from pathlib import Path
import importlib.util
from functools import lru_cache
@lru_cache(maxsize=1)
def _original():
    path=Path(__file__).resolve().parents[1]/"upstream/utility/pcl2depth.py"
    spec=importlib.util.spec_from_file_location("milliego_original_projection",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.velo_points_2_pano
def project_with_original(points, *, v_res, h_res, v_fov, h_fov, max_v, depth):
    # Preserve source collision order, pixel offsets, boundaries and empty-input behavior.
    # Cropping, temporal aggregation and resize are caller responsibilities.
    return _original()(points,v_res=v_res,h_res=h_res,v_fov=v_fov,h_fov=h_fov,max_v=max_v,depth=depth)
