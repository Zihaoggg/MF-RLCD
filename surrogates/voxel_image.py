from pathlib import Path

import numpy as np


def _read_cells(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    count = None
    lamellae = None
    orientations = None
    for index, line in enumerate(lines):
        section = line.strip()
        if section == "**cell":
            count = int(lines[index + 1])
        elif section == "*lam":
            values = []
            for row in lines[index + 1:]:
                if row.lstrip().startswith("*"):
                    break
                values.extend(map(int, row.split()))
            lamellae = np.asarray(values, dtype=np.float64) - 1
        elif section == "*ori":
            if count is None:
                raise ValueError("Tessellation cell count must precede orientations")
            orientations = np.asarray([
                [float(value) for value in row.split()]
                for row in lines[index + 2:index + 2 + count]
            ], dtype=np.float64)
    if count is None or count < 1:
        raise ValueError("Missing tessellation cell count")
    if orientations is None or orientations.shape != (count, 3):
        raise ValueError("Expected three orientation values per cell")
    if lamellae is None or lamellae.shape != (count,):
        raise ValueError("Expected one lamellar phase label per cell")
    return np.column_stack((lamellae, orientations))


def _read_voxels(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    values = []
    reading = False
    for line in lines:
        if reading:
            if line.lstrip().startswith("*"):
                break
            values.extend(map(int, line.split()))
        elif line.strip().lower() == "ascii":
            reading = True
    if len(values) != 32 ** 3:
        raise ValueError(f"Expected 32768 voxel cell IDs, found {len(values)}")
    return np.asarray(values, dtype=np.int64).reshape((32, 32, 32), order="F")


def image_gen_3D_BO(ms_id, sh_dir):
    directory = Path(sh_dir)
    if directory.name == "generate_tess.sh":
        directory = directory.parent
    tessellations = sorted(directory.glob("*.tess"))
    if not tessellations:
        raise FileNotFoundError(f"No tessellation found in {directory}")
    tessellation = tessellations[0]
    raster = tessellation.with_suffix(".tesr")
    if not raster.is_file():
        raise FileNotFoundError(f"Missing matching raster: {raster}")
    features = _read_cells(tessellation)
    cell_ids = _read_voxels(raster)
    if cell_ids.min() < 1 or cell_ids.max() > len(features):
        raise ValueError("Raster contains a cell ID outside the tessellation")
    return [features[cell_ids - 1]]
