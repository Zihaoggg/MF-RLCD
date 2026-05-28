from pathlib import Path

import numpy as np


def image_gen_3D_BO(ms_id, sh_dir):
    base_dir = Path(str(sh_dir).replace("generate_tess.sh", ""))
    tess_file = _first_file(base_dir, "*.tess")
    tesr_file = _first_file(base_dir, "*.tesr")
    image = _build_input_image(tess_file, tesr_file)
    return [image[:, :, :, 1:5]]


def image_gen_3D_CNN(dir_original, dir_raw_input, dir_raw_target, dir_preprocessed, move, gen):
    import shutil

    original = Path(dir_original)
    raw_input = Path(dir_raw_input)
    raw_target = Path(dir_raw_target)
    preprocessed = Path(dir_preprocessed)

    if move:
        for pattern, target_dir in [("**/*.tesr", raw_input), ("**/*.tess", raw_input), ("**/*.x1", raw_target)]:
            for source in original.glob(pattern):
                copied = shutil.copy(source, target_dir)
                new_name = target_dir / str(source).replace(str(original), "").replace("/", "_").replace("\\", "_")
                Path(copied).rename(new_name)

    if not gen:
        return

    import pandas as pd
    from scipy import interpolate

    raster_size = 32
    col_names = ["step", "INCR", "Fx", "Fy", "Fz", "area A", "TIME"]
    area = 0.000001
    strain = np.linspace(0, 0.015, 7)
    strain_new = np.linspace(0, 0.015, 70)
    images, idx_list, stress_rows, moduli, yields = [], [], [], [], []

    for force_file in raw_target.glob("**/*.x1"):
        idx = remove_suffix(str(force_file).replace(str(raw_target), ""), "_post.force.x1")
        data = pd.read_csv(force_file, skiprows=[0, 1], delimiter="   ", names=col_names, engine="python")
        stress = [value / area for value in data["Fx"].tolist()]
        modulus = (stress[1] / 0.0025 + stress[2] / 0.005) / 2
        offset = modulus * strain_new - modulus * 0.002
        stress_interp = interpolate.interp1d(strain, stress, "linear")(strain_new)
        crossing = np.argwhere(np.diff(np.sign(stress_interp - offset))).flatten()
        yield_strength = stress_interp[crossing][0]
        image = _build_input_image(_first_file(raw_input, f"{idx}*.tess"), _first_file(raw_input, f"{idx}*.tesr"))
        images.append(image)
        idx_list.append(idx)
        stress_rows.append(stress)
        moduli.append(modulus)
        yields.append(yield_strength)

    stress_df = pd.DataFrame(stress_rows, columns=["0.00", "0.25", "0.50", "0.75", "1.00", "1.25", "1.50"])
    stress_df["E"] = moduli
    stress_df["Yield"] = yields
    stress_df.index = idx_list
    preprocessed.mkdir(parents=True, exist_ok=True)
    stress_df.to_csv(preprocessed / "stress.csv")
    np.save(preprocessed / "input", images)


def position_gen():
    coords = np.linspace(0.015, 0.985, 32)
    with open("./positions", "w", encoding="utf-8") as fh:
        for x in coords:
            for y in coords:
                for z in coords:
                    fh.write(f"{x} {y} {z}\n")


def remove_suffix(input_string, suffix):
    text = str(input_string)
    return text[: -len(suffix)] if suffix and text.endswith(suffix) else text


def image_gen_3D_cGAN(typeid=4):
    data_types = ["seq", "strain", "strain-eq", "strain-pl", "strain-pl-eq"]
    for grain in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
        raw_dir = Path(f"/home/xiao/projects/inverse_mat_des/ML/dataset/{grain}_cGAN/{data_types[typeid]}")
        stpoint_dir = Path(f"/home/xiao/projects/inverse_mat_des/ML/dataset/{grain}_cGAN/stpoint")
        raw_input = Path("/home/xiao/projects/inverse_mat_des/ML/dataset/all_input")
        preprocessed = Path(f"/home/xiao/projects/inverse_mat_des/ML/dataset/{grain}_cGAN/preprocessed_{data_types[typeid]}")
        preprocessed.mkdir(parents=True, exist_ok=True)

        target_images, input_images, idx_list = [], [], []
        for target_file in raw_dir.glob(f"*{data_types[typeid]}.step2"):
            idx = remove_suffix(str(target_file).replace(str(raw_dir), ""), f"_{data_types[typeid]}.step2")
            stpoint = np.loadtxt(stpoint_dir / f"{idx}.stpoint")
            target = np.loadtxt(target_file)
            target_image = np.zeros((32, 32, 32, 1))
            counter = 0
            for x in range(32):
                for y in range(32):
                    for z in range(32):
                        target_image[x, y, z] = target[int(stpoint[counter]) - 1]
                        counter += 1
            input_image = _build_input_image(_first_file(raw_input, f"{idx}*.tess"), _first_file(raw_input, f"{idx}*.tesr"))
            target_images.append(target_image)
            input_images.append(input_image)
            idx_list.append(idx)

        np.save(preprocessed / "target", target_images)
        np.save(preprocessed / "input", input_images)
        np.save(preprocessed / "idx", idx_list)


def _first_file(base_dir, pattern):
    matches = sorted(Path(base_dir).glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matches {Path(base_dir) / pattern}")
    return matches[0]


def _read_tess(tess_file):
    lines = Path(tess_file).read_text(encoding="utf-8", errors="ignore").splitlines()
    num_cells = 0
    lam_start = None
    lam_list = []
    orientations = []

    for index, line in enumerate(lines):
        if "**cell" in line:
            num_cells = int(lines[index + 1].strip())
        elif "*lam" in line:
            lam_start = index + 1
        elif "*group" in line and lam_start is not None:
            for lam_line in lines[lam_start:index]:
                lam_list.extend(int(value) - 1 for value in lam_line.split())
        elif "*ori" in line:
            for ori_line in lines[index + 2:index + 2 + num_cells]:
                values = ori_line.split()
                if len(values) >= 3:
                    orientations.append([float(values[0]), float(values[1]), float(values[2])])
                else:
                    orientations.append([float(ori_line[2:17]), float(ori_line[20:35]), float(ori_line[38:53])])

    if len(lam_list) < num_cells:
        lam_list.extend([0] * (num_cells - len(lam_list)))
    return lam_list, orientations


def _read_tesr_cell_ids(tesr_file):
    lines = Path(tesr_file).read_text(encoding="utf-8", errors="ignore").splitlines()
    cell_ids = []
    reading = False
    for line in lines:
        if reading:
            for value in line.split():
                try:
                    cell_ids.append(int(value))
                except ValueError:
                    pass
        if line.strip().lower() == "ascii":
            reading = True
    return cell_ids[: 32 * 32 * 32]


def _build_input_image(tess_file, tesr_file):
    lam_list, orientations = _read_tess(tess_file)
    cell_ids = _read_tesr_cell_ids(tesr_file)
    image = np.zeros((32, 32, 32, 5), dtype=np.float64)
    counter = 0
    for z in range(32):
        for y in range(32):
            for x in range(32):
                cell_id = cell_ids[counter]
                cell_index = cell_id - 1
                image[x, y, z, 0] = cell_id
                image[x, y, z, 1] = lam_list[cell_index]
                image[x, y, z, 2:5] = orientations[cell_index]
                counter += 1
    return image
