from __future__ import annotations

import glob
import json
import os
import random
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from subprocess import TimeoutExpired

import matlab.engine
import numpy as np
import torch
from experiment_tracking import ExperimentTracker, build_metric_snapshot, format_metric_delta
from importlib.machinery import SourceFileLoader
from joblib import load

sys.path.insert(1, './Common_lib/')
from cgan_3d import Discriminator, Generator
from preprocessing import DeNormalizeData
from microstructure_image import image_gen_3D_BO
from yield_strength import yield_strength_cal

BETA_CANDIDATES = [5, 10, 15]
MAX_N_PRIOR_BETA = max(BETA_CANDIDATES)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

@dataclass
class MicrostructureParams:

    n_priorBeta: int = 5
    angles: np.ndarray = None
    seeds: np.ndarray = None
    lam_ratio: float = 1.0

    def __post_init__(self):
        if self.angles is None:
            self.angles = np.random.uniform(0, 360, self.n_priorBeta * 3)
        if self.seeds is None:
            self.seeds = np.random.randint(1, 30, self.n_priorBeta * 3)

    def to_vector(self):

        return np.concatenate([self.angles, self.seeds.astype(float), [self.lam_ratio]])

    @classmethod
    def from_vector(cls, vector, n_priorBeta=5, max_n_priorBeta=None):

        v = np.asarray(vector, dtype=np.float64)

        if max_n_priorBeta is None:
            max_n_priorBeta = n_priorBeta

        if len(v) < (max_n_priorBeta * 6 + 1):
            angles = v[:n_priorBeta * 3]
            seeds = v[n_priorBeta * 3:n_priorBeta * 6]
            lam_ratio = float(v[-1]) if len(v) > n_priorBeta * 6 else 1.0
            return cls(n_priorBeta=n_priorBeta, angles=angles, seeds=seeds, lam_ratio=lam_ratio)

        angles_full = v[:max_n_priorBeta * 3]
        seeds_full  = v[max_n_priorBeta * 3:max_n_priorBeta * 6]
        lam_ratio   = float(v[max_n_priorBeta * 6])

        phi1 = angles_full[0:n_priorBeta]
        Phi  = angles_full[max_n_priorBeta:max_n_priorBeta + n_priorBeta]
        phi2 = angles_full[2 * max_n_priorBeta:2 * max_n_priorBeta + n_priorBeta]
        angles = np.concatenate([phi1, Phi, phi2]).astype(np.float64)

        sx = seeds_full[0:n_priorBeta]
        sy = seeds_full[max_n_priorBeta:max_n_priorBeta + n_priorBeta]
        sz = seeds_full[2 * max_n_priorBeta:2 * max_n_priorBeta + n_priorBeta]
        seeds = np.concatenate([sx, sy, sz]).astype(np.float64)

        return cls(n_priorBeta=n_priorBeta, angles=angles, seeds=seeds, lam_ratio=lam_ratio)

class GraphSAGEPredictor:
    def __init__(self,
                 checkpoint_path='./models/checkpoints/checkpoint.state_dict.pth',
                 label_norm_path='./models/checkpoints/norm.npz',
                 max_node_num=300,
                 device='cuda',
                 gm_model_path='./Common_lib/graph_sage_model.py',
                 meta_path=None
                 ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.max_node_num = int(max_node_num)
        self.meta_path = meta_path

        p = os.path.abspath(checkpoint_path)
        sd = torch.load(p, map_location=self.device)

        gm_path = Path(gm_model_path)
        if not gm_path.exists():
            raise FileNotFoundError(f'gm_model_path not found: {gm_path}')
        GraphModel = SourceFileLoader('graphsage_model', str(gm_path)).load_module().GraphModel

        hidden = None
        n_layers = None
        try:
            if self.meta_path and os.path.exists(self.meta_path):
                meta = json.load(open(self.meta_path))
                hidden = int(meta.get('latent_dim1', 0))
                n_layers = int(meta.get('latent_dim2', 0))
        except Exception:
            pass

        if hidden is None:
            hidden = sd['enc.weight'].shape[0]
        if n_layers is None:
            layer_ids = set()
            for k in sd.keys():
                if k.startswith('layers.'):
                    try:
                        layer_ids.add(int(k.split('.')[1]))
                    except Exception:
                        pass
            n_layers = (max(layer_ids) + 1) if layer_ids else 2

        self.model = GraphModel(self.max_node_num, 5, hidden, n_layers).to(self.device)
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()

        norm = np.load(label_norm_path, allow_pickle=True)['norm']
        self.y_mean = float(norm[0]); self.y_std = float(norm[1])

    @staticmethod
    def _sym_norm_with_self_loop(adj_np: np.ndarray) -> np.ndarray:
        A = adj_np.copy()
        np.fill_diagonal(A, 1.0)
        deg = A.sum(axis=0)
        deg[deg == 0] = 1.0
        D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
        return D_inv_sqrt @ A @ D_inv_sqrt

    def _read_tess_tesr(self, save_dir):
        tess_file = next(Path(save_dir).glob("*.tess"))
        tesr_file = next(Path(save_dir).glob("*.tesr"))
        with open(tess_file, 'r') as f: lines_tess = f.readlines()
        with open(tesr_file, 'r') as f: lines_tesr = f.readlines()

        for ln, line in enumerate(lines_tess):
            if '**cell' in line:
                num_cells = int(lines_tess[ln + 1]); break

        ori_list = []
        for ln, line in enumerate(lines_tess):
            if '*ori' in line:
                for k in range(ln + 2, ln + 2 + num_cells):
                    r1 = float(lines_tess[k][2:17])
                    r2 = float(lines_tess[k][20:35])
                    r3 = float(lines_tess[k][38:53])
                    ori_list.append([r1, r2, r3])
                break

        cellids = []
        for ln, line in enumerate(lines_tesr):
            if 'ascii\n' in line:
                for i in range(ln + 1, len(lines_tesr) - 1):
                    for s in lines_tesr[i].split():
                        cellids.append(int(s))
                break
        raster = 32
        cell_grid = np.array(cellids, dtype=np.int32).reshape(raster, raster, raster)

        alpha_prop = None
        stgroups = list(Path(save_dir).glob('*.stgroup'))
        if stgroups:
            with open(stgroups[0], 'r') as f:
                ls = f.readlines()
            try:
                alpha_prop = float(ls[1].strip())
            except Exception:
                alpha_prop = float(ls[0].strip())

        return num_cells, np.array(ori_list, dtype=np.float32), cell_grid, float(alpha_prop if alpha_prop is not None else 0.5)

    def _build_graph(self, save_dir):
        num_cells, ori_list, cell_grid, alpha_prop = self._read_tess_tesr(save_dir)

        edges = set()
        X, Y, Z = cell_grid.shape
        for x in range(X - 1):
            a = cell_grid[x, :, :]; b = cell_grid[x + 1, :, :]
            mask = (a != b); ia, ib = a[mask], b[mask]
            for u, v in zip(ia, ib):
                if u != v:
                    u0, v0 = (u - 1), (v - 1)
                    if u0 > v0: u0, v0 = v0, u0
                    edges.add((u0, v0))
        for y in range(Y - 1):
            a = cell_grid[:, y, :]; b = cell_grid[:, y + 1, :]
            mask = (a != b); ia, ib = a[mask], b[mask]
            for u, v in zip(ia, ib):
                if u != v:
                    u0, v0 = (u - 1), (v - 1)
                    if u0 > v0: u0, v0 = v0, u0
                    edges.add((u0, v0))
        for z in range(Z - 1):
            a = cell_grid[:, :, z]; b = cell_grid[:, :, z + 1]
            mask = (a != b); ia, ib = a[mask], b[mask]
            for u, v in zip(ia, ib):
                if u != v:
                    u0, v0 = (u - 1), (v - 1)
                    if u0 > v0: u0, v0 = v0, u0
                    edges.add((u0, v0))

        N = num_cells

        flat = cell_grid.reshape(-1)
        voxel_count = np.bincount(flat - 1, minlength=N).astype(np.float32)

        feats = np.concatenate([
            np.array(ori_list, dtype=np.float32),
            voxel_count.reshape(-1, 1),
            np.full((N, 1), alpha_prop, dtype=np.float32)
        ], axis=1)

        K = min(self.max_node_num, N)
        top_idx = np.argsort(-voxel_count)[:K]
        id_map = {int(old): i for i, old in enumerate(top_idx.tolist())}

        adj = np.zeros((K, K), dtype=np.float32)
        for u, v in edges:
            if (u in id_map) and (v in id_map):
                iu, iv = id_map[u], id_map[v]
                adj[iu, iv] = 1.0
                adj[iv, iu] = 1.0

        node_attr = feats[top_idx]

        if node_attr.shape[0] > 1:
            m = node_attr[:, 3].mean()
            s = node_attr[:, 3].std()
            if s > 0:
                node_attr[:, 3] = (node_attr[:, 3] - m) / s
            else:
                node_attr[:, 3] = 0.0

        adj = self._sym_norm_with_self_loop(adj)

        adj = torch.from_numpy(adj).unsqueeze(0).to(self.device)
        node_attr = torch.from_numpy(node_attr).unsqueeze(0).to(self.device)
        return adj, node_attr

    @torch.no_grad()
    def predict_one(self, save_dir, t_value: float) -> float:
        adj, node_attr = self._build_graph(save_dir)
        t_tensor = torch.tensor([[t_value]], dtype=torch.float32, device=self.device)
        y = self.model(adjacency_matrix=adj, node_attr_matrix=node_attr, t_matrix=t_tensor)
        return float(y.squeeze().item() * self.y_std + self.y_mean)

    def predict_six(self, save_dir, strain_list) -> np.ndarray:
        vals = [self.predict_one(save_dir, float(t)) for t in strain_list]
        return np.asarray(vals, dtype=np.float32).reshape(1, 6)

class MicrostructureEnvironment:

    def __init__(self, gnn_model, cgan_generator, target_properties,
                 matlab_engine=None, work_dir='./opt_run/RL_Diffusion/'):
        self.gnn_model = gnn_model
        self.cgan_generator = cgan_generator
        self.target = target_properties
        self.eng = matlab_engine
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.strain_list = np.array([2.5, 5.0, 7.5, 10, 12.5, 15], dtype=np.float32)
        self.tracker = ExperimentTracker(self.work_dir)
        self.eval_cache = {}
        self.eval_history = []

        if self.eng is not None:
            try:
                self.eng.addpath(os.getcwd())
                self.eng.addpath('./matlab_scripts/')
                self.eng.addpath('./')
                print("MATLAB paths configured")
            except Exception as e:
                print(f"Failed to configure MATLAB paths: {e}")

        try:
            self.sc = load('./models/std_scalerSGD_alldata_200eps.bin')
        except:
            print("Warning: failed to load scaler; inverse normalization will be skipped")
            self.sc = None

        try:
            min_max = np.load('./models/min_max_final_all_data_20221026.npy')
            self.data_min = min_max[0]
            self.data_max = min_max[1]
        except:
            print("Warning: failed to load min-max normalization parameters")
            self.data_min = 0
            self.data_max = 1

        self.beta_candidates = list(BETA_CANDIDATES)
        self.max_n_priorBeta = int(MAX_N_PRIOR_BETA)
        self.default_n_priorBeta = int(self.beta_candidates[0])

        self.n_colonies_max = 1.0
        self.lamwidth_beta = 0.15

        self.eval_counter = 0

    def set_target(self, target_properties):

        self.target = target_properties

    def set_work_dir(self, work_dir, reset_counter=True):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = ExperimentTracker(self.work_dir)
        if reset_counter:
            self.eval_counter = 0

    def _make_params_cache_key(self, params_vector):
        v = np.asarray(params_vector, dtype=np.float64).reshape(-1)
        return tuple(np.round(v, 6).tolist())

    def summarize_metrics(self, E, sigma_y, Kt):
        return build_metric_snapshot(self.target, E, sigma_y, Kt, self.compute_reward)

    def compute_error_components(self, E, sigma_y, Kt):
        if E is None or sigma_y is None or Kt is None:
            return None
        E_err = abs(E - self.target.E_target) / self.target.E_target
        Sy_err = abs(sigma_y - self.target.sigma_y_target) / self.target.sigma_y_target
        Kt_err = abs(Kt - self.target.Kt_target) / abs(self.target.Kt_target)
        return {
            "E_error": float(E_err),
            "sigma_y_error": float(Sy_err),
            "Kt_error": float(Kt_err),
            "error_sum": float(E_err + Sy_err + Kt_err),
        }

    def _log_evaluation(self, eval_id, params_vector, snapshot, metadata=None, cached=False, eval_dir=None):
        metadata = metadata or {}
        params = np.asarray(params_vector, dtype=np.float64).reshape(-1)
        row = {
            "eval_id": int(eval_id),
            "cached": bool(cached),
            "eval_dir": "" if eval_dir is None else str(eval_dir),
            "target_E": float(self.target.E_target),
            "target_sigma_y": float(self.target.sigma_y_target),
            "target_Kt": float(self.target.Kt_target),
            "E": snapshot.E,
            "sigma_y": snapshot.sigma_y,
            "Kt": snapshot.Kt,
            "reward": float(snapshot.reward),
            "success": bool(snapshot.success),
            "E_error": snapshot.E_error,
            "sigma_y_error": snapshot.sigma_y_error,
            "Kt_error": snapshot.Kt_error,
            "n_priorBeta": int(self._decode_n_priorBeta(params)),
            "lam_ratio": float(params[self.max_n_priorBeta * 6]),
        }
        for key, value in metadata.items():
            row[key] = value
        self.eval_history.append(row)
        self.tracker.log_evaluation(row)

    def _decode_n_priorBeta(self, params_vector) -> int:

        v = np.asarray(params_vector, dtype=np.float64)
        need_len = self.max_n_priorBeta * 6 + 2
        if len(v) >= need_len:
            raw = float(v[self.max_n_priorBeta * 6 + 1])
            n = int(round(raw))
            if n in self.beta_candidates:
                return n

            return int(min(self.beta_candidates, key=lambda x: abs(x - n)))
        return self.default_n_priorBeta

    def evaluate_params(self, params_vector, metadata=None):

        metadata = metadata or {}
        self.eval_counter += 1
        eval_id = self.eval_counter
        cache_key = self._make_params_cache_key(params_vector)

        if cache_key in self.eval_cache:
            cached_E, cached_sigma_y, cached_Kt = self.eval_cache[cache_key]
            snapshot = self.summarize_metrics(cached_E, cached_sigma_y, cached_Kt)
            self._log_evaluation(eval_id, params_vector, snapshot, metadata=metadata, cached=True)
            print(
                f"[Eval {eval_id:04d}] cache-hit | "
                f"{format_metric_delta('E(GPa)', snapshot.E, self.target.E_target, snapshot.E_error, scale=1000.0)} | "
                f"{format_metric_delta('Sy', snapshot.sigma_y, self.target.sigma_y_target, snapshot.sigma_y_error)} | "
                f"{format_metric_delta('Kt', snapshot.Kt, self.target.Kt_target, snapshot.Kt_error)} | "
                f"reward={snapshot.reward:.3f} | success={snapshot.success}"
            )
            return cached_E, cached_sigma_y, cached_Kt

        n_priorBeta = self._decode_n_priorBeta(params_vector)
        params = MicrostructureParams.from_vector(
            params_vector,
            n_priorBeta=n_priorBeta,
            max_n_priorBeta=self.max_n_priorBeta
        )

        eval_dir = self.work_dir / f'eval_{eval_id}'
        eval_dir.mkdir(exist_ok=True)
        print(f"[Eval {eval_id:04d}] generating structure in {eval_dir}")

        try:
            images = self._generate_microstructure(params, eval_dir)
            if images is None:
                snapshot = self.summarize_metrics(None, None, None)
                self._log_evaluation(eval_id, params_vector, snapshot, metadata=metadata, cached=False, eval_dir=eval_dir)
                print(f"[Eval {eval_id:04d}] generation failed | reward={snapshot.reward:.3f} | success={snapshot.success}")
                return None, None, None

            if hasattr(self.gnn_model, 'predict_six'):
                stress = self.gnn_model.predict_six(eval_dir, self.strain_list)
            else:
                stress = self.gnn_model.predict(images)
                if self.sc is not None:
                    stress = self.sc.inverse_transform(stress)

            E, sigma_y = yield_strength_cal(stress)
            pred = self.cgan_generator(
                images[0].reshape(-1, 32, 32, 32, 4),
                training=False
            )
            Kt = self._calculate_kt(pred.numpy())
            result = (float(E[0]), float(sigma_y[0]), float(Kt))
            self.eval_cache[cache_key] = result

            snapshot = self.summarize_metrics(*result)
            self._log_evaluation(eval_id, params_vector, snapshot, metadata=metadata, cached=False, eval_dir=eval_dir)
            print(
                f"[Eval {eval_id:04d}] "
                f"{format_metric_delta('E(GPa)', snapshot.E, self.target.E_target, snapshot.E_error, scale=1000.0)} | "
                f"{format_metric_delta('Sy', snapshot.sigma_y, self.target.sigma_y_target, snapshot.sigma_y_error)} | "
                f"{format_metric_delta('Kt', snapshot.Kt, self.target.Kt_target, snapshot.Kt_error)} | "
                f"reward={snapshot.reward:.3f} | success={snapshot.success}"
            )
            return result

        except Exception as e:
            print(f"[Eval {eval_id:04d}] failed: {e}")
            snapshot = self.summarize_metrics(None, None, None)
            fail_meta = dict(metadata)
            fail_meta["error"] = str(e)
            self._log_evaluation(eval_id, params_vector, snapshot, metadata=fail_meta, cached=False, eval_dir=eval_dir)
            return None, None, None

    def _generate_microstructure(self, params: MicrostructureParams, save_dir):

        def _seed_stats_and_repair(seeds_xyz, min_gap=2, max_tries=200, rng_seed=123):

            coords = np.array(seeds_xyz, dtype=int).copy()
            n = coords.shape[0]
            uniq0 = len({tuple(v) for v in coords})
            mind0 = np.inf
            for i in range(n):
                for j in range(i + 1, n):
                    mind0 = min(mind0, np.max(np.abs(coords[i] - coords[j])))

            changed = False
            if uniq0 < n or mind0 < min_gap:
                rng = np.random.default_rng(rng_seed)
                placed = []
                for i in range(n):
                    v = coords[i]
                    ok = False
                    for _ in range(max_tries):

                        if (tuple(v) not in {tuple(p) for p in placed} and
                                all(np.max(np.abs(v - p)) >= min_gap for p in placed)):
                            ok = True
                            break
                        v = rng.integers(1, 30, size=3)
                    coords[i] = v
                    placed.append(v)
                changed = True

            uniq1 = len({tuple(v) for v in coords})
            mind1 = np.inf
            for i in range(n):
                for j in range(i + 1, n):
                    mind1 = min(mind1, np.max(np.abs(coords[i] - coords[j])))

            if changed:
                print(f"[seed-fix] duplicates:{n - uniq0}, min-cheby:{mind0} -> "
                      f"duplicates:{n - uniq1}, min-cheby:{mind1}")
            else:
                print(f"[seed-ok] duplicates:{n - uniq0}, min-cheby:{mind0}")

            return coords, changed

        def _write_seeds_file(coords_int, seed_file_path):

            with open(seed_file_path, 'w') as f:
                for x, y, z in coords_int:
                    f.write(f"{x * 0.03125} {y * 0.03125} {z * 0.03125}\n")

        def _run_neper_and_check(sh_path, save_dir_path, log_path, timeout_s=1200):

            try:
                with open(log_path, 'wb') as logf:
                    proc = subprocess.Popen(
                        [str(sh_path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT,
                        cwd=str(Path('.').resolve())
                    )
                    proc.communicate(timeout=timeout_s)
            except TimeoutExpired:
                proc.kill()
                print(f"Tessellation timeout for {save_dir_path}")
                return False

            tess_files = list(Path(save_dir_path).glob('*.tess'))
            tesr_files = list(Path(save_dir_path).glob('*.tesr'))
            ok = bool(tess_files and tesr_files)
            if not ok:
                print(f"No tess/tesr in {save_dir_path}. See {log_path}")
            return ok

        try:
            save_dir = Path(save_dir).resolve()

            n = params.n_priorBeta
            ori_list = []
            for i in range(n):
                phi1 = float(params.angles[i])
                Phi = float(params.angles[i + n])
                phi2 = float(params.angles[i + 2 * n])
                ori_list.append([phi1, Phi, phi2])
            oriBeta_inp = matlab.double(ori_list)

            self.eng.generate_microstructure(
                float(params.n_priorBeta),
                oriBeta_inp,
                float(self.n_colonies_max),
                float(self.lamwidth_beta),
                float(params.lam_ratio),
                float(1.0),
                'RL_eval',
                float(1.0),
                str(save_dir) + '/',
                nargout=0
            )

            seeds_int = np.column_stack([
                params.seeds[:n],
                params.seeds[n:2 * n],
                params.seeds[2 * n:3 * n]
            ]).astype(int)

            seeds_fixed, changed = _seed_stats_and_repair(seeds_int, min_gap=2)
            seed_file = save_dir / 'seeds'
            _write_seeds_file(seeds_fixed, seed_file)

            sh_name = save_dir / 'generate_tess.sh'
            if not sh_name.exists():
                print(f"generate_tess.sh not found: {sh_name}")
                return None

            st = os.stat(sh_name)
            os.chmod(sh_name, st.st_mode | stat.S_IEXEC)

            log_file = save_dir / 'neper.log'
            ok = _run_neper_and_check(sh_name, save_dir, log_file, timeout_s=600)

            def _alpha_ok(dirpath):
                stgroups = glob.glob(str(Path(dirpath) / '*.stgroup'))
                if not stgroups:
                    return True
                try:
                    with open(stgroups[0], 'r') as f_stg:
                        lines = f_stg.readlines()
                    alpha_ratio = float(lines[0].strip())
                    if alpha_ratio < 0.15:
                        print('Alpha phase ratio out of lower limit!')
                        return False
                    if alpha_ratio > 0.95:
                        print('Alpha phase ratio out of upper limit!')
                        return False
                    return True
                except Exception:
                    return True

            def _try_images(dirpath):
                try:
                    return image_gen_3D_BO(1.0, str(Path(dirpath)) + '/')
                except IndexError:
                    print(f"Tessellation failed for {dirpath}")
                    return None
                except Exception as e_img:
                    print(f"image_gen_3D_BO error: {e_img}")
                    return None

            images = None
            if ok and _alpha_ok(save_dir):
                images = _try_images(save_dir)

            if images is None:
                print("[retry] resampling seeds on 1..29 grid and rerunning Neper.")
                rng = np.random.default_rng(2025)
                coords_try = rng.integers(1, 30, size=(n, 3))
                coords_try, _ = _seed_stats_and_repair(coords_try, min_gap=2, rng_seed=2026)
                _write_seeds_file(coords_try, seed_file)

                log_file_retry = save_dir / 'neper_retry.log'
                ok2 = _run_neper_and_check(sh_name, save_dir, log_file_retry, timeout_s=600)
                if ok2 and _alpha_ok(save_dir):
                    images = _try_images(save_dir)

            if images is None:
                print(f"No valid tessellation after retry in {save_dir}. "
                      f"Check {log_file} / {save_dir / 'neper_retry.log'}")
                return None

            return images

        except Exception as e:
            print(f"Microstructure generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_images_from_files(self, tess_file, tesr_file, save_dir):

        raster_size = 32
        n_channels = 5

        with open(tess_file, 'r') as f:
            lines_tess = f.readlines()

        with open(tesr_file, 'r') as f:
            lines_tesr = f.readlines()

        cellids_list = []
        ori_list = []

        for line_number, line in enumerate(lines_tess):
            if '**cell' in line:
                num_cells = int(lines_tess[line_number + 1])

            if '*lam' in line:
                lam_line_start = line_number + 1
            if '*group' in line:
                lam_line_end = line_number
                lam_list = []
                for i in range(lam_line_start, lam_line_end):
                    lams_str = lines_tess[i]
                    for lam_str in lams_str.split():
                        lam_int = int(lam_str) - 1
                        lam_list.append(lam_int)

            if '*ori' in line:
                for ori in range(line_number + 2, line_number + 2 + num_cells):
                    r1 = float(lines_tess[ori][2:17])
                    r2 = float(lines_tess[ori][20:35])
                    r3 = float(lines_tess[ori][38:53])
                    r = [r1, r2, r3]
                    ori_list.append(r)

        for line_number, line in enumerate(lines_tesr):
            if 'ascii\n' in line:
                for i in range(line_number + 1, len(lines_tesr) - 1):
                    cellids_str = lines_tesr[i]
                    for cellid_str in cellids_str.split():
                        cellid_int = int(cellid_str)
                        cellids_list.append(cellid_int)

        image = np.zeros((raster_size, raster_size, raster_size, n_channels))
        counter = 0
        while counter < len(cellids_list):
            for z in range(0, raster_size):
                for y in range(0, raster_size):
                    for x in range(0, raster_size):
                        image[x, y, z, 0] = cellids_list[counter]
                        image[x, y, z, 1] = lam_list[cellids_list[counter] - 1]
                        image[x, y, z, 2:n_channels] = ori_list[cellids_list[counter] - 1]
                        counter = counter + 1

        return [image[:, :, :, 1:5]]

    def _calculate_kt(self, pred):

        pred = DeNormalizeData(pred, self.data_min, self.data_max)
        idx_max = np.unravel_index(np.argmax(pred), pred.shape)
        stress_max = pred[idx_max]
        stress_maxslice = pred[:, idx_max[1], :, :, :]
        stress_nom = np.average(stress_maxslice.flatten())
        return stress_max / stress_nom

    def compute_reward(self, E, sigma_y, Kt, prev_error_sum=None, best_error_sum=None):
        if E is None or sigma_y is None or Kt is None:
            return -5.0

        errors = self.compute_error_components(E, sigma_y, Kt)
        E_err = errors["E_error"]
        Sy_err = errors["sigma_y_error"]
        Kt_err = errors["Kt_error"]
        err_sum = errors["error_sum"]

        def score(e):
            return 1.0 / (1.0 + 5.0 * e)

        wE, wSy, wKt = 1.0, 1.0, 2.0
        base = (wE * score(E_err) + wSy * score(Sy_err) + wKt * score(Kt_err)) / (wE + wSy + wKt)

        bonus = 0.0
        for threshold, gain in ((0.15, 0.20), (0.10, 0.30), (0.07, 0.45)):
            bonus += gain * float(E_err < threshold)
            bonus += gain * float(Sy_err < threshold)
            bonus += gain * float(Kt_err < threshold)

        if prev_error_sum is not None:
            improvement = float(prev_error_sum) - err_sum
            bonus += 3.0 * np.clip(improvement, -0.20, 0.20)
        if best_error_sum is not None:
            best_improvement = float(best_error_sum) - err_sum
            if best_improvement > 0:
                bonus += 0.75 + 5.0 * min(best_improvement, 0.20)

        if (E_err < self.target.tolerance['E'] and
            Sy_err < self.target.tolerance['sigma_y'] and
            Kt_err < self.target.tolerance['Kt']):
            bonus += 5.0

        reward = base + bonus
        return float(reward)

    def get_state(self, E, sigma_y, Kt):

        if E is None:
            E = 100000
        if sigma_y is None:
            sigma_y = 900
        if Kt is None:
            Kt = 1.5

        state = np.array([
            E / 120000,
            sigma_y / 1200,
            Kt / 2.0,
            self.target.E_target / 120000,
            self.target.sigma_y_target / 1200,
            self.target.Kt_target / 2.0,
            (E - self.target.E_target) / 120000,
            (sigma_y - self.target.sigma_y_target) / 1200,
            (Kt - self.target.Kt_target) / 2.0
        ])
        return state
