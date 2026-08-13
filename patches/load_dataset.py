# =====================================================================
# Visual Place Recognition: A Tutorial - dataset loaders (SFU augmented)
# Adds support for extra SFU query images in images/SFU/jan by expanding GT.
# =====================================================================

import os
import re as _re
import urllib.request
import zipfile
from glob import glob
from PIL import Image
import numpy as np
from scipy.signal import convolve2d
from abc import ABC, abstractmethod


class Dataset(ABC):
    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def download(self, destination: str):
        raise NotImplementedError


class GardensPointDataset(Dataset):
    def __init__(self, destination: str = 'images/GardensPoint/'):
        self.destination = destination

    def load(self):
        print('===== Load dataset GardensPoint day_right--night_right')
        if not os.path.exists(self.destination):
            self.download(self.destination)

        fns_db = sorted(glob(os.path.join(self.destination, 'day_right', '*.jpg')))
        fns_q = sorted(glob(os.path.join(self.destination, 'night_right', '*.jpg')))

        imgs_db = [np.array(Image.open(fn).convert("RGB")) for fn in fns_db]
        imgs_q  = [np.array(Image.open(fn).convert("RGB")) for fn in fns_q]

        nref = len(imgs_db)
        nqry = len(imgs_q)

        # Base GT: identity mapping for the original queries (Image000..Image199)
        GThard = np.zeros((nref, nqry), dtype=bool)
        for i in range(min(nref, nqry)):
            GThard[i, i] = True

        # GT mapping for extra queries Image200..Image249
        # Each entry: query_index -> reference_index (both zero-based)
        extra_query_gt = {
            200: 1,   201: 6,   202: 7,   203: 8,   204: 11,
            205: 22,  206: 23,  207: 24,  208: 26,  209: 28,
            210: 35,  211: 39,  212: 40,  213: 50,  214: 55,
            215: 56,  216: 57,  217: 59,  218: 62,  219: 67,
            220: 70,  221: 71,  222: 86,  223: 87,  224: 88,
            225: 91,  226: 97,  227: 107, 228: 108, 229: 114,
            230: 117, 231: 129, 232: 137, 233: 139, 234: 143,
            235: 150, 236: 151, 237: 154, 238: 163, 239: 166,
            240: 168, 241: 173, 242: 181, 243: 182, 244: 185,
            245: 188, 246: 189, 247: 190, 248: 191, 249: 197,
        }


        q_basenames = [os.path.basename(f) for f in fns_q]
        for q_num, ref_idx in extra_query_gt.items():
            q_name = f'Image{q_num:03d}.jpg'
            if q_name in q_basenames:
                qi = q_basenames.index(q_name)
                GThard[:, qi] = False
                GThard[ref_idx, qi] = True

        GTsoft = convolve2d(GThard.astype(int), np.ones((17, 1), int), mode='same').astype(bool)
        return imgs_db, imgs_q, GThard, GTsoft

    def download(self, destination):
        fn = 'GardensPoint_Walking.zip'
        url = 'https://www.tu-chemnitz.de/etit/proaut/datasets/' + fn
        os.makedirs(destination, exist_ok=True)
        urllib.request.urlretrieve(url, os.path.join(destination, fn))
        with zipfile.ZipFile(os.path.join(destination, fn), 'r') as z:
            z.extractall(destination)
        os.remove(os.path.join(destination, fn))


class StLuciaDataset(Dataset):
    def __init__(self, destination: str = 'images/StLucia_small/'):
        self.destination = destination

    def load(self):
        print('===== Load dataset StLucia 100909_0845--180809_1545 (small version)')
        if not os.path.exists(self.destination):
            self.download(self.destination)

        fns_db = sorted(glob(os.path.join(self.destination, '100909_0845', '*.jpg')))
        fns_q  = sorted(glob(os.path.join(self.destination, '180809_1545', '*.jpg')))

        imgs_db = [np.array(Image.open(fn).convert("RGB")) for fn in fns_db]
        imgs_q  = [np.array(Image.open(fn).convert("RGB")) for fn in fns_q]

        gt = np.load(os.path.join(self.destination, 'GT.npz'), allow_pickle=True)
        nref = len(imgs_db)
        nqry = len(imgs_q)
        GThard0 = gt['GThard'].astype(bool)
        base_n  = GThard0.shape[1]

        if nqry <= base_n:
            return imgs_db, imgs_q, GThard0[:, :nqry], gt['GTsoft'].astype(bool)[:, :nqry]

        # Extra queries injected (HAGVA augmentation): extend GT
        GThard = np.zeros((nref, nqry), dtype=bool)
        GThard[:, :base_n] = GThard0
        q_base = [os.path.basename(p) for p in fns_q]
        for new_idx in range(base_n, nqry):
            fname = q_base[new_idx]
            if not fname.startswith('_hagva_tmp_'):
                continue
            inner = fname[len('_hagva_tmp_'):]
            m = _re.search(r'_ele[\d.]+_r[\d.]+', inner)
            stem = _re.sub(r'^\d+_', '', inner[:m.start()]) if m else None
            if stem and (stem + '.jpg') in q_base:
                src_idx = q_base.index(stem + '.jpg')
                if src_idx < base_n:
                    GThard[:, new_idx] = GThard0[:, src_idx]

        GTsoft = convolve2d(GThard.astype(int), np.ones((17, 1), int), mode='same').astype(bool)
        return imgs_db, imgs_q, GThard, GTsoft

    def download(self, destination):
        fn = 'StLucia_small.zip'
        url = 'https://www.tu-chemnitz.de/etit/proaut/datasets/' + fn
        os.makedirs(destination, exist_ok=True)
        urllib.request.urlretrieve(url, os.path.join(destination, fn))
        with zipfile.ZipFile(os.path.join(destination, fn), 'r') as z:
            z.extractall(destination)
        os.remove(os.path.join(destination, fn))


class SFUDataset(Dataset):
    def __init__(self, destination: str = 'images/SFU/'):
        self.destination = destination

    def load(self):
        print('===== Load dataset SFU dry--jan')
        if not os.path.exists(self.destination):
            self.download(self.destination)

        # dry = reference, jan = query (as used by tutorial)
        fns_db = sorted(glob(os.path.join(self.destination, 'dry', '*.jpg')))
        fns_q  = sorted(glob(os.path.join(self.destination, 'jan', '*.jpg')))

        imgs_db = [np.array(Image.open(fn).convert("RGB")) for fn in fns_db]
        imgs_q  = [np.array(Image.open(fn).convert("RGB")) for fn in fns_q]

        gt_path = os.path.join(self.destination, 'GT.npz')
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(gt_path)

        gt_data = np.load(gt_path, allow_pickle=True)
        GThard0 = gt_data['GThard'].astype(bool)
        GTsoft0 = gt_data['GTsoft'].astype(bool)

        nref = len(imgs_db)
        nqry = len(imgs_q)

        if GThard0.shape[0] != nref:
            raise ValueError(f"SFU GT rows {GThard0.shape[0]} != #ref images {nref}. Wrong GT.npz?")

        q_base = [os.path.basename(p) for p in fns_q]

        # Overwrite mapping for newly added queries (b147..b196)
        copy_from = {
            'b147.jpg': 'a004.jpg',
            'b148.jpg': 'a013.jpg',
            'b149.jpg': 'a014.jpg',
            'b150.jpg': 'a016.jpg',
            'b151.jpg': 'a017.jpg',
            'b152.jpg': 'a023.jpg',
            'b153.jpg': 'a045.jpg',
            'b154.jpg': 'a048.jpg',
            'b155.jpg': 'a050.jpg',
            'b156.jpg': 'a053.jpg',
            'b157.jpg': 'a058.jpg',
            'b158.jpg': 'a072.jpg',
            'b159.jpg': 'a080.jpg',
            'b160.jpg': 'a082.jpg',
            'b161.jpg': 'a102.jpg',
            'b162.jpg': 'a111.jpg',
            'b163.jpg': 'a112.jpg',
            'b164.jpg': 'a113.jpg',
            'b165.jpg': 'a115.jpg',
            'b166.jpg': 'a120.jpg',
            'b167.jpg': 'a126.jpg',
            'b168.jpg': 'a136.jpg',
            'b169.jpg': 'a141.jpg',
            'b170.jpg': 'a143.jpg',
            'b171.jpg': 'a173.jpg',
            'b172.jpg': 'a175.jpg',
            'b173.jpg': 'a177.jpg',
            'b174.jpg': 'a184.jpg',
            'b175.jpg': 'a195.jpg',
            'b176.jpg': 'a215.jpg',
            'b177.jpg': 'a217.jpg',
            'b178.jpg': 'a230.jpg',
            'b179.jpg': 'a236.jpg',
            'b180.jpg': 'b020.jpg',
            'b181.jpg': 'b036.jpg',
            'b182.jpg': 'b041.jpg',
            'b183.jpg': 'b049.jpg',
            'b184.jpg': 'b063.jpg',
            'b185.jpg': 'b064.jpg',
            'b186.jpg': 'b070.jpg',
            'b187.jpg': 'b071.jpg',
            'b188.jpg': 'b089.jpg',
            'b189.jpg': 'b094.jpg',
            'b190.jpg': 'b108.jpg',
            'b191.jpg': 'b119.jpg',
            'b192.jpg': 'b121.jpg',
            'b193.jpg': 'b128.jpg',
            'b194.jpg': 'b135.jpg',
            'b195.jpg': 'b139.jpg',
            'b196.jpg': 'b141.jpg',
        }

        # Resize hard GT to match current number of queries (expand or truncate)
        if GThard0.shape[1] != nqry:
            GThard = np.zeros((nref, nqry), dtype=bool)
            common = min(GThard0.shape[1], nqry)
            GThard[:, :common] = GThard0[:, :common]
        else:
            GThard = GThard0.copy()

        # Overwrite mapped columns (even if they already existed)
        for new_q, old_q in copy_from.items():
            if new_q not in q_base:
                continue
            if old_q not in q_base:
                raise FileNotFoundError(
                    f"Expected existing query '{old_q}' not found in images/SFU/jan/. Cannot copy its GT for '{new_q}'."
                )
            new_idx = q_base.index(new_q)
            old_idx = q_base.index(old_q)

            if old_idx >= GThard0.shape[1]:
                raise ValueError(
                    f"'{old_q}' is at index {old_idx} in jan/, but base GT has only {GThard0.shape[1]} columns. "
                    f"GT.npz does not correspond to current sorted jan/ list."
                )

            GThard[:, new_idx] = GThard0[:, old_idx]

        # Handle HAGVA injected queries (_hagva_tmp_*): parse source stem and copy GT
        for new_idx in range(len(q_base)):
            fname = q_base[new_idx]
            if not fname.startswith('_hagva_tmp_'):
                continue
            inner = fname[len('_hagva_tmp_'):]
            m = _re.search(r'_ele[\d.]+_r[\d.]+', inner)
            stem = _re.sub(r'^\d+_', '', inner[:m.start()]) if m else None
            if stem and (stem + '.jpg') in q_base:
                src_idx = q_base.index(stem + '.jpg')
                if src_idx < GThard0.shape[1]:
                    GThard[:, new_idx] = GThard0[:, src_idx]

        GTsoft = convolve2d(GThard.astype(int), np.ones((17, 1), int), mode='same').astype(bool)

        # Backup original GT once and overwrite GT.npz (so future runs are consistent)
        backup = os.path.join(self.destination, 'GT_original.npz')
        if not os.path.isfile(backup):
            np.savez_compressed(backup, GThard=gt_data['GThard'], GTsoft=gt_data['GTsoft'])
        np.savez_compressed(gt_path, GThard=GThard, GTsoft=GTsoft)

        return imgs_db, imgs_q, GThard, GTsoft
    def download(self, destination):
        fn = 'SFU.zip'

        url = 'https://www.tu-chemnitz.de/etit/proaut/datasets/' + fn
        os.makedirs(destination, exist_ok=True)
        urllib.request.urlretrieve(url, os.path.join(destination, fn))
        with zipfile.ZipFile(os.path.join(destination, fn), 'r') as z:
            z.extractall(destination)
        os.remove(os.path.join(destination, fn))


class CorridorDataset(Dataset):
    def __init__(self, destination: str = 'images/Corridor/'):
        self.destination = destination

    def load(self):
        print('===== Load dataset Corridor ref--query')

        ref_dir = os.path.join(self.destination, 'ref')
        qry_dir = os.path.join(self.destination, 'query')
        gt_path = os.path.join(self.destination, 'ground_truth_new.npy')

        if not os.path.isdir(ref_dir):
            raise FileNotFoundError(ref_dir)
        if not os.path.isdir(qry_dir):
            raise FileNotFoundError(qry_dir)
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(gt_path)

        def _stem_int(p):
            stem = os.path.splitext(os.path.basename(p))[0]
            # Handle injected files named z_210, z_211 etc
            if stem.startswith('z_'):
                return 100000 + int(stem[2:])
            return int(stem)

        fns_db = sorted(glob(os.path.join(ref_dir, '*.jpg')), key=_stem_int)
        fns_q  = sorted(glob(os.path.join(qry_dir, '*.jpg')), key=_stem_int)

        imgs_db = [np.array(Image.open(fn).convert("RGB")) for fn in fns_db]
        imgs_q  = [np.array(Image.open(fn).convert("RGB")) for fn in fns_q]

        nref = len(imgs_db)
        nqry = len(imgs_q)

        gt = np.load(gt_path, allow_pickle=True)

        # Build GThard: the stored window lists are the hard GT positives
        GThard = np.zeros((nref, nqry), dtype=bool)
        if isinstance(gt, np.ndarray) and gt.ndim == 2 and gt.shape[1] == 2 and gt.dtype != object:
            # Format A: Nx2 int pairs [ref_idx, q_idx]
            for ref_idx, q_idx in gt.astype(int):
                if 0 <= ref_idx < nref and 0 <= q_idx < nqry:
                    GThard[ref_idx, q_idx] = True
        else:
            # Format B: object rows [q_idx, refs_list]
            for row in gt:
                q_idx = int(row[0])
                if q_idx >= nqry:
                    continue
                refs = row[1]
                ref_list = [int(refs)] if isinstance(refs, (int, np.integer)) else [int(r) for r in list(refs)]
                for ref_idx in ref_list:
                    if 0 <= ref_idx < nref:
                        GThard[ref_idx, q_idx] = True

        # GTsoft: blur GThard with a ±8 (17-row) window
        GTsoft = convolve2d(GThard.astype(int), np.ones((17, 1), int), mode='same').astype(bool)

        return imgs_db, imgs_q, GThard, GTsoft

    def download(self, destination):
        raise NotImplementedError('Corridor dataset must be provided manually.')


class ESSEX3IN1Dataset(Dataset):
    def __init__(self, destination: str = 'images/ESSEX3IN1/'):
        self.destination = destination

    def load(self):
        print('===== Load dataset ESSEX3IN1 ref--query')

        ref_dir = os.path.join(self.destination, 'ref')
        qry_dir = os.path.join(self.destination, 'query')
        gt_path = os.path.join(self.destination, 'GT.npz')

        if not os.path.isdir(ref_dir):
            raise FileNotFoundError(ref_dir)
        if not os.path.isdir(qry_dir):
            raise FileNotFoundError(qry_dir)
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(gt_path)

        def _stem_int(p):
            stem = os.path.splitext(os.path.basename(p))[0]
            # Handle injected files named z_210, z_211 etc
            if stem.startswith('z_'):
                return 100000 + int(stem[2:])
            return int(stem)

        fns_db = sorted(glob(os.path.join(ref_dir,  '*.jpg')), key=_stem_int)
        fns_q  = sorted(glob(os.path.join(qry_dir, '*.jpg')), key=_stem_int)

        imgs_db = [np.array(Image.open(fn).convert("RGB")) for fn in fns_db]
        imgs_q  = [np.array(Image.open(fn).convert("RGB")) for fn in fns_q]

        gt = np.load(gt_path, allow_pickle=True)
        GThard0 = gt['GThard'].astype(bool)
        GTsoft0 = gt['GTsoft'].astype(bool)

        nref = len(imgs_db)
        nqry = len(imgs_q)

        if GThard0.shape[0] != nref:
            raise ValueError(f"ESSEX3IN1 GT rows {GThard0.shape[0]} != #ref images {nref}. Wrong GT.npz?")

        if GThard0.shape[1] == nqry:
            return imgs_db, imgs_q, GThard0, GTsoft0

        if GThard0.shape[1] > nqry:
            return imgs_db, imgs_q, GThard0[:, :nqry], GTsoft0[:, :nqry]

        # Expand GT to cover new query images (210..259)
        copy_from = {
            210: 1,   211: 103, 212: 104, 213: 105, 214: 118, 215: 119,
            216: 12,  217: 121, 218: 123, 219: 13,  220: 133, 221: 134,
            222: 143, 223: 148, 224: 149, 225: 15,  226: 151, 227: 154,
            228: 161, 229: 162, 230: 176, 231: 177,  232: 178, 233: 180,
            234: 186, 235: 195, 236: 196, 237: 200,  238: 26,  239: 35,
            240: 39,  241: 45,  242: 46,  243: 49,   244: 57,  245: 6,
            246: 66,  247: 68,  248: 71,  249: 75,   250: 79,  251: 8,
            252: 80,  253: 81,  254: 82,  255: 85,   256: 88,  257: 90,
            258: 96,  259: 97,
        }
        q_stems = [int(os.path.splitext(os.path.basename(p))[0]) for p in fns_q]
        stem_to_qi = {s: i for i, s in enumerate(q_stems)}

        GThard = np.zeros((nref, nqry), dtype=bool)
        base_cols = GThard0.shape[1]
        for stem in range(base_cols):
            if stem in stem_to_qi:
                GThard[:, stem_to_qi[stem]] = GThard0[:, stem]

        for new_stem, src_stem in copy_from.items():
            if new_stem not in stem_to_qi:
                continue
            if src_stem < 0 or src_stem >= base_cols:
                raise ValueError(f"ESSEX3IN1 source stem {src_stem} out of range for base cols {base_cols}")
            GThard[:, stem_to_qi[new_stem]] = GThard0[:, src_stem]

        GTsoft = convolve2d(GThard.astype(int), np.ones((17, 1), int), mode='same').astype(bool)

        backup = os.path.join(self.destination, 'GT_original.npz')
        if not os.path.isfile(backup):
            np.savez_compressed(backup, GThard=GThard0, GTsoft=GTsoft0)
        np.savez_compressed(gt_path, GThard=GThard, GTsoft=GTsoft)

        return imgs_db, imgs_q, GThard, GTsoft

    def download(self, destination):
        raise NotImplementedError('ESSEX3IN1 is a local dataset.')
