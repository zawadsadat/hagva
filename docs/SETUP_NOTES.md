# What this repo contains, and what it does not

## Contains
- view_synthesis/ : GenWarp + ZoeDepth novel view generation (run first)
- module_a/   : Signal 1 (geometric) + Signal 2 (DINOv2 semantic) scoring notebook
- module_a/scores/ : precomputed per-view scores; lets you reproduce every
                gate/ablation/Exp5 result WITHOUT running GenWarp or DINOv2
- gate/       : 400-view threshold-sweep evaluation, one script per dataset
- exp5/       : per-image single-view injection correlation
- ablation/   : signal ablation, NetVLAD descriptor ablation, threshold-rule studies
- patches/    : MODIFIED demo.py and datasets/load_dataset.py from the VPR tutorial
                framework. These are required — stock versions will not work.
- tools/      : gt_audit.py verifies every query column has ground truth after
                injection. Run it whenever an injection path changes.
- results/    : RESULTS.md with all reported numbers (per-run CSVs removed;
                see note at the end of RESULTS.md)

## Not contained (user must supply)
- VPR tutorial framework (github.com/stschubert/VPR_Tutorial), with patches/ applied
- Datasets: GardensPoint, SFU, StLucia_small
- GenWarp + ZoeDepth for view synthesis
- Synthesized view folders (garden_400_out_low, sfu_400_out_low, st_400_out_low
  and the 50-view equivalents)
- conda env with patchnetvlad, torch, DINOv2 via torch.hub

## Paths
Every script has an absolute-path config block at the top pointing at
/mnt/sda/Zawad/genwarp/... These must be edited for any other machine.

## Ground-truth handling — read before modifying anything
Injected synthetic views need (a) filenames that sort AFTER all original
queries, and (b) an explicit ground-truth column copied from their source
image. Getting either wrong produces plausible but wrong AUC values.

Three distinct mechanisms are in use:
  GardensPoint : runtime patch writing extra_query_gt[idx] = ref_idx into
                 load_dataset.py; injected files named Image200.jpg onward
  SFU          : runtime patch writing copy_from['<injected>'] = '<source>'
  StLucia      : explicit GT matrix extension written to the staging dir
                 (write_gt), injected files named zzz_inject_NNNN.jpg

The StLucia approach is the only one that does not modify the framework at
runtime and is the pattern to standardise on in future work.

Always run tools/gt_audit.py against a staged, injected dataset after changing
an injection path. Compare all-zero column counts against the ORIGINAL GT —
StLucia legitimately has 35 empty columns before any injection.

## module_a/VPR.ipynb cell map
The notebook accumulated across the project and contains per-dataset duplicates.
Config paths in each cell reflect whichever dataset was run last; edit them.

  cell 0-1   : imports, model loading (ZoeDepth, GenWarp, DINOv2)
  cell 2-4   : Signal 1 / Signal 2 / Combined  (paths last set to StLucia 50-view)

To regenerate scores for a dataset, use the cell 2-4 trio and set
synthesized_dir, source_dir, and the three output dirs under
hagva_module_a/<name>/. The 400-view scores in module_a/scores/ were produced
this way with synthesized_dir pointed at garden_400_out_low / sfu_400_out_low /
st_400_out_low.

