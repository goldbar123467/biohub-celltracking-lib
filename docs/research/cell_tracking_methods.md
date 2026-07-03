# Cell Tracking Methods

## Ultrack

Ultrack targets 2D/3D cell tracking under segmentation uncertainty. Its docs emphasize out-of-memory intermediate storage and multiple candidate segmentations rather than committing to one mask. For this competition, the relevant lesson is to keep candidate generation broad enough for crowded embryos, then let a graph optimization or linking stage choose a consistent subset.

Sources: https://github.com/royerlab/ultrack, https://royerlab.github.io/ultrack/, https://www.nature.com/articles/s41592-025-02778-0

## Cell Tracking Challenge

The CTC ecosystem is useful for file-format and benchmark discipline, but this competition uses sparse GEFF lineage graphs and a custom edge/division metric. CTC-style mask metrics should not be treated as local validation unless explicitly adapted.

Sources: https://celltrackingchallenge.net/, https://www.nature.com/articles/s41592-023-01879-y

## StarDist

StarDist predicts star-convex shapes and object probabilities, with 3D microscopy support. It is useful as an instance-segmentation reference, but it generally expects dense instance labels, while Biohub training labels are sparse centers/lineage edges.

Sources: https://stardist.net/, https://github.com/stardist/stardist

## Cellpose And Cellpose-SAM

Cellpose is a generalist cellular segmentation approach, and Cellpose-SAM points toward strong prompt/generalization behavior. For Kaggle, any pretrained model must be public/free and packaged as a Kaggle dataset or vendored asset, never downloaded at inference.

Source: https://www.cellpose.org/

## Trackastra

Trackastra links segmented detections using transformer-predicted associations across a temporal window and supports dividing objects. Its key competition relevance is separating image-heavy detection from graph/node association and keeping linking tractable with local candidate windows.

Sources: https://github.com/weigertlab/trackastra, https://arxiv.org/abs/2405.15700

## btrack

btrack builds tracklets with spatial and appearance information, then uses multiple-hypothesis testing and integer programming for global track assembly. It is a useful reference for gap closing, motion priors, and lineage constraints.

Sources: https://btrack.readthedocs.io/, https://github.com/quantumjot/btrack

## Motile, LapTrack, And ILP/LAP

Motile is relevant for global ILP graph optimization. LapTrack is relevant for lighter LAP-style association. In this repo, keep both behind interfaces because Kaggle dependency/runtime constraints may make pure-Python greedy/LAP fallbacks necessary.

Sources: https://github.com/funkelab/motile, https://academic.oup.com/bioinformatics/article/39/1/btac799/6887138

## GEFF And Zarr

GEFF is an exchange format for spatial graph data in life sciences and is not designed as a mutable application database. Zarr is the chunked array storage layer for both images and GEFF-backed arrays. This repo reads Zarr lazily and keeps an internal graph representation for fast experiments.

Sources: https://janelia.figshare.com/articles/code/GEFF_Graph_Exchange_File_Format/31943145, https://github.com/live-image-tracking-tools/geff, https://zarr.readthedocs.io/, https://zarr.dev/

