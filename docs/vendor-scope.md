# Vendored inference code

Model checkpoints are downloaded separately and are not committed. Use the root
[installation instructions](../README.md#installation) to prepare weights and
build optional CUDA extensions.

## SAM2

`sparc/detectors/sam2/` contains the image-prediction inference package, model
configurations, CUDA source, packaging metadata, checkpoint downloader, upstream
installation and release notes, README, and licenses. The pipeline uses SAM2
image segmentation.

## AllTracker

`sparc/detectors/alltracker/` contains `nets/{alltracker,blocks}.py`,
`utils/misc.py`, their package initializers, and the upstream README and license.
These are the local modules required by the inference network. The MLP mixer
includes its required `einops.layers.torch.Reduce` import. The pipeline's
`load_alltracker_raw_model()` downloads the checkpoint.

## RobotSeg

`setup/prepare_robotseg.py` installs a pinned upstream checkout under
`vendors/robotseg/RobotSeg`, applies the NumPy frame-input adapter, removes the
upstream prebuilt x86 binary, and installs the supplied checkpoint. This source
and its weights are local assets outside the release archive.

The upstream READMEs describe their full projects, including training, demos,
and evaluation tools beyond this release's inference scope. Follow this
repository's root README for supported setup and usage.
