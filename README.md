# Auto_SISH
Contains:
 * Code made during research on "Automating Assessment of Silver-enhanced in situ Hybridization for Evaluation of Cancer Properties"
 * Scripts made as part of the effort to refine the pipeline described in that paper

 If looking for the code as described in the article, look through the [corresponding notebooks](research_notebooks). "sandbox_" notebooks contain most of the tried ideas and may not be well-ordered. pipeline.ipynb contains the working proof-of-concept pipeline as described in the Workflow chapter.

## Requirements
- Java
- Python 3.10
- opencv-python
- numpy
- openslide C lib, as well as openslide for Python
- valis-wsi
- libvips C lib, as well as pyvips

The easiest way to install for development is on a Linux subsystem via Conda; we're working on an easy and repeatable Windows installation, but for now you may get stuck in .dll purgatory.
