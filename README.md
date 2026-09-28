# Neural Plagiarism — ICCV 2025

Official implementation of **Attention to Neural Plagiarism: Diffusion Models Can Plagiarize Your Copyrighted Images!**

**[Zihang Zou](https://scholar.google.com/citations?user=GLuGAK0AAAAJ&hl=en), Boqing Gong, and Liqiang Wang** · ICCV 2025 · pp. 19546–19556

[Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Zou_Attention_to_Neural_Plagiarism_Diffusion_Models_Can_Plagiarize_Your_Copyrighted_ICCV_2025_paper.html) · [Open-access PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Zou_Attention_to_Neural_Plagiarism_Diffusion_Models_Can_Plagiarize_Your_Copyrighted_ICCV_2025_paper.pdf) · [DOI](https://doi.org/10.1109/ICCV51701.2025.01817) · [Poster](images/iccv25_poster_neural_plagiarism.png) · [BibTeX](citation.bib)

## Research overview

How robust are image copyright protections when images are processed by diffusion models? This work studies **neural plagiarism**: producing semantically similar versions of copyrighted images that evade visible or invisible copyright markers.

The method uses inverse latents as anchors and optimized perturbations as shims. Perturbing cross-attention at selected diffusion timesteps changes the image's semantic content to different degrees. The approach uses gradient-based optimization without additional model training or fine-tuning. The paper evaluates the method on MS-COCO and real-world copyrighted images.

This work is relevant to research on diffusion models, image copyright protection, watermark robustness, and evaluation of generative AI. See the paper for the threat model, experimental settings, results, and limitations.

![Anchor-and-shim pipeline for studying copyright-marker robustness in diffusion models](images/attack_pipeline.png)

## Code and experiments

| File | Purpose |
| --- | --- |
| [run_attack.py](run_attack.py) | Experiment entry point and command-line arguments |
| [attack_stable_diffusion.py](attack_stable_diffusion.py) | Attack pipeline |
| [inverse_stable_diffusion.py](inverse_stable_diffusion.py) | Diffusion inversion pipeline |
| [modified_stable_diffusion.py](modified_stable_diffusion.py) | Modified diffusion components |
| [requirements.txt](requirements.txt) | Recorded Python dependencies |
| [samples](samples) | Example input image |

Review the pinned dependencies and your PyTorch/CUDA environment before installation. After dependencies are installed, inspect the available arguments with:

```bash
python run_attack.py --help
```

The current entry point is `run_attack.py`. Older README examples referenced `optimize_latent_images_folder.py` and a `--noisy_start` flag; neither is present in this checkout. Consult the current argument definitions and the paper when configuring an experiment. This documentation update does not establish a fully validated reproduction environment.

![Example images from the neural plagiarism experiment](images/elon100.jpg)

## Citation

If you build on the method, use the code, or discuss the findings, please cite the published ICCV paper:

```bibtex
@inproceedings{Zou_2025_ICCV,
  author = {Zou, Zihang and Gong, Boqing and Wang, Liqiang},
  title = {Attention to Neural Plagiarism: Diffusion Models Can Plagiarize Your Copyrighted Images!},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year = {2025},
  pages = {19546--19556},
  doi = {10.1109/ICCV51701.2025.01817},
  url = {https://openaccess.thecvf.com/content/ICCV2025/html/Zou_Attention_to_Neural_Plagiarism_Diffusion_Models_Can_Plagiarize_Your_Copyrighted_ICCV_2025_paper.html}
}
```

Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff), with the paper as the preferred citation, and in [citation.bib](citation.bib).

## Related research

[Anti-Neuron Watermarking (ECCV 2022)](https://github.com/zzzucf/anti-neuron-watermarking) studies verification of unauthorized use of personal images in neural network training. Neural Plagiarism studies the robustness of image copyright markers under diffusion-based transformations.

[Zihang Zou on Google Scholar](https://scholar.google.com/citations?user=GLuGAK0AAAAJ&hl=en) · [Research profile](https://github.com/zzzucf)
