# Neural Plagiarism
Official Implementation of ICCV 2025 paper _"Attention to Neural Plagiarism: Diffusion Models Can Plagiarize Your Copyrighted Images!"_. 

[[ICCV Page](https://openaccess.thecvf.com/content/ICCV2025/html/Zou_Attention_to_Neural_Plagiarism_Diffusion_Models_Can_Plagiarize_Your_Copyrighted_ICCV_2025_paper.html)][[Poster](https://github.com/zzzucf/Neural-Plagiarism/blob/main/images/iccv25_poster_neural_plagiarism.png)]

# **Citation:**
<pre>
@InProceedings{Zou_2025_ICCV,
    author    = {Zou, Zihang and Gong, Boqing and Wang, Liqiang},
    title     = {Attention to Neural Plagiarism: Diffusion Models Can Plagiarize Your Copyrighted Images!},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {19546-19556}
}
</pre>

# Plagiarism Attack:
Our plagiarism attack is a **general, training-free, untargeted** attack on copyrighted data. It exploits the attention mechanisms underlying advanced neural models to remove copyright markers, ranging from **invisible watermarks** to **visible signatures**.

_Example: plagiarize Elon Musk portrait_

These attacked images resemble Elon Musk, but none of these can be him without complicated facial surgery or genetic engineering :)

<p align="center">
  <img src="https://github.com/zzzucf/Neural-Plagiarism/blob/main/images/elon100.jpg?raw=true" width="50%">
</p>

**Attack pipeline:**
Our attack adds shims to anchor variable to diverge the trajectory of the copyrighted data and introduce different level of semantic alterations to bypass copyright protection. As shown below, we can generate a semantic similar plagiarized image **without** knowing the prompt or watermarking methods of target image. More details can be seen in our [poster](https://github.com/zzzucf/Neural-Plagiarism/blob/main/images/iccv25_poster_neural_plagiarism.png) and [paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Zou_Attention_to_Neural_Plagiarism_Diffusion_Models_Can_Plagiarize_Your_Copyrighted_ICCV_2025_paper.pdf).

<p align="center">
  <img src="https://github.com/zzzucf/Neural-Plagiarism/blob/main/images/attack_pipeline.png?raw=true" width="50%">
</p>





## For visible watermark, add large semantic alteration, 
<pre>
python optimize_latent_images_folder.py --target_folder YOUR_IMAGES_DIR_TO_BE_ATTACKED --gen_seed 0 --start 0 --end 1000 --gpu 0 --start_step 15 --k 25 45 --eps 10 --iters 10 --output_folder YOU_OUTPUT_DIR
</pre>

## For invisible watermark, add small unnoticeable noise,
<pre>
python optimize_latent_images_folder.py --target_folder YOUR_IMAGES_DIR_TO_BE_ATTACKED --gen_seed 0 --start 0 --end 1000 --gpu 0 --start_step 45 --k 47 --eps 10 --iters 10 --noisy_start --output_folder YOU_OUTPUT_DIR
</pre>
