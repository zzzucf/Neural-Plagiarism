# [Attention to Neural Plagiarism: Diffusion Models Can Plagiarize Your Copyrighted Images!](https://openaccess.thecvf.com/content/ICCV2025/html/Zou_Attention_to_Neural_Plagiarism_Diffusion_Models_Can_Plagiarize_Your_Copyrighted_ICCV_2025_paper.html)
Official Implementation of ICCV 2025 paper. [[ICCV Page](https://openaccess.thecvf.com/content/ICCV2025/html/Zou_Attention_to_Neural_Plagiarism_Diffusion_Models_Can_Plagiarize_Your_Copyrighted_ICCV_2025_paper.html)]

_Example: plagiarize Elon Musk portrait_

![100 plagiarized images for Elon Musk](https://github.com/zzzucf/Neural-Plagiarism/blob/main/images/elon100.jpg)

# Plagiarism Attack:
## For visible watemark, add large semantic alteration, 
<pre>
python optimize_latent_images_folder.py --target_folder YOUR_IMAGES_DIR_TO_BE_ATTACKED --gen_seed 0 --start 0 --end 1000 --gpu 0 --start_step 15 --k 25 45 --eps 10 --iters 10 --output_folder YOU_OUTPUT_DIR
</pre>

## For invisible watermark, add small unnoticeable noise,
<pre>
python optimize_latent_images_folder.py --target_folder YOUR_IMAGES_DIR_TO_BE_ATTACKED --gen_seed 0 --start 0 --end 1000 --gpu 0 --start_step 45 --k 47 --eps 10 --iters 10 --noisy_start --output_folder YOU_OUTPUT_DIR
</pre>

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
