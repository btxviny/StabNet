# Deep Online Video Stabilization With Multi-Grid Warping Transformation Learning

This is a PyTorch implementation of the [paper](https://cg.cs.tsinghua.edu.cn/papers/TIP-2019-VideoStab.pdf).

![Video Stabilization Example](https://github.com/btxviny/Deep-Motion-Blind-Video-Stabilization/blob/main/result.gif).

I provide the original online algorithm described in the paper and a second implementation using a buffer of future frames. The latter can no longer be categorized as an online algorithm but it achieves better stabilization results

## Inference Instructions

Follow these instructions to perform video stabilization using the pretrained model:

1. **Download the pretrained models:**
   - Download the pretrained models [weights](https://drive.google.com/drive/folders/1K8HfenNEr_0Joi6RdX4SfKVnCg-GjhvW?usp=drive_link).
   - Place the downloaded weights folder in the main folder of your project.

2. **Run the Stabilization Script:**
   - For the original model run:
     ```bash
     python stabilize_online.py --in_path unstable_video_path --out_path result_path
     ```
   - Replace `unstable_video_path` with the path to your input unstable video.
   - Replace `result_path` with the desired path for the stabilized output video.
   - For the second model with future frames:
     ```bash
     python stabilize_future_frames.py --in_path unstable_video_path --out_path result_path
     ```

Make sure you have the necessary dependencies installed, and that your environment is set up correctly before running the stabilization scripts.
```bash
     pip install numpy opencv-python torch==2.1.2 matplotlib
```



## Training Instructions

Follow these instructions to train the model:

1. Download Datasets:
   - Download the training dataset:
     - [DeepStab Modded](https://hyu-my.sharepoint.com/personal/kashifali_hanyang_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fkashifali%5Fhanyang%5Fac%5Fkr%2FDocuments%2FDeepStab%5FMod%2Erar&parent=%2Fpersonal%2Fkashifali%5Fhanyang%5Fac%5Fkr%2FDocuments&ga=1)

   - Extract the contents of the downloaded dataset to a location on your machine.

2. Create the optical flow dataset and matched feature dataset used for the loss functions described in the paper as I demonstrate in the notebooks [Flows](https://github.com/btxviny/StabNet/blob/main/Flows_dataset.ipynb),
[matched features](https://github.com/btxviny/StabNet/blob/main/matched_features_dataset.ipynb)

3. I then provide notebooks for training the two different implementations/:
   Online version: [train_vgg19_16x16_online.ipynb](https://github.com/btxviny/StabNet/blob/main/train_vgg19_16x16_online.ipynb)
   Future frame version: [train_vgg19_16x16_future_frames.ipynb](https://github.com/btxviny/StabNet/edit/main/train_vgg19_16x16_future_frames.ipynb)
   Make sure to change ckpt_dir to the destination you want the model checkpoints to be saved at.

5. I provide metrics.py which computes the cropping, distortion and stability scores for the generated results.
