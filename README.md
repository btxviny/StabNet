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

1. **Download Datasets:**
   - Download the training dataset: [DeepStab](https://cg.cs.tsinghua.edu.cn/people/~miao/stabnet/data.zip).
   - Extract the contents of the downloaded dataset to a location on your machine.

2. **Create Datasets for Loss Functions:**
   - Create the optical flows and matched feature datasets to be used in the loss functions descrined in the paper:
     - [Flows_dataset_raft.ipynb](https://github.com/btxviny/StabNet/blob/main/Flows_dataset_raft.ipynb) for optical flow dataset.
     - [matched_features_dataset.ipynb](https://github.com/btxviny/StabNet/blob/main/matched_features_dataset.ipynb) for matched feature dataset.
     - create a train_list.txt containing the file paths for each sample input, using [create.txt](https://github.com/btxviny/StabNet/blob/main/create_txt.ipynb)(adjust paths as needed).

3. **Training Notebooks:**
   - Online version: [train_vgg19_16x16_online.ipynb](https://github.com/btxviny/StabNet/blob/main/train_vgg19_16x16_online.ipynb)
   - Future frame version: [train_vgg19_16x16_future_frames.ipynb](https://github.com/btxviny/StabNet/edit/main/train_vgg19_16x16_future_frames.ipynb)
   - Make sure to change `ckpt_dir` to the destination you want the model checkpoints to be saved at.

4. **Metrics Calculation:**
   - Use `metrics.py` to compute cropping, distortion, and stability scores for the generated results.
