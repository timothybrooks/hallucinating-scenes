# Hallucinating Pose-Compatible Scenes
Official code release.

### Install dependencies

We use Anaconda to manage our python environment. Please install Anaconda and run the commands below to install dependencies and activate the environment:
```
conda env create -f environment.yml -n hallucinating-scenes
conda activate hallucinating-scenes
```

### Download pretrained models

We used pretrained Inception, VGG and OpenPose networks for evaluation and dataset construction. We also provide a pretrained checkpoint of our final model. Please download all of these checkpoints [here](https://drive.google.com/drive/folders/1VsqGT9eEedW97HVN6OEwLT77B3fB_Wj2?usp=sharing). Place `inception-2015-12-05.pt` and `vgg16.pt` under `metrics/pretrained/`, `open_pose.pt` under `open_pose/pretrained/`, and `ours.ckpt` under `checkpoints/ours/`.

| fileame                | folder               |
| :---:                  | :---:                |
| inception-2015-12-05.pt| metrics/pretrained/  |
| vgg16.pt               | metrics/pretrained/  |
| open_pose.pt           | open_pose/pretrained/|
| ours.ckpt              | checkpoints/ours/    |

### Construct dataset

Please see the dataset page [here](https://github.com/timothybrooks/hallucinating-scenes/blob/master/dataset.md) for instructions to download the 10 source datasets and construct our meta-dataset. The dataset (or some subset of it) is necessary to train models as well as to provide input poses for generating images and evaluating the model.

### Generate images

See the file `generate.ipynb` for example code to generate images using our pretrained model. Note that both the pretrained model checkpoint and the dataset must be downloaded to generate images. 

### Train/evaluate model

You can train a new model by running `python run_train.py`. We use Hydra to manage our experiment configurations. You can modify the config in `configs/run_train/ours.yaml` or by overriding parameters on the command line.

You can evaluate models by running `python run_metrics.py MODEL_ID --dataset PATH_TO_DATASET`.

To evaluate our pretrained model, set `MODEL_ID` to `ours`.

### Citation

```
@inproceedings{brooks2021hallucinating,
    title={Hallucinating Pose-Compatible Scenes},
    author={Brooks, Tim and Efros, Alexei A},
    booktitle=ECCV,
    year={2022}
}
```


