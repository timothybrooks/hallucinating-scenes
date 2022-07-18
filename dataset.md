We introduce the Humans in Context dataset in our work. The dataset is sourced from 10 existing research datasets, heavily filtered to contain 19M frames of humans in everyday environments, and supplemented with pose labels obtained using OpenPose.

The final dataset cannot be released directly due to licensing limitatons of the source datasets. However, below are instructions to download all of the 10 source datasets and construct the full Humans in Context meta-dataset we use in our paper. It is also possible to construct a similar dataset from different sources or a subset of the source datasets.

## Download source datasets

- Holistic Video Understanding (HVU)  
https://holistic-video-understanding.github.io/  
The dataset is released as [YouTube IDs and annotations](https://github.com/holistic-video-understanding/HVU-Dataset). They provide a [download tool](https://github.com/holistic-video-understanding/HVU-Downloader) as well as a [form](https://docs.google.com/forms/d/e/1FAIpQLSdtx0DRzSuchK9TVvn49-DpS4enlgx_r-cQy4N5dcR6lUbKsg/viewform) to request access to the test videos or other missing videos.

- Moments in Time  
http://moments.csail.mit.edu/  
To download the dataset, complete this [form](https://docs.google.com/forms/d/e/1FAIpQLSc0rovlbTCDqJyuJXKLHWtpIX6fiuc1jlAnhT68p86D9NCF9g/viewform) to request access to the videos.

- Kinetics (700-2020)  
https://www.deepmind.com/open-source/kinetics  
The dataset is released as metadata and YouTube IDs. There is no official download tool, but [this](https://github.com/Showmax/kinetics-downloader) or other open source tools can be used.

- Charades  
https://prior.allenai.org/projects/charades  
Videos are available for download directly on the dataset webpage.

- InstaVariety
https://github.com/akanazawa/human_dynamics/blob/master/doc/insta_variety.md  
A script for downloading the dataset videos is provided on the dataset writeup.

- Oops  
https://oops.cs.columbia.edu/data/  
Videos are available for download directly on the dataset webpage.

- MPII Human Pose Dataset (videos)  
http://human-pose.mpi-inf.mpg.de/  
Videos can be downloaded from [this](http://human-pose.mpi-inf.mpg.de/#download) page.

- VLOG (people)  
https://web.eecs.umich.edu/~fouhey/2017/VLOG/  
Videos are available for download directly on the dataset webpage, or as a list of YouTube IDs. We only use the video IDs known to contain people, as listed [here](https://github.com/akanazawa/human_dynamics/blob/master/doc/vlog_people.md).

- PennAction  
http://dreamdragon.github.io/PennAction/  
Videos are available for download directly on the dataset webpage.

- YouTube-VOS  
https://youtube-vos.org/dataset/  
The dataset webpage provides instructions for downloading the dataset as part of their benchmark challange tasks.

## Construct Humans in Context meta-dataset
After downloading the source datasets, the Humans in Context meta-dataset can be constructed by running a sequence of three processing scripts on each source dataset.

### 1. Extract high quality frames.
We filter images and video frames for sufficient resolution and bitrate, and resize so that the short edge is 256 resolution. This step is achieved by running `python data_from_images.py input_dir=INPUT_DIR output_dir=OUTPUT_DIR` when the source dataset consists of directories of frames stored as images, or `python data_from_videos.py input_dir=INPUT_DIR output_dir=OUTPUT_DIR` when the source dataset consists of video files. The output directory should be specific to each source dataset and will be used as the input the next phase of processing. For example:

```
python data_from_videos.py input_dir=kinetics output_dir=kinetics_frames_256
```

### 2. Filter frames with person detection.
Next, we filter clips using a person detection network. Run `python data_filter_people.py input_dir=INPUT_DIR output_dir=OUTPUT_DIR` where the input directory is the output of the previous step, and the output directory is again specific to each source. For example:

```
python data_filter_people.py input_dir=kinetics_frames_256 output_dir=kinetics_people
```

### 3. Filter and label using OpenPose.
Lastly, we filter clips and provide labels by running OpenPose. Make sure to get the OpenPose pretrained model from the main README. Then run `python data_detect_pose.py input_dir=INPUT_DIR output_dir=OUTPUT_DIR` where the input directory is again the output of the previous step. For example:

```
python data_detect_pose.py input_dir=kinetics_people output_dir=kinetics_pose
```

The LMDB outputs for poses, clips and frames should then be arranged to form the following directory structure:

```
charades 
  clips_db 
  frames_db 
  poses_db
hvu  
  clips_db 
  frames_db 
  poses_db
insta_variety  
  clips_db 
  frames_db 
  poses_db
kinetics 
  clips_db 
  frames_db 
  poses_db
moments  
  clips_db 
  frames_db 
  poses_db
mpii  
  clips_db 
  frames_db 
  poses_db
oops   
  clips_db 
  frames_db 
  poses_db
penn_action   
  clips_db 
  frames_db 
  poses_db
vlog_people   
  clips_db 
  frames_db 
  poses_db
youtube_vos 
  clips_db 
  frames_db 
  poses_db
```
