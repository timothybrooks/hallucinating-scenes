We introduce the Humans in Context dataset in our work. The dataset is sourced from 10 existing research datasets, heavily filtered to contain 19M frames of humans in everyday environments, and supplemented with pose labels obtained using OpenPose.

We release a subset of the Humans in Context dataset available for direct download here. The entire dataset cannot be released directly due to licensing limitatons of the source datasets, however, below are instructions to download all of the 10 source datasets and construct the full Humans in Context meta-dataset we use in our paper.

### 1. Download source datasets

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
