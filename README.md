# VideoUpsampler

## MP4 video upsampler
### Video
Paired video with lower resolution was used for training

Using `ffmpeg` can be generated downsampled video

##### Example from 1080p with x2 downsampled scale:
```console
ffmpeg -i input/mp4/video -s 960x540 -crf 30 -vsync vfr output/mp4/video
```

### Model
Model based on UNet architecture (more in `tf_model.py`)

Trained on MP4 encoded videos with lower resolution

[Pretrained x2 TF model](https://drive.google.com/file/d/1NlSXLaTOdkk41eQEy10R3Bag0WxsTqnq/view?usp=sharing)

### Usage
```console
foo@bar:~/VideoUpsampler$ python upsampler.py --file path/to/mp4/file --model path/to/model 
```
New video will be saved near original file with `_upsampled` suffix

Also suffix `_audio` will be added for video with copied audio

### YouTube example
<a href="http://www.youtube.com/watch?feature=player_embedded&v=
7XsLhcHGr0s" target="_blank"><img src="http://img.youtube.com/vi/7XsLhcHGr0s/0.jpg"/></a>
