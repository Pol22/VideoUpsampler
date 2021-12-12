# VideoUpsampler

## MP4 video upsampler
### Model
Model based on UNet architecture (more in `tf_model.py`)

Trained on MP4 encoded videos with lower resolution

[Pretrained TF model](https://drive.google.com/file/d/1NlSXLaTOdkk41eQEy10R3Bag0WxsTqnq/view?usp=sharing)

### Usage
```console
foo@bar:~/VideoUpsampler$ python upsampler.py --file path/to/mp4/file --model path/to/model 
```
New video will be saved near original file with `_upsampled` suffix

### YouTube example
<a href="http://www.youtube.com/watch?feature=player_embedded&v=
7XsLhcHGr0s" target="_blank"><img src="http://img.youtube.com/vi/7XsLhcHGr0s/0.jpg"/></a>