# UNet

## Setup environment

```shell
docker build . -t cuda
docker run -it -v /abs/path/to/semseg-grip-image:/code --gpus all cuda bash
```
## Download Pascal VOC

```shell
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
```
