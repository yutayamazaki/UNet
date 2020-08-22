# semantic segmentation pytorch

![GitHub Actions](https://github.com/yutayamazaki/semantic-segmentation-pytorch/workflows/build/badge.svg)

## Setup environment

```shell
docker build . -t cuda
docker run -it -v /abs/path/to/semantic-segmentation-pytorch:/code --gpus all cuda bash
```
## Download Pascal VOC

```shell
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
```

## Directories

- experiments: A directory to save experimental results.
- src: Python modules and scripts to run semantic segmentation.
- tests: Unittests.
- VOCdevkit: Datset directory.

