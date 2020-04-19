# UNet
## 実行手順

- Pascal VOCデータセットのダウンロード

```shell
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
```

- Python環境のセットアップ

```shell
pip install -r requirements.txt
```

- 学習実行

```shell
python train.py
```
