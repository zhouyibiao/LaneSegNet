cd /data/yibiao.zhou/LaneSegNet-Adaption/OpenLane-V2
pip install -e .
cd /data/yibiao.zhou/LaneSegNet-Adaption/mmsegmentation-v0.29.1
pip install -e .
cd /data/yibiao.zhou/LaneSegNet-Adaption/mmdetect-v2.26.0
pip install -e .
cd /data/yibiao.zhou/LaneSegNet-Adaption/mmdet3d-v1.0.0rc6
pip install -e .
cd /data/yibiao.zhou/LaneSegNet-Adaption/mmcv-v1.5.2
FORCE_MUSA=1 pip install -e .


pip install networkx==3.4.2
pip install protobuf==3.19.6
pip install yapf==0.31.0


cd /data/yibiao.zhou
ln -s /home/torch_musa/ ./torch_musa_wkdir
ln -s /home/pytorch/ ./pytorch