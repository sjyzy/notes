## 问题

如下代码可以正常使用

```python
from mmdet.apis import init_detector, inference_detector
import mmrotate

config_file = 'oriented_rcnn_r50_fpn_1x_dota_le90.py'
checkpoint_file = 'oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')
inference_detector(model, 'demo/demo.jpg')
```

如下代码不可以正常使用（删除import mmrotate）

```python
from mmdet.apis import init_detector, inference_detector

config_file = 'oriented_rcnn_r50_fpn_1x_dota_le90.py'
checkpoint_file = 'oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')
inference_detector(model, 'demo/demo.jpg')

#KeyError:  'OrientedRCNN is not in the models registry'
```

代码中并未使用mmrotate，删除import mmrotate，为什么就不能使用了？

## 结果

mmdetection中会检测是否使用下游库，import之后虽然代码中未使用，但是mmdetection中检测到了mmrotate这个下游库。