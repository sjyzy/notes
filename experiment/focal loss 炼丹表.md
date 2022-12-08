focal loss

|         | alpha=0.25 | alpha=0.5 | alpha=0.75 |
| ------- | ---------- | --------- | ---------- |
| gamma=0 | *0.418*    | *0.446*   | *0.455*    |
| gamma=1 | *0.411*    | *0.452*   | *0.455*    |
| gamma=2 | *0.406*    | *0.441*   | *0.455*    |
| gamma=3 | *0.402*    | *0.43*    | *0.439*    |
| gamma=4 | *0.392*    | *0.418*   | *0.43*     |
| gamma=5 | *0.37*     | *0.408*   | *0.424*    |

dyhead

python tools/train.py configs/dyhead/atss_swin-l-p4-w12_fpn_dyhead_mstrain_24e_logo.py

OrderedDict([('bbox_mAP', 0.0), ('bbox_mAP_50', 0.0), ('bbox_mAP_75', 0.0), ('bbox_mAP_s', 0.0), ('bbox_mAP_m', 0.0), ('bbox_mAP_l', 0.0), ('bbox_mAP_copypaste', '0.000 0.000 0.000 0.000 0.000 0.000')])



python tools/train.py configs/dyhead/atss_swin-l-p4-w12_fpn_dyhead_mstrain_24e_logo.py

OrderedDict([('bbox_mAP', 0.002), ('bbox_mAP_50', 0.005), ('bbox_mAP_75', 0.001), ('bbox_mAP_s', 0.0), ('bbox_mAP_m', 0.004), ('bbox_mAP_l', 0.003), ('bbox_mAP_copypaste', '0.002 0.005 0.001 0.000 0.004 0.003')])

