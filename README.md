# my-yolo
$1.$对yolov1与yolov2进行融合更改，用了yolov1的损失函数和检测头，用了yolov2的主体结构，但这在多尺度目标下效果会较差，但好处是较易与实现。  
$2.$正负样本匹配用yolov3的方法，具体方法如下。每个grid cell的框与真实框通过iou来进行框的匹配。
- 第一步，如果一个anchor与所有的gt_box的最大 IoU小于ignore_thresh 时，那这个anchor就是负样本。（一般ignore_thresh=0.7）
- 第二步，如果gt_box的中心点落在一个区域中，该区域就负责检测该物体。将与该物体有最大IoU的anchor作为正样本。注意这里没有用到ignore_thresh, 即使该最大IoU小于ignore_thresh也不会影响该anchor为正样本, 对于其他anchor，若IoU>ignore_thresh, 但不是最佳匹配，设置为忽略样本。  

根据上面匹配规则，yolo3中anchor有三种样本：正样本，负样本，忽略样本。
- 正样本：和gt_box有最大的IoU，无论是否满足IoU大于ignore_thresh，用1标记
- 负样本：不是和gt_box有最大IoU的anchor，且IoU 小于ignore_thresh， 用0标记
- 忽略样本：不是和gt_box有最大IoU的anchor，且IoU 大于 ignore_thresh， 用-1标记
只有正负样本才参与损失的计算，忽略样本不进损失的计算。
$3.$正样本的参与分类损失、边界框损失和是否存在物体损失。负样本只参与是否存在物体损失。
## 更改
$1.$ 使用了yolov1的检测头，并且每个grid cell有b个框，b是一个超参数。

## 注意
$1.$ 输出中的object那一项不能省略，否则的话负样本就计算不了损失了。
$2.$ 在后续更改中可以对分类损失改为交叉熵损失并且对交叉熵损失加上focus loss。