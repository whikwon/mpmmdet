# Anchor
SOTA 성능을 내는 많은 detector는 anchor를 활용해서 bbox coordinate을 학습합니다. anchor가 무엇인지, 어떻게 one-stage, two-stage detector에서 사용되는지 알아보겠습니다.

## What is Anchor?
- Firstly suggested paper: [Faster-RCNN](https://arxiv.org/pdf/1506.01497.pdf)
- Papers using anchors
	- one-stage
		- [RetinaNet](https://arxiv.org/abs/1708.02002.pdf)
		- [SSD](https://arxiv.org/pdf/1512.02325.pdf)
	- two-stage
		- [Faster-RCNN](https://arxiv.org/pdf/1506.01497.pdf)
- Purpose:
	- image에 object가 있는 영역을 box로 예측해야 하는데, 예측을 용이하게 해주기 위해서 image로부터 얻은 feature map의 위치마다 default로 box를 여러 개를 그려서(anchor) 이 anchor들의 크기를 기준으로 차이에 대해서 학습하게 합니다. 즉, anchor의 크기가 적절하지 못한 경우에는 차이의 편차가 커지게 될 것이므로 학습이 어려워질 수 있어서 적절한 크기를 선정하는게 중요합니다. 
- Parameters:
	- scale: anchor size in **feature map**
	- ratio: anchor ratio in **feature map**
		- scale and ratio makes **base anchor size** in **feature map**
		- real anchor sizes = base anchor size * stride
	- stride: stride * feature map pixel location = absolute center point of original image

## How to draw grid anchors(`anchor_generator`)
anchor를 그리기 위해서는 위 parameter들이 필요합니다. box를 예측할 때, 우리는 feature map의 pixel 단위로 예측하기 때문에 anchor도 feature map과 같은 width, height를 가지면 됩니다. 

stride는 `[image_width // anchor_width], [image_height // anchor_height]`로 지정하는 경우에 image와 feature map 비율만큼의 크기를 anchor의 1개 pixel이 가지게 됩니다. 즉, image에서 상상을 하면 stride만큼 띄어서 anchor가 존재한다고 생각하시면 됩니다.(`grid_anchors`)

중심 좌표가 stride 만큼 떨어져서 존재한다고 보면 되고, 그 위에 그려지는 box의 크기는 base_anchor_size(`AnchorGenerator.base_anchors`)가 결정하게 됩니다. scale, ratio 2개 parameter로 결정되는 크기이고 크기의 단위는 1stride가 됩니다. 

feature map이 작은 경우, stride가 커지게 되고 scale, ratio의 image에서 실제 크기는 stride에 의해 결정되기 때문에 anchor box의 크기도 매우 커져서 예측하려는 물체가 상당히 클 것입니다. 

반대로 feature map이 큰 경우는 stride가 작고 위와 반대로 anchor box의 크기가 작아져서 예측하려는 물체가 작을 것입니다. 

이는 feature map의 크기에 따라서 예측하는 물체의 크기와도 상관이 있습니다.(보통 큰 feature map이 high-level 정보를 가지고 있어서 큰 물체를 예측 잘 하고, 작은 feature가 low-level 정보를 다뤄서 작은 물체 예측을 잘 한다고 알려져 있습니다.)

## Anchor as a target(`anchor_target`)
anchor는 학습할 때 box의 기본 틀로 사용된다고 했습니다. 위에서 anchor를 grid에 그리는 것을 완료했으면, target으로 변환해주는 과정을 거쳐야 합니다. 

학습 목표가 되는 target의 값은 anchor와 ground truth의 차이로 이루어지기 때문에 (**delta 수식 추가**) anchor와 ground truth 간의 overlap이 어느 정도 생기는 지를 계산하고, 일정 IoU 이상 겹치는 경우에 label을 주고 그 anchor의 부분만 차이에 해당하는 delta를 계산해야 gt가 있는 anchor에 대해서만 실제 학습할 수 있게 됩니다. 

## Train anchor
anchor target을 만들었다면 앞에서 나온 feature를 network(`anchor_head`)를 통과시켜 reg_pred로 delta를 예측하도록, score로 class를 예측하도록 학습시키면 됩니다. 

loss는 one/two-stage network 마다 다르게 적용되나 공통적으로 regression은 smooth-l1를, classification은 cross entropy를 가장 많이 사용합니다.

## Test 
anchor에 대해서 bbox 예측을 delta로 하기 때문에, delta를 bbox로 변환해주는 과정이 필요합니다. 

delta는 **anchor에 대한 차이**이기 때문에 anchor grid를 가지고 있으면 재변환해주는 과정은 어렵지 않습니다.

## anchor level, image의 관계 
주로 딥러닝에서 image 당 학습하는 게 일반적인데 feature가 FPN과 같이 여러 개가 생성되게 되는 경우에 image 별로, level 별로 feature가 생성이 되는 경우가 있고 그에 맞춰서 anchor도 level 별로 만들고 image 별로 대응해줘야 한다.

## Reference
- FasterRCNN
	- At each sliding-window location, we simultaneously predict multiple region proposals, where the number of maximum possible proposals for each location is denoted as $k$. So the *reg* layer has $4k$ outputs encoding the coordinates of $k$ boxes, and the *cls* layer outputs *2k* scores that estimate probability of object or not object for each proposal. The $k$ proposals are parameterized relative to $k$ reference boxes, which we call *anchors*. An anchor is centered at the sliding window in question, and is associated with a scale and aspect ratio. By default we use 3 scales and 3 aspect ratios, yielding $k = 9$ anchors at each sliding position. For a convolutional feature map of a size $W \times H$(typically $\sim 2400$), there are $WHk$ anchors in total.
- SSD
	- Our SSD is very similar to the region proposal network (RPN) in Faster R-CNN in that we also use a fixed set of (default) boxes for prediction, similar to the anchor boxes in the RPN. 
	- Our default boxes are similar to the anchor boxes used in Faster R-CNN, however we apply them to several feature maps of different resolutions.
- RetinaNet
	- We use translation-invariant anchor boxes similar to those in the RPNu variant [20]. The anchors have areas of $32^2$ to $512^2$ on pyramid levels $P_3$ to $P_7$, respectively. As in [20], at each pyramid level we use anchors at three aspect ratios ${1:2,1:1,2:1}$.  For denser scale coverage than in [20], at each level we add anchors of sizes ${2^0,2^{1/3},2^{2/3}}$ of the original set of 3 aspect ratio anchors. This improve AP in our setting. In total there are $A= 9$ anchors per level and across levels they cover the scale range $32 -813$ pixels with respect to the network’s input image. Each anchor is  assigned a length $K$ one-hot vector of classification targets,  where $K$ is the number of object classes, and a $4$-vector of box regression targets. We use the assignment rule from RPN [28] but modified for multi-class detection and with adjusted thresholds. Specifically,anchors are assigned to ground-truth object boxes using an intersection-over-union (IoU) threshold of $0.5;$ and to background if their IoU is in $[0, 0.4)$. As each anchor is assigned to at most one object box, we set the corresponding entry in its length $K$ label vector to $1$ and all other entries to $0$.If an anchor is unassigned, which may happen with overlap in $[0.4, 0.5)$, it is ignored during training. Box regression targets are computed as the offset between each anchor andits assigned object box, or omitted if there is no assignment.
