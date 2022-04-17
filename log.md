1. loss_rpn_loc goes to inf. change learning rate, use giou, not l1_loss, drop small boxes
2. It is the gt_box_deltas that becomes -inf
3. the method box2box_transform.get_deltas can output: dw = -inf