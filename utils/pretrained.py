import torch


def _copy_conv(dst, src):
    dst.conv.weight.data.copy_(src.conv.weight.data)
    dst.norm.weight.data.copy_(src.bn.weight.data)
    dst.norm.bias.data.copy_(src.bn.bias.data)
    dst.norm.running_mean.data.copy_(src.bn.running_mean.data)
    dst.norm.running_var.data.copy_(src.bn.running_var.data)


def _copy_split_conv(dst1, dst2, src_cv1):
    """Official cv1 produces 2c channels then .chunk(2, 1). Splitting its
    weights/BN stats in half reproduces the same function across your two
    separate conv1/conv2 blocks (same trick used in the public C2f_v2 split)."""
    c = dst1.conv.out_channels
    dst1.conv.weight.data.copy_(src_cv1.conv.weight.data[:c])
    dst2.conv.weight.data.copy_(src_cv1.conv.weight.data[c:2 * c])
    for dst, lo, hi in [(dst1, 0, c), (dst2, c, 2 * c)]:
        dst.norm.weight.data.copy_(src_cv1.bn.weight.data[lo:hi])
        dst.norm.bias.data.copy_(src_cv1.bn.bias.data[lo:hi])
        dst.norm.running_mean.data.copy_(src_cv1.bn.running_mean.data[lo:hi])
        dst.norm.running_var.data.copy_(src_cv1.bn.running_var.data[lo:hi])


def _copy_csp(dst_csp, src_c2f):
    _copy_split_conv(dst_csp.conv1, dst_csp.conv2, src_c2f.cv1)
    _copy_conv(dst_csp.conv3, src_c2f.cv2)
    for dst_res, src_bottleneck in zip(dst_csp.res_m, src_c2f.m):
        _copy_conv(dst_res.res_m[0], src_bottleneck.cv1)
        _copy_conv(dst_res.res_m[1], src_bottleneck.cv2)


def _copy_spp(dst_spp, src_sppf):
    _copy_conv(dst_spp.conv1, src_sppf.cv1)
    _copy_conv(dst_spp.conv2, src_sppf.cv2)


def _copy_head(dst_head, src_detect, copy_cls_final):
    # box regression branch: shapes are nc-independent, always safe to copy
    for d_branch, s_branch in zip(dst_head.box, src_detect.cv2):
        for d, s in zip(d_branch, s_branch):
            if isinstance(d, torch.nn.Conv2d):
                d.weight.data.copy_(s.weight.data)
                d.bias.data.copy_(s.bias.data)
            else:
                _copy_conv(d, s)

    # classification branch: final 1x1 conv depends on nc
    for d_branch, s_branch in zip(dst_head.cls, src_detect.cv3):
        mods, smods = list(d_branch), list(s_branch)
        for i, (d, s) in enumerate(zip(mods, smods)):
            is_last = (i == len(mods) - 1)
            if isinstance(d, torch.nn.Conv2d):
                if is_last and not copy_cls_final:
                    continue  # leave randomly initialized, shape won't match
                d.weight.data.copy_(s.weight.data)
                d.bias.data.copy_(s.bias.data)
            else:
                _copy_conv(d, s)


def load_pretrained(model, weights_path, nc):
    """
    Transfer official COCO-pretrained YOLOv8 weights into this from-scratch
    model. Requires `pip install ultralytics` purely to unpickle the
    checkpoint -- not used anywhere else in training.

    model:        an nn.yolo_v8_n/s/m/l/x instance, *before* .cuda()
    weights_path: path to the official .pt (e.g. the file you downloaded
                  from the Ultralytics page on Hugging Face)
    nc:           number of classes in your dataset. If it isn't 80, the
                  final per-class conv in each detection branch is skipped
                  (left randomly initialized) since its shape won't match.
    """
    from ultralytics import YOLO
    official = YOLO(weights_path).model.model
    official.eval()
    copy_cls_final = (nc == 80)

    with torch.no_grad():
        _copy_conv(model.net.p1[0], official[0])
        _copy_conv(model.net.p2[0], official[1])
        _copy_csp(model.net.p2[1], official[2])
        _copy_conv(model.net.p3[0], official[3])
        _copy_csp(model.net.p3[1], official[4])
        _copy_conv(model.net.p4[0], official[5])
        _copy_csp(model.net.p4[1], official[6])
        _copy_conv(model.net.p5[0], official[7])
        _copy_csp(model.net.p5[1], official[8])
        _copy_spp(model.net.p5[2], official[9])

        _copy_csp(model.fpn.h1, official[12])
        _copy_csp(model.fpn.h2, official[15])
        _copy_conv(model.fpn.h3, official[16])
        _copy_csp(model.fpn.h4, official[18])
        _copy_conv(model.fpn.h5, official[19])
        _copy_csp(model.fpn.h6, official[21])

        _copy_head(model.head, official[22], copy_cls_final)

    status = 'copied (nc=80 matches COCO)' if copy_cls_final else f'reinitialized (custom nc={nc})'
    print(f'[pretrained] backbone+neck+box-head loaded from {weights_path}, cls-head {status}')
    return model
