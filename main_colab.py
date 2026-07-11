import argparse
import copy
import csv
import os
import warnings

import numpy
import torch
import tqdm
import yaml
from torch.utils import data
from torchvision.ops import box_iou as _box_iou
# import ultralytics

from utils.pretrained import load_pretrained
from nets import nn
from utils import util
from utils import d4
from utils.dataset import Dataset

warnings.filterwarnings("ignore")

def decode_for_nms(outputs, criterion):
    """
    Decode raw YOLOv8 feature maps into (B, 4+nc, N) tensor ready for NMS.
    Mirrors ComputeLoss.__call__ decoding exactly:
      - DFL softmax + project  -> ltrb distances
      - anchor_points + stride -> x1y1x2y2 pixel coords
      - sigmoid on class scores
    Then re-encodes as cx,cy,w,h so wh2xy() inside non_max_suppression
    converts back to correct x1y1x2y2 pixel boxes.
    """
    x = outputs[1] if isinstance(outputs, tuple) else outputs

    # Concatenate all scales: (B, no, N)
    no  = criterion.no
    dfl_ch = criterion.dfl_ch
    cat = torch.cat([i.view(x[0].shape[0], no, -1) for i in x], dim=2)

    # Split into box distribution and class scores
    pred_dist, pred_scores = cat.split((4 * dfl_ch, criterion.nc), dim=1)

    # DFL decode: (B, 4*dfl_ch, N) -> (B, N, 4) ltrb
    B, _, N = pred_dist.shape
    pred_dist = pred_dist.permute(0, 2, 1).contiguous()
    pred_dist = pred_dist.view(B, N, 4, dfl_ch).softmax(3)
    pred_dist = pred_dist.matmul(criterion.project.type(pred_dist.dtype))

    # Anchor points + stride tensor
    anchor_points, stride_tensor = util.make_anchors(x, criterion.stride, 0.5)

    # ltrb -> x1y1x2y2 in pixel space
    lt, rb     = pred_dist.chunk(2, -1)
    boxes_xyxy = torch.cat((anchor_points - lt,
                             anchor_points + rb), -1) * stride_tensor  # (B, N, 4)

    # Class scores: sigmoid -> (B, N, nc)
    scores = pred_scores.permute(0, 2, 1).sigmoid()

    # NMS expects (B, 4+nc, N) with boxes in cx,cy,w,h so wh2xy() inside NMS
    # converts them back to x1y1x2y2 pixel coords correctly
    cx = (boxes_xyxy[..., 0] + boxes_xyxy[..., 2]) / 2
    cy = (boxes_xyxy[..., 1] + boxes_xyxy[..., 3]) / 2
    w  =  boxes_xyxy[..., 2] - boxes_xyxy[..., 0]
    h  =  boxes_xyxy[..., 3] - boxes_xyxy[..., 1]
    cxcywh = torch.stack([cx, cy, w, h], dim=-1)               # (B, N, 4)

    return torch.cat([cxcywh, scores], dim=-1).permute(0, 2, 1).contiguous()


def decode_dense(outputs, criterion):
    """
    Same DFL decode as decode_for_nms, but returns per-cell predictions
    WITHOUT concatenating (cx,cy,w,h) for NMS and WITHOUT any no_grad/NMS step.
    This is the version used for the dense equivariance-consistency loss:
    every cell is kept (no thresholding, no suppression), so cells line up
    1:1 with a permutation derived from the same D4 op applied to the image,
    and gradients flow through normally when called on the student branch.

    Returns:
        boxes_xyxy: (B, N, 4)  decoded pixel-space x1y1x2y2
        scores:     (B, N, nc) sigmoid class scores
        shapes:     list[(H, W)] per FPN scale, in the same order they were
                    concatenated into N, needed to build the per-scale grid
                    permutation and stitch it back together.
    """
    x = outputs[1] if isinstance(outputs, tuple) else outputs

    shapes = [(xi.shape[-2], xi.shape[-1]) for xi in x]

    no = criterion.no
    dfl_ch = criterion.dfl_ch
    cat = torch.cat([i.view(x[0].shape[0], no, -1) for i in x], dim=2)

    pred_dist, pred_scores = cat.split((4 * dfl_ch, criterion.nc), dim=1)

    B, _, N = pred_dist.shape
    pred_dist = pred_dist.permute(0, 2, 1).contiguous()
    pred_dist = pred_dist.view(B, N, 4, dfl_ch).softmax(3)
    pred_dist = pred_dist.matmul(criterion.project.type(pred_dist.dtype))

    anchor_points, stride_tensor = util.make_anchors(x, criterion.stride, 0.5)

    lt, rb = pred_dist.chunk(2, -1)
    boxes_xyxy = torch.cat((anchor_points - lt,
                             anchor_points + rb), -1) * stride_tensor  # (B, N, 4)

    scores = pred_scores.permute(0, 2, 1).sigmoid()  # (B, N, nc)

    return boxes_xyxy, scores, shapes


def build_full_permutation(shapes, op, device):
    """
    Stitches per-scale grid_permutation() outputs into one permutation over
    the full concatenated N (matching the scale order used in decode_dense /
    ComputeLoss). Each scale is permuted independently (a P3 cell can only
    correspond to another P3 cell under a D4 op) with an offset added so
    indices point into the right slice of the concatenated tensor.
    """
    perms = []
    offset = 0
    for (h, w) in shapes:
        p = d4.grid_permutation(h, w, op, device) + offset
        perms.append(p)
        offset += h * w
    return torch.cat(perms, dim=0)  # (N,)


def consistency_weight_schedule(epoch, total_epochs, ramp_epochs, rampdown_epochs, weight_max):
    """
    Piecewise-linear schedule for the consistency loss weight:
      - epochs [0, ramp_epochs):                     linear ramp 0 -> weight_max
      - epochs [ramp_epochs, rampdown_start):         flat at weight_max
      - epochs [rampdown_start, total_epochs):        linear ramp weight_max -> 0

    rampdown_start = total_epochs - rampdown_epochs, clamped to be >= ramp_epochs
    so short runs (where ramp and rampdown windows would otherwise overlap)
    degrade gracefully into a triangular ramp-up-then-down instead of a
    negative-length flat window.

    Motivation: on the 60-epoch mosaic-off test, holding cons_weight flat for
    the entire post-warmup period made the model underperform baseline once
    mosaic turned off (epoch 50-60) -- the regularizer seemed to compete with
    late-stage fine-tuning on clean (non-mosaic) images rather than help it.
    Ramping back down to 0 over that same window lets the consistency loss
    only act during the phase it demonstrably helped (early/mid training)
    and gets out of the way once det_loss should be doing the fine-tuning.
    """
    if weight_max == 0.0:
        return 0.0

    rampdown_start = max(ramp_epochs, total_epochs - rampdown_epochs)

    if epoch < ramp_epochs:
        return weight_max * (epoch + 1) / ramp_epochs
    if epoch < rampdown_start:
        return weight_max
    remaining = max(1, total_epochs - 1 - rampdown_start)
    frac = (epoch - rampdown_start) / remaining
    return weight_max * max(0.0, 1.0 - frac)


def learning_rate(args, params):
    def fn(x):
        return (1 - x / args.epochs) * (1.0 - params['lrf']) + params['lrf']

    return fn


def train(args, params):
    # Model
    model = nn.yolo_v8_n(len(params['names'].values()))
    if args.weights:
        load_pretrained(model, args.weights, len(params['names'].values()))
    model = model.cuda()
    # Optimizer
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * accumulate / 64

    p = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
            p[2].append(v.bias)
        if isinstance(v, torch.nn.BatchNorm2d):
            p[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
            p[0].append(v.weight)

    optimizer = torch.optim.SGD(p[2], params['lr0'], params['momentum'], nesterov=True)

    optimizer.add_param_group({'params': p[0], 'weight_decay': params['weight_decay']})
    optimizer.add_param_group({'params': p[1]})
    del p

    # Scheduler
    lr = learning_rate(args, params)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr, last_epoch=-1)

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    filenames = []
    with open(args.train_txt) as reader:

        # for filename in reader.readlines():
        #     filename = filename.rstrip().split('/')[-1]
        #     filenames.append(filename.strip())

        for filename in reader:
            filenames.append(filename.strip())
    dataset = Dataset(filenames, args.input_size, params, True)

    if args.world_size <= 1:
        sampler = None
    else:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset, args.batch_size, sampler is None, sampler,
                             num_workers=2, pin_memory=True, collate_fn=Dataset.collate_fn)

    if args.world_size > 1:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    # Start training
    best = 0
    num_batch = len(loader)
    amp_scale = torch.cuda.amp.GradScaler()
    criterion = util.ComputeLoss(model, params)
    num_warmup = max(round(params['warmup_epochs'] * num_batch), 1000)
    with open('weights/step.csv', 'w') as f:
        if args.local_rank == 0:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'mAP@50', 'mAP'])
            writer.writeheader()
        for epoch in range(args.epochs):
            model.train()

            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            m_loss = util.AverageMeter()
            m_cons = util.AverageMeter()
            if args.world_size > 1:
                sampler.set_epoch(epoch)
            p_bar = enumerate(loader)
            if args.local_rank == 0:
                print(('\n' + '%10s' * 4) % ('epoch', 'memory', 'loss', 'cons'))
            if args.local_rank == 0:
                p_bar = tqdm.tqdm(p_bar, total=num_batch)  # progress bar

            optimizer.zero_grad()

            for i, (samples, targets, _) in p_bar:
                x = i + num_batch * epoch  # number of iterations
                samples = samples.cuda().float() / 255
                targets = targets.cuda()

                # Warmup
                if x <= num_warmup:
                    xp = [0, num_warmup]
                    fp = [1, 64 / (args.batch_size * args.world_size)]
                    accumulate = max(1, numpy.interp(x, xp, fp).round())
                    for j, y in enumerate(optimizer.param_groups):
                        if j == 0:
                            fp = [params['warmup_bias_lr'], y['initial_lr'] * lr(epoch)]
                        else:
                            fp = [0.0, y['initial_lr'] * lr(epoch)]
                        y['lr'] = numpy.interp(x, xp, fp)
                        if 'momentum' in y:
                            fp = [params['warmup_momentum'], params['momentum']]
                            y['momentum'] = numpy.interp(x, xp, fp)

                with torch.cuda.amp.autocast():
                    outputs = model(samples)
                det_loss = criterion(outputs, targets)

                # --- Consistency regularization (dense, D4, task-space) ---
                #
                # What changed vs. the previous version, and why:
                #
                # 1. BUG FIX: the old code wrapped BOTH the teacher decode
                #    (`outputs`) and the student decode (`outputs_aug`) in the
                #    same `torch.no_grad()` block before matching. That means
                #    cons_loss had NO gradient at all -- it never influenced a
                #    single weight update, regardless of thresholds. Only the
                #    teacher branch should ever be detached.
                #
                # 2. Matching is now DENSE and EXACT instead of post-NMS/
                #    greedy-IoU. Every D4 op maps grid cell (r,c) to a unique
                #    cell in the transformed grid, so instead of running NMS
                #    and hoping IoU-matching finds the same object twice, we
                #    directly permute the teacher's per-cell predictions with
                #    that exact mapping. No matching failures, no dependence
                #    on confidence thresholds to "find" a pair, fully
                #    differentiable end to end.
                #
                # 3. Random D4 op instead of hflip-only: adds rotation
                #    equivariance too. Reuses the same DFL/anchor decode
                #    logic already validated in decode_for_nms.
                #
                # 4. Stop-gradient teacher (Mean-Teacher / BYOL style) instead
                #    of symmetric matching: cheaper (reuses `outputs`, which
                #    is already computed for det_loss -- no extra forward
                #    pass beyond the augmented view), and avoids the collapse
                #    risk of two branches chasing each other with no anchor.
                #
                # 5. Confidence masking: only supervise cells where the
                #    TEACHER (original, unaugmented -- more reliable) is
                #    already confident. Dense wheat-head images are mostly
                #    background; unmasked dense consistency would mostly
                #    train the model to agree on "nothing here", which is
                #    free and uninformative.
                #
                # 6. Loss weight ramps up over the first RAMP_EPOCHS epochs.
                #    Early in training the network's grid-cell semantics
                #    aren't calibrated yet, and a strong consistency signal
                #    on noisy predictions can drag training in a bad
                #    direction before the supervised loss finds its footing.
                CONF_THRES  = 0.10             # teacher confidence gate (per-cell max class prob)
                RAMP_EPOCHS = args.ramp_epochs # linear ramp of consistency weight
                CONS_WEIGHT_MAX = args.cons_weight  # weight once ramp completes; 0.0 == baseline
                CLS_LOSS_W  = 1.0
                BOX_LOSS_W  = 2.0
                DIAG_EVERY  = 50

                if CONS_WEIGHT_MAX == 0.0:
                    # Skip the augmented forward pass and dense decode entirely --
                    # avoids wasted compute if this script is ever run in
                    # baseline mode, and keeps `loss == det_loss` exactly rather
                    # than det_loss + 0*cons_loss (which would still perturb the
                    # RNG stream via the D4 op sampling / extra forward call).
                    loss = det_loss
                    cons_loss = torch.tensor(0.0, device=samples.device)
                    cons_weight = 0.0
                else:
                    op = d4.D4_OPS[torch.randint(len(d4.D4_OPS), (1,)).item()]
                    samples_aug = d4.transform_image(samples, op)

                    with torch.cuda.amp.autocast():
                        outputs_aug = model(samples_aug)

                    # Teacher: original-frame predictions, stop-gradient only.
                    # Reuses `outputs` computed above for det_loss -- no extra
                    # forward pass. Decode outside autocast is fine; inputs are
                    # already fp16/fp32 tensors from the model.
                    with torch.no_grad():
                        t_boxes, t_scores, shapes = decode_dense(outputs, criterion)

                    # Student: augmented-frame predictions, gradient flows.
                    s_boxes, s_scores, _ = decode_dense(outputs_aug, criterion)

                    img_h, img_w = samples.shape[-2], samples.shape[-1]

                    # Reorder the teacher's cells into the student's raster order,
                    # and express its boxes in the augmented image's pixel frame.
                    perm = build_full_permutation(shapes, op, samples.device)
                    t_boxes_aligned = d4.transform_boxes_xyxy(t_boxes[:, perm, :], op, img_w, img_h)
                    t_scores_aligned = t_scores[:, perm, :]

                    mask = t_scores_aligned.max(dim=-1).values > CONF_THRES  # (B, N)
                    n_valid = mask.sum()

                    if n_valid > 0:
                        cls_loss_cons = torch.nn.functional.mse_loss(
                            s_scores[mask], t_scores_aligned[mask].detach()
                        )
                        try:
                            from torchvision.ops import complete_box_iou_loss
                            box_loss_cons = complete_box_iou_loss(
                                s_boxes[mask], t_boxes_aligned[mask].detach(), reduction='mean'
                            )
                        except (ImportError, AttributeError):
                            # older torchvision without CIoU loss: fall back to smooth L1
                            box_loss_cons = torch.nn.functional.smooth_l1_loss(
                                s_boxes[mask], t_boxes_aligned[mask].detach()
                            )
                        cons_loss = CLS_LOSS_W * cls_loss_cons + BOX_LOSS_W * box_loss_cons
                    else:
                        cons_loss = torch.tensor(0.0, device=samples.device)

                    cons_weight = consistency_weight_schedule(
                        epoch, args.epochs, RAMP_EPOCHS, args.rampdown_epochs, CONS_WEIGHT_MAX
                    )

                    if args.local_rank == 0 and x % DIAG_EVERY == 0:
                        frac_valid = n_valid.item() / mask.numel()
                        print(f'\n[DIAG iter {x}] op={op} '
                              f'frac_masked_cells={frac_valid:.4f} '
                              f'cons_weight={cons_weight:.3f} '
                              f'cons={cons_loss.item():.4f}')

                    loss = det_loss + cons_weight * cons_loss

                m_loss.update(loss.item(), samples.size(0))
                m_cons.update(cons_loss.item() if isinstance(cons_loss, torch.Tensor) else cons_loss, samples.size(0))

                loss *= args.batch_size  # loss scaled by batch_size
                loss *= args.world_size  # gradient averaged between devices in DDP mode

                # Backward
                amp_scale.scale(loss).backward()

                # Optimize
                if x % accumulate == 0:
                    amp_scale.unscale_(optimizer)  # unscale gradients
                    util.clip_gradients(model)  # clip gradients
                    amp_scale.step(optimizer)  # optimizer.step
                    amp_scale.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                # Log
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'
                    s = ('%10s' * 2 + '%10.4g' * 2) % (f'{epoch + 1}/{args.epochs}', memory, m_loss.avg, m_cons.avg)
                    p_bar.set_description(s)

                del loss
                del outputs

            # Scheduler
            scheduler.step()

            if args.local_rank == 0:
                # mAP
                last = test(args, params, ema.ema)
                writer.writerow({'mAP': str(f'{last[1]:.3f}'),
                                 'epoch': str(epoch + 1).zfill(3),
                                 'mAP@50': str(f'{last[0]:.3f}')})
                f.flush()

                # Update best mAP
                if last[1] > best:
                    best = last[1]

                # Save model
                ckpt = {'model': copy.deepcopy(ema.ema).half()}

                # Save last, best and delete
                torch.save(ckpt, './weights/last.pt')
                if best == last[1]:
                    torch.save(ckpt, './weights/best.pt')
                del ckpt

    if args.local_rank == 0:
        util.strip_optimizer('./weights/best.pt')  # strip optimizers
        util.strip_optimizer('./weights/last.pt')  # strip optimizers

    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, params, model=None):
    filenames = []
    with open(args.val_txt) as reader:
        # for filename in reader.readlines():
        #     filename = filename.rstrip().split('/')[-1]
        #     filenames.append(filename.strip())
        for filename in reader:
          filenames.append(filename.strip())    

    dataset = Dataset(filenames, args.input_size, params, False)
    loader = data.DataLoader(dataset, 8, False, num_workers=2,
                             pin_memory=True, collate_fn=Dataset.collate_fn)

    if model is None:
        model = torch.load('./weights/best.pt', map_location='cuda', weights_only=False)['model'].float()

    model.half()
    model.eval()

    # Configure
    iou_v = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0.
    m_rec = 0.
    map50 = 0.
    mean_ap = 0.
    metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 3) % ('precision', 'recall', 'mAP'))
    for samples, targets, shapes in p_bar:
        samples = samples.cuda()
        targets = targets.cuda()
        samples = samples.half()  # uint8 to fp16/32
        samples = samples / 255  # 0 - 255 to 0.0 - 1.0
        _, _, height, width = samples.shape  # batch size, channels, height, width

        # Inference
        # print('samp',samples.shape)
        outputs = model(samples)
        # print('ouputs_1',outputs)
        # print(outputs.shape)
        # print(outputs[0, :, :5])      # if shape is (B,C,N)


        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height)).cuda()  # to pixels
        outputs = util.non_max_suppression(outputs, 0.001, 0.65)
        # print('ouputs_2',outputs)
        # Metrics
        for i, output in enumerate(outputs):
            labels = targets[targets[:, 0] == i, 1:]
            correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if labels.shape[0]:
                    metrics.append((correct, *torch.zeros((3, 0)).cuda()))
                continue

            detections = output.clone()
            util.scale(detections[:, :4], samples[i].shape[1:], shapes[i][0], shapes[i][1])

            # Evaluate
            if labels.shape[0]:
                tbox = labels[:, 1:5].clone()  # target boxes
                tbox[:, 0] = labels[:, 1] - labels[:, 3] / 2  # top left x
                tbox[:, 1] = labels[:, 2] - labels[:, 4] / 2  # top left y
                tbox[:, 2] = labels[:, 1] + labels[:, 3] / 2  # bottom right x
                tbox[:, 3] = labels[:, 2] + labels[:, 4] / 2  # bottom right y
                util.scale(tbox, samples[i].shape[1:], shapes[i][0], shapes[i][1])

                correct = numpy.zeros((detections.shape[0], iou_v.shape[0]))
                correct = correct.astype(bool)

                t_tensor = torch.cat((labels[:, 0:1], tbox), 1)
                iou = util.box_iou(t_tensor[:, 1:], detections[:, :4])
                # print('iou',iou)
                correct_class = t_tensor[:, 0:1] == detections[:, 5]
                # print('correct class',correct_class)
                for j in range(len(iou_v)):
                    x = torch.where((iou >= iou_v[j]) & correct_class)
                    if x[0].shape[0]:
                        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                        matches = matches.cpu().numpy()
                        # print('matches', matches)
                        if x[0].shape[0] > 1:
                            matches = matches[matches[:, 2].argsort()[::-1]]
                            matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                            matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
                        correct[matches[:, 1].astype(int), j] = True
                correct = torch.tensor(correct, dtype=torch.bool, device=iou_v.device)
                # print('correct', correct)
            metrics.append((correct, output[:, 4], output[:, 5], labels[:, 0]))

    # Compute metrics

    metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics)

    # Print results
    print('%10.3g' * 3 % (m_pre, m_rec, mean_ap))

    # Return results
    model.float()  # for training
    return map50, mean_ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--train-txt', default='', type=str)
    parser.add_argument('--val-txt',   default='', type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--weights', default='', type=str,
                     help='path to pretrained .pt to initialize from (e.g. yolov8n.pt)')
    parser.add_argument('--seed', default=0, type=int,
                     help='random seed for python/numpy/torch, overrides util.setup_seed() default')
    parser.add_argument('--cons-weight', default=1.0, type=float,
                     help='max weight of the D4 consistency loss after ramp-up; 0.0 disables it entirely (baseline mode)')
    parser.add_argument('--ramp-epochs', default=20, type=int,
                     help='number of epochs over which the consistency loss weight linearly ramps to --cons-weight')
    parser.add_argument('--rampdown-epochs', default=0, type=int,
                     help='number of FINAL epochs over which the consistency loss weight linearly ramps back down to 0 '
                          '(0 disables rampdown, i.e. flat at --cons-weight after ramp-up). '
                          'Try setting this to match your mosaic cutoff (default mosaic-off window is the last 10 epochs).')

    args = parser.parse_args()

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))

    if args.world_size > 1:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    util.setup_seed()
    util.setup_multi_processes()

    # util.setup_seed() typically pins a fixed default seed (commonly 0) --
    # re-seed explicitly here so --seed actually varies the run. This must
    # happen before the DataLoader is constructed/iterated in train(), since
    # worker processes derive their seed from torch's RNG state at spawn time.
    import random
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    with open(os.path.join('utils', 'args.yaml'), errors='ignore') as f:
        params = yaml.safe_load(f)

    if args.train:
        train(args, params)
    if args.test:
        test(args, params)


if __name__ == "__main__":
    main()
