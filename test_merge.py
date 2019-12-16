import argparse
import json

from torch.utils.data import DataLoader

from models import *
from merge_model import MergeModel
from utils.datasets import *
from utils.utils import *


def test_merge(cfg,
                data,
                weights_rgb=None,
                weights_d=None,
                weights_merge=None,
                batch_size=16,
                img_size=416,
                iou_thres=0.5,
                conf_thres=0.001,
                nms_thres=0.5,
                save_json=False,
                model_rgb=None,
                model_d=None,
                model_merge=None):
    # Initialize/load model and set device
    if model_rgb is None and model_d is None and model_merge is None:
        device = torch_utils.select_device(opt.device)
        verbose = True

        # Initialize model
        model_rgb = Darknet(cfg, img_size).to(device)
        model_d = Darknet(cfg, img_size).to(device)
        model_merge = MergeModel().to(device)

        # Load weights
        # attempt_download(weights)
        model_rgb.load_state_dict(torch.load(weights_rgb, map_location=device)['model'])
        model_d.load_state_dict(torch.load(weights_d, map_location=device)['model'])
        model_merge.load_state_dict(torch.load(weights_merge, map_location=device)['model'])
    else:
        device = next(model_rgb.parameters()).device
        verbose = False

    # Configure run
    data = parse_data_cfg(data)
    nc = int(data['classes'])  # number of classes
    test_path = data['valid']  # path to test images
    names = load_classes(data['names'])  # class names

    # Dataloader
    dataset = LoadImagesAndLabels(test_path, img_size, batch_size)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=min([os.cpu_count(), batch_size, 16]),
                            pin_memory=True,
                            collate_fn=dataset.collate_fn,
                            drop_last=True)

    seen = 0
    model_rgb.eval()
    model_d.eval()
    model_merge.eval()
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3)
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, stuff in enumerate(tqdm(dataloader, desc=s)):
        imgs, targets, paths, shapes = stuff
        targets = targets.to(device)
        img_rgb = imgs[0].to(device)
        img_d = imgs[1].to(device)
        _, _, height, width = img_rgb.shape  # batch size, channels, height, width

        # Run model
        inf_rgb_out, train_rgb_out = model_rgb(img_rgb)  # inference and training outputs
        inf_d_out, train_d_out = model_d(img_d)  # inference and training outputs

        rgb_pred = non_max_suppression(inf_rgb_out)
        d_pred = non_max_suppression(inf_d_out)

        # normalize again, because somehow the predictions de-normalize stuff, no idea how
        for pred in rgb_pred:
            if pred is not None:
                pred[:, :4] /= img_size
                pred[:, :4] = xyxy2xywh(pred[:, :4])

        for pred in d_pred:
            if pred is not None:
                pred[:, :4] /= img_size
                pred[:, :4] = xyxy2xywh(pred[:, :4])

        # take first 10 predictions after nms of both nets
        # -> (bs, 20, 12)
        # feed predictions into own net
        # input: 2x [bs, Tensor([[x,y,w,h,o,c0,..c6,cls]])]
        all_preds = [[] for _ in range(batch_size)]
        for batch in range(batch_size):
            prgb = rgb_pred[batch]
            pd = d_pred[batch]
            if prgb is not None:
                for j in prgb:
                    all_preds[batch].extend(j[:-1])

            if pd is not None:
                for j in pd:
                    all_preds[batch].extend(j[:-1])

            for _ in range(len(all_preds[batch]), 240):  # to 20 * 12
                all_preds[batch].append(torch.Tensor([0]))

        # make tensor from this
        all_preds = torch.Tensor(all_preds).to(device)

        pred = model_merge(all_preds)

        # Run NMS
        # Targets: tensor([[batch_i, cls, x, y, w, h]])
        output = non_max_suppression(pred, conf_thres=conf_thres, nms_thres=nms_thres)

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Assign all predictions as incorrect
            correct = [0] * len(pred)
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                tbox[:, [0, 2]] *= width
                tbox[:, [1, 3]] *= height

                # Search for correct predictions
                for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                    # Break if all targets already located in image
                    if len(detected) == nl:
                        break

                    # Continue if predicted class not among image classes
                    if pcls.item() not in tcls:
                        continue

                    # Best iou, index between pred and targets
                    m = (pcls == tcls_tensor).nonzero().view(-1)
                    iou, bi = bbox_iou(pbox, tbox[m]).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > iou_thres and m[bi] not in detected:  # and pcls == tcls[bi]:
                        correct[i] = 1
                        detected.append(m[bi])

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1, *(loss / len(dataloader)).tolist()), maps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/household.data', help='coco.data file path')
    parser.add_argument('--weights-rgb', type=str, default='weights/best_color.pt', help='path to weights file')
    parser.add_argument('--weights-d', type=str, default='weights/best_depth.pt', help='path to weights file')
    parser.add_argument('--weights-merge', type=str, default='weights/best_merge.pt', help='path to weights file')
    parser.add_argument('--batch-size', type=int, default=4, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        test_merge(opt.cfg,
             opt.data,
             opt.weights_rgb,
             opt.weights_d,
             opt.weights_merge,
             opt.batch_size,
             opt.img_size,
             opt.iou_thres,
             opt.conf_thres,
             opt.nms_thres,
             opt.save_json)
