import torch
import torch.optim as optim

from utils.datasets import *
from utils import torch_utils
from models import *
from merge_model import MergeModel
import argparse

from test_merge import test_merge


def train():
    cfg = opt.cfg
    data = opt.data
    img_size = opt.img_size
    epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    weights_rgb = opt.weights_rgb  # initial training weights
    weights_d = opt.weights_d
    accumulate = opt.accumulate

    # Configure run
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    nc = int(data_dict['classes'])  # number of classes

    model_rgb = model_d = model_merge = None

    # init model
    if model_rgb is None and model_d is None and model_merge is None:
        device = torch_utils.select_device(opt.device)
        verbose = True

        # Initialize models
        model_rgb = Darknet(cfg, img_size).to(device)
        model_d = Darknet(cfg, img_size).to(device)
        model_merge = MergeModel().to(device)

        # Load weights
        model_rgb.load_state_dict(torch.load(weights_rgb, map_location=device)['model'])
        model_d.load_state_dict(torch.load(weights_d, map_location=device)['model'])

        model_rgb.eval()
        model_d.eval()
        model_merge.train()
    else:
        return -1

    if opt.adam:
        optimizer = optim.Adam(model_merge.parameters(), lr=0.001)
        # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
    else:
        optimizer = optim.SGD(model_merge.parameters(), lr=0.001, momentum=0.5, nesterov=True)

    # init dataloader for bot datapoints
    # Dataset
    dataset = LoadImagesAndLabels(train_path,
                                  img_size,
                                  batch_size,
                                  rect=False,
                                  augment=True)

    # Dataloader
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=min([os.cpu_count(), batch_size, 16]),
                                             shuffle=True,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)
    nb = len(dataloader)
    # train
    for epoch in range(epochs):

        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, stuff in pbar:
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs, targets, paths, shapes = stuff
            targets = targets.to(device)
            img_rgb = imgs[0].to(device)
            img_d = imgs[1].to(device)

            # feed forward
            inf_rgb_out, train_rgb_out = model_rgb(img_rgb)  # inference and training outputs
            inf_d_out, train_d_out = model_d(img_d)  # inference and training outputs

            # feed predictions into own net
            pred, _ = model_merge(inf_rgb_out, inf_d_out)

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model_rgb)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            loss.backward()

            # Accumulate gradient for x batches before optimizing
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 6) % (
                '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *mloss, len(targets), img_size)
            pbar.set_description(s)

            # end batch ------------------------------------------------------------------------------------------------

            # Update scheduler

        final_epoch = epoch + 1 == epochs
        # Process epoch results
        if not (opt.notest or (opt.nosave and epoch < 10)) or final_epoch:
            with torch.no_grad():
                # TODO: Rewrite test class properly
                results, maps = test_merge(cfg,
                                          data,
                                          batch_size=batch_size,
                                          img_size=opt.img_size,
                                          model=model_merge,
                                          conf_thres=0.001 if final_epoch and epoch > 0 else 0.1,  # 0.1 for speed
                                          save_json=final_epoch and epoch > 0 and 'coco.data' in data)

        # Write epoch results
        results_file = 'results_merge.txt'
        with open(results_file, 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int,
                        default=20)  # 500200 batches at bs 16, 117263 images = 273 epochs
    parser.add_argument('--batch-size', type=int,
                        default=4)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--accumulate', type=int, default=1, help='batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/household.data', help='*.data file path')
    parser.add_argument('--multi-scale', action='store_true',
                        help='adjust (67% - 150%) img_size every 10 batches')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--transfer', action='store_true', help='transfer learning')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights-rgb', type=str, default='weights/best_color.pt', help='initial weights')
    parser.add_argument('--weights-d', type=str, default='weights/best_depth.pt', help='initial weights')
    parser.add_argument('--arc', type=str, default='default', help='yolo architecture')  # defaultpw, uCE, uBCE
    parser.add_argument('--prebias', action='store_true', help='transfer-learn yolo biases prior to training')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--var', type=float, help='debug variable')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--img-weights', action='store_true', help='select training images by weight')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    opt = parser.parse_args()
    print(opt)
    device = torch_utils.select_device(opt.device)
    train()
