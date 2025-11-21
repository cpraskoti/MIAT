from __future__ import print_function
import torch as t
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import time
import numpy as np
from tqdm import tqdm

import data_loader as lo
from config import args, device
import model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Evaluate():
    def __init__(self, drawImg=False):
        self.op = 0
        self.drawImg = drawImg
        self.scale = 0.3048  # Convert feet to meters
        self.prop = 1

    def maskedMSETest(self, y_pred, y_gt, mask):
        acc = t.zeros_like(mask)
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        out = t.pow(x - muX, 2) + t.pow(y - muY, 2)
        acc[:, :, 0] = out
        acc[:, :, 1] = out
        acc = acc * mask
        lossVal = t.sum(acc[:, :, 0], dim=1)
        counts = t.sum(mask[:, :, 0], dim=1)
        loss = t.sum(acc) / t.sum(mask)
        return lossVal, counts, loss

    def logsumexp(self, inputs, dim=None, keepdim=False):
        if dim is None:
            inputs = inputs.view(-1)
            dim = 0
        s, _ = t.max(inputs, dim=dim, keepdim=True)
        outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
        if not keepdim:
            outputs = outputs.squeeze(dim)
        return outputs

    def maskedNLLTest(self, fut_pred, lat_pred, lon_pred, fut, op_mask, num_lat_classes=3, num_lon_classes=2,
                      use_maneuvers=True):
        if use_maneuvers:
            acc = t.zeros(op_mask.shape[0], op_mask.shape[1], num_lon_classes * num_lat_classes).to(device)
            count = 0
            for k in range(num_lon_classes):
                for l in range(num_lat_classes):
                    wts = lat_pred[:, l] * lon_pred[:, k]
                    wts = wts.repeat(len(fut_pred[0]), 1)
                    y_pred = fut_pred[k * num_lat_classes + l]
                    y_gt = fut
                    muX = y_pred[:, :, 0]
                    muY = y_pred[:, :, 1]
                    sigX = y_pred[:, :, 2]
                    sigY = y_pred[:, :, 3]
                    rho = y_pred[:, :, 4]
                    ohr = t.pow(1 - t.pow(rho, 2), -0.5)
                    x = y_gt[:, :, 0]
                    y = y_gt[:, :, 1]
                    
                    out = -(0.5 * t.pow(ohr, 2) * (
                            t.pow(sigX, 2) * t.pow(x - muX, 2) + 0.5 * t.pow(sigY, 2) * t.pow(
                        y - muY, 2) - rho * t.pow(sigX, 1) * t.pow(sigY, 1) * (x - muX) * (
                                    y - muY)) - t.log(sigX * sigY * ohr) + 1.8379)
                    acc[:, :, count] = out + t.log(wts)
                    count += 1
            acc = -self.logsumexp(acc, dim=2)
            acc = acc * op_mask[:, :, 0]
            loss = t.sum(acc) / t.sum(op_mask[:, :, 0])
            lossVal = t.sum(acc, dim=1)
            counts = t.sum(op_mask[:, :, 0], dim=1)
            return lossVal, counts, loss
        else:
            acc = t.zeros(op_mask.shape[0], op_mask.shape[1], 1).to(device)
            y_pred = fut_pred
            y_gt = fut
            muX = y_pred[:, :, 0]
            muY = y_pred[:, :, 1]
            sigX = y_pred[:, :, 2]
            sigY = y_pred[:, :, 3]
            rho = y_pred[:, :, 4]
            ohr = t.pow(1 - t.pow(rho, 2), -0.5)
            x = y_gt[:, :, 0]
            y = y_gt[:, :, 1]
            
            out = 0.5 * t.pow(ohr, 2) * (
                    t.pow(sigX, 2) * t.pow(x - muX, 2) + t.pow(sigY, 2) * t.pow(y - muY,
                                                                                2) - 2 * rho * t.pow(
                sigX, 1) * t.pow(sigY, 1) * (x - muX) * (y - muY)) - t.log(sigX * sigY * ohr) + 1.8379
            acc[:, :, 0] = out
            acc = acc * op_mask[:, :, 0:1]
            loss = t.sum(acc[:, :, 0]) / t.sum(op_mask[:, :, 0])
            lossVal = t.sum(acc[:, :, 0], dim=1)
            counts = t.sum(op_mask[:, :, 0], dim=1)
            return lossVal, counts, loss

    def draw(self, hist, fut, nbrs, mask, fut_pred, train_flag, lon_man, lat_man, op_mask, indices):
        hist = hist.cpu()
        fut = fut.cpu()
        nbrs = nbrs.cpu()
        mask = mask.cpu()
        op_mask = op_mask.cpu()
        IPL = 0
        
        for i in range(hist.size(1)):
            lon_man_i = lon_man[i].item()
            lat_man_i = lat_man[i].item()
            plt.axis('on')
            plt.ylim(-18 * self.scale, 18 * self.scale)
            plt.xlim(-180 * self.scale * self.prop, 180 * self.scale * self.prop)
            plt.figure(dpi=300)
            
            IPL_i = mask[i, :, :, :].sum().sum()
            IPL_i = int((IPL_i / args['encoder_size']).item()) # Updated to use args['encoder_size']
            for ii in range(IPL_i):
                plt.plot(nbrs[:, IPL + ii, 1] * self.scale * self.prop, nbrs[:, IPL + ii, 0] * self.scale, ':',
                         color='blue',
                         linewidth=0.5)
            IPL = IPL + IPL_i
            plt.plot(hist[:, i, 1] * self.scale * self.prop, hist[:, i, 0] * self.scale, ':', color='red',
                     linewidth=0.5)
            plt.plot(fut[:, i, 1] * self.scale * self.prop, fut[:, i, 0] * self.scale, '-', color='black',
                     linewidth=0.5)
            
            if train_flag:
                fut_pred = fut_pred.detach().cpu()
                plt.plot(fut_pred[:, i, 1] * self.scale * self.prop, fut_pred[:, i, 0] * self.scale, color='green',
                         linewidth=0.2)
            else:
                for j in range(len(fut_pred)):
                    fut_pred_i = fut_pred[j].detach().cpu()
                    if j == indices[i].item():
                        plt.plot(fut_pred_i[:, i, 1] * self.scale * self.prop, fut_pred_i[:, i, 0] * self.scale,
                                 color='red', linewidth=0.2)
                    else:
                        plt.plot(fut_pred_i[:, i, 1] * self.scale * self.prop, fut_pred_i[:, i, 0] * self.scale,
                                 color='green', linewidth=0.2)
            
            plt.gca().set_aspect('equal', adjustable='box')
            pic_dir = os.path.join(args['path'], 'pic', str(lon_man_i + 1) + '_' + str(lat_man_i + 1))
            os.makedirs(pic_dir, exist_ok=True)
            plt.savefig(pic_dir + '/' + str(self.op) + '.png')
            self.op += 1
            plt.close()

    def main(self, name, val):
        # args['train_flag'] = False 
        
        l_path = args['path']
        generator = model.Generator(args=args)
        gdEncoder = model.GDEncoder(args=args)
        
        if name == "best":
            model_path_g = os.path.join(l_path, 'models/best/epochbest_g.tar')
            model_path_gd = os.path.join(l_path, 'models/best/epochbest_gd.tar')
        else:
            model_path_g = os.path.join(l_path, 'models/trained', f'epoch{name}_g.tar')
            model_path_gd = os.path.join(l_path, 'models/trained', f'epoch{name}_gd.tar')

        print(f"Loading models from: {model_path_g}")
        generator.load_state_dict(t.load(model_path_g, map_location=device))
        gdEncoder.load_state_dict(t.load(model_path_gd, map_location=device))
        
        generator = generator.to(device)
        gdEncoder = gdEncoder.to(device)
        generator.eval()
        gdEncoder.eval()

        if args['dataset'] == "ngsim":
            test_dataset = lo.NgsimDataset(args['test_set'], enc_size=args['encoder_size'])
        else:
            test_dataset = lo.HighdDataset(args['highd_test_set'], enc_size=args['encoder_size'])

        valDataloader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, 
                                   num_workers=args['num_worker'], collate_fn=test_dataset.collate_fn)

        lossVals = t.zeros(args['out_length']).to(device)
        counts = t.zeros(args['out_length']).to(device)
        avg_val_loss = 0
        
        print("Starting evaluation...")
        with t.no_grad():
            for idx, data in enumerate(tqdm(valDataloader)):
                hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, va, nbrsva, lane, nbrslane, dis, nbrsdis, cls, nbrscls, map_positions = data
                hist = hist.to(device)
                nbrs = nbrs.to(device)
                mask = mask.to(device)
                lat_enc = lat_enc.to(device)
                lon_enc = lon_enc.to(device)
                fut = fut[:args['out_length'], :, :].to(device)
                op_mask = op_mask[:args['out_length'], :, :].to(device)
                va = va.to(device)
                nbrsva = nbrsva.to(device)
                lane = lane.to(device)
                nbrslane = nbrslane.to(device)
                cls = cls.to(device)
                nbrscls = nbrscls.to(device)

                values = gdEncoder(hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls)
                fut_pred, lat_pred, lon_pred = generator(values, lat_enc, lon_enc)

                if not args.get('train_flag', True): # Actually args is a dict, train_flag defaults to True in model if not set, but we are evaluating
                    indices = []
                    if args['val_use_mse']:
                        fut_pred_max = t.zeros_like(fut_pred[0])
                        for k in range(lat_pred.shape[0]):
                            lat_man = t.argmax(lat_enc[k, :]).detach()
                            lon_man = t.argmax(lon_enc[k, :]).detach()
                            index = lon_man * 3 + lat_man
                            indices.append(index)
                            fut_pred_max[:, k, :] = fut_pred[index][:, k, :]
                        l, c, loss = self.maskedMSETest(fut_pred_max, fut, op_mask)
                    else:
                        l, c, loss = self.maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,
                                                        use_maneuvers=args['use_maneuvers'])
                    
                    if self.drawImg and idx < 10: # Limit drawing to first few batches
                        lat_man = t.argmax(lat_enc, dim=-1).detach()
                        lon_man = t.argmax(lon_enc, dim=-1).detach()
                        self.draw(hist, fut, nbrs, mask, fut_pred, False, lon_man, lat_man, op_mask, indices)
                else:
                    # Fallback if train_flag is somehow True (shouldn't be for eval)
                     if args['val_use_mse']:
                        l, c, loss = self.maskedMSETest(fut_pred, fut, op_mask)
                     else:
                        l, c, loss = self.maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,
                                                        use_maneuvers=args['use_maneuvers'])

                lossVals += l.detach()
                counts += c.detach()
                avg_val_loss += loss.item()

        if args['val_use_mse']:
            print('RMSE (m):', t.pow(lossVals / counts, 0.5) * 0.3048)
        else:
            print('NLL:', lossVals / counts)

if __name__ == '__main__':
    evaluate = Evaluate(drawImg=False)
    evaluate.main(name="best", val=False)
