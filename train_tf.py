from torch.utils.data import DataLoader
import loader2 as lo
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau  # dynamic LR scheduler
from tqdm import tqdm
import os
import logging
from evaluate5f import Evaluate
from config import args_cmd, args, model, learning_rate, dataset, device, seed

from torch.utils.tensorboard import SummaryWriter
import torch as t

# Define base path and create subdirectories for logs, tensorboard, and models
base_path = args["path"]
log_dir = os.path.join(base_path, "logs")
tensorboard_dir = os.path.join(base_path, "tensorboard")
models_dir = os.path.join(base_path, "models")
trained_dir = os.path.join(models_dir, "trained")
best_dir = os.path.join(models_dir, "best")

for directory in [base_path, log_dir, tensorboard_dir, models_dir, trained_dir, best_dir]:
    os.makedirs(directory, exist_ok=True)

# Set up logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

def maskedNLL(y_pred, y_gt, mask):
    acc = t.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    sigX = y_pred[:, :, 2]
    sigY = y_pred[:, :, 3]
    rho = y_pred[:, :, 4]
    ohr = t.pow(1 - t.pow(rho, 2), -0.5)
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = 0.5 * t.pow(ohr, 2) * (
            t.pow(sigX, 2) * t.pow(x - muX, 2) + t.pow(sigY, 2) * t.pow(y - muY, 2)
            - 2 * rho * sigX * sigY * (x - muX) * (y - muY)
          ) - t.log(sigX * sigY * ohr) + 1.8379
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = t.sum(acc) / t.sum(mask)
    return lossVal

def MSELoss2(g_out, fut, mask):
    acc = t.zeros_like(mask)
    muX = g_out[:, :, 0]
    muY = g_out[:, :, 1]
    x = fut[:, :, 0]
    y = fut[:, :, 1]
    out = t.pow(x - muX, 2) + t.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = t.sum(acc) / t.sum(mask)
    return lossVal

def CELoss(pred, target):
    value = t.log(t.sum(pred * target, dim=-1))
    return -t.sum(value) / value.shape[0]

def save_model(name, gdEncoder, generator, save_best=False):
    if save_best:
        save_path_gd = os.path.join(best_dir, f'epoch{name}_gd.tar')
        save_path_g = os.path.join(best_dir, f'epoch{name}_g.tar')
    else:
        save_path_gd = os.path.join(trained_dir, f'epoch{name}_gd.tar')
        save_path_g = os.path.join(trained_dir, f'epoch{name}_g.tar')
    t.save(gdEncoder.state_dict(), save_path_gd)
    t.save(generator.state_dict(), save_path_g)
    logger.info("Saved models at epoch %s to %s", name, save_path_gd)

# New validation function: compute average loss_g over validation set using same loss calculation as training
def validate_model(epoch, gdEncoder, generator, valDataloader):
    gdEncoder.eval()
    generator.eval()
    total_loss = 0.0
    total_samples = 0
    with t.no_grad():
        for data in valDataloader:
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
            dis = dis.to(device)
            nbrsdis = nbrsdis.to(device)
            map_positions = map_positions.to(device)
            cls = cls.to(device)
            nbrscls = nbrscls.to(device)

            values = gdEncoder(hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls)
            g_out, lat_pred, lon_pred = generator(values, lat_enc, lon_enc)
            
            if args['use_mse']:
                loss_g1 = MSELoss2(g_out, fut, op_mask)
            else:
                if epoch < args['pre_epoch']:
                    loss_g1 = MSELoss2(g_out, fut, op_mask)
                else:
                    loss_g1 = maskedNLL(g_out, fut, op_mask)
            loss_gx_3 = CELoss(lat_pred, lat_enc)
            loss_gx_2 = CELoss(lon_pred, lon_enc)
            loss_gx = loss_gx_3 + loss_gx_2
            loss_g = loss_g1 + args["scale_cross_entropy_loss"] * loss_gx

            # fut is of shape [out_length, batch, features]; use batch size
            batch_size = fut.shape[1]
            total_loss += loss_g.item()
            total_samples += batch_size

    avg_val_loss = total_loss / total_samples if total_samples > 0 else 0

    return avg_val_loss

def main():
    args['train_flag'] = True
    logger.info("Training for Model: %s", str(args['model']))
    # We no longer use Evaluate.main; we use our own validation loop.
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Initialize training models
    gdEncoder = model.GDEncoder(args)
    generator = model.Generator(args)
    gdEncoder = gdEncoder.to(device)
    generator = generator.to(device)
    
    # Load from last best model if available
    best_gd_path = os.path.join(best_dir, "epochbest_gd.tar")
    best_g_path = os.path.join(best_dir, "epochbest_g.tar")
    if os.path.exists(best_gd_path) and os.path.exists(best_g_path):
        gdEncoder.load_state_dict(t.load(best_gd_path))
        generator.load_state_dict(t.load(best_g_path))
        logger.info("Loaded best model from %s and %s", best_gd_path, best_g_path)
    
    gdEncoder.train()
    generator.train()
    # Load training and validation datasets
    if dataset == "ngsim":
        if args['lon_length'] == 3:
            train_dataset = lo.NgsimDataset('data/dataset_t_v_t/TrainSet.mat')
            val_dataset = lo.NgsimDataset('data/dataset_t_v_t/TestSet.mat')
        else:
            train_dataset = lo.NgsimDataset('../data/5feature/TrainSet.mat')
            val_dataset = lo.NgsimDataset('../data/5feature/ValSet.mat')
    else:
        train_dataset = lo.HighdDataset('../data/highD/TrainSet_highd.mat')
        val_dataset = lo.HighdDataset('../data/highD/ValSet_highd.mat')
    
    trainDataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True,
                                 num_workers=args['num_worker'], collate_fn=train_dataset.collate_fn)
    valDataloader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False,
                               num_workers=args['num_worker'], collate_fn=val_dataset.collate_fn)
    optimizer_gd = optim.Adam(gdEncoder.parameters(), lr=learning_rate)
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
    # optimizer_gd = optim.AdamW(gdEncoder.parameters(), lr=learning_rate)
    # optimizer_g = optim.AdamW(generator.parameters(), lr=learning_rate)
    scheduler_gd = ReduceLROnPlateau(optimizer_gd, mode='min', factor=0.6, patience=2, verbose=True)
    scheduler_g = ReduceLROnPlateau(optimizer_g, mode='min', factor=0.6, patience=2, verbose=True)

    best_val_loss = float('inf')
    global_step = 0
    patience_counter = 0  # Counter for early stopping

    for epoch in range(args['epoch']):
        logger.info("Epoch: %d, current LR: Encoder %.6f, Generator %.6f", 
                    epoch+1, optimizer_gd.param_groups[0]['lr'], optimizer_g.param_groups[0]['lr'])
        epoch_loss = 0.0
        loss_gi1 = 0
        loss_gix = 0
        loss_gx_2i = 0
        loss_gx_3i = 0
        for idx, data in enumerate(tqdm(trainDataloader)):
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
            dis = dis.to(device)
            nbrsdis = nbrsdis.to(device)
            map_positions = map_positions.to(device)
            cls = cls.to(device)
            nbrscls = nbrscls.to(device)

            values = gdEncoder(hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls)
            g_out, lat_pred, lon_pred = generator(values, lat_enc, lon_enc)
            
            if args['use_mse']:
                loss_g1 = MSELoss2(g_out, fut, op_mask)
            else:
                if epoch < args['pre_epoch']:
                    loss_g1 = MSELoss2(g_out, fut, op_mask)
                else:
                    loss_g1 = maskedNLL(g_out, fut, op_mask)
            loss_gx_3 = CELoss(lat_pred, lat_enc)
            loss_gx_2 = CELoss(lon_pred, lon_enc)
            loss_gx = loss_gx_3 + loss_gx_2
            loss_g = loss_g1 + args["scale_cross_entropy_loss"] * loss_gx

            optimizer_g.zero_grad()
            optimizer_gd.zero_grad()
            loss_g.backward()
            a = t.nn.utils.clip_grad_norm_(generator.parameters(), 10)
            a = t.nn.utils.clip_grad_norm_(gdEncoder.parameters(), 10)
            optimizer_g.step()
            optimizer_gd.step()

            epoch_loss += loss_g.item()
            writer.add_scalar('Train/Loss', loss_g.item(), global_step)
            global_step += 1

            loss_gi1 += loss_g1.item()
            loss_gx_2i += loss_gx_2.item()
            loss_gx_3i += loss_gx_3.item()
            loss_gix += loss_gx.item()
            if idx % 10000 == 9999:
                logger.info("mse: %.4f | loss_gx_2: %.4f | loss_gx_3: %.4f",
                            loss_gi1 / 10000, loss_gx_2i / 10000, loss_gx_3i / 10000)
                logger.info("Epoch %d, Step %d: Loss=%.4f", epoch+1, idx, epoch_loss / 10000)
                loss_gi1 = 0
                loss_gix = 0
                loss_gx_2i = 0
                loss_gx_3i = 0
                epoch_loss = 0.0

        # Save a regular checkpoint for this epoch
        save_model(name=str(epoch+1), gdEncoder=gdEncoder, generator=generator, save_best=False)

        # Validation: compute average loss_g on the validation set using our loss calculation
        avg_val_loss = validate_model(epoch, gdEncoder, generator, valDataloader)
        gdEncoder.train()
        generator.train()

        writer.add_scalar('Val/Loss', avg_val_loss, epoch+1)
        logger.info("Epoch %d: Validation Loss = %.4f", epoch+1, avg_val_loss)

        # Early stopping and best model saving based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_model(name='best', gdEncoder=gdEncoder, generator=generator, save_best=True)
            logger.info("New best model at epoch %d with val_loss: %.4f", epoch+1, avg_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= 10:
                logger.info("Early stopping: no improvement in validation loss for 10 consecutive epochs.")
                break

        # Step LR schedulers based on validation loss
        scheduler_gd.step(avg_val_loss)
        scheduler_g.step(avg_val_loss)
        writer.add_scalar('LR/Encoder', optimizer_gd.param_groups[0]['lr'], epoch+1)
        writer.add_scalar('LR/Generator', optimizer_g.param_groups[0]['lr'], epoch+1)

    writer.close()

if __name__ == '__main__':
    main()
