import os
import logging
import torch as t
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import data_loader as lo
from config import args, device
import model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

def maskedNLL(y_pred, y_gt, mask):
    """
    Masked Negative Log Likelihood loss.
    """
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
    """
    Masked MSE loss.
    """
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
    """
    Cross Entropy Loss.
    """
    value = t.log(t.sum(pred * target, dim=-1))
    return -t.sum(value) / value.shape[0]

def save_model(name, gdEncoder, generator, base_path, save_best=False):
    models_dir = os.path.join(base_path, "models")
    trained_dir = os.path.join(models_dir, "trained")
    best_dir = os.path.join(models_dir, "best")
    
    if save_best:
        save_path_gd = os.path.join(best_dir, f'epoch{name}_gd.tar')
        save_path_g = os.path.join(best_dir, f'epoch{name}_g.tar')
    else:
        save_path_gd = os.path.join(trained_dir, f'epoch{name}_gd.tar')
        save_path_g = os.path.join(trained_dir, f'epoch{name}_g.tar')
        
    t.save(gdEncoder.state_dict(), save_path_gd)
    t.save(generator.state_dict(), save_path_g)
    logger.info("Saved models at epoch %s to %s", name, save_path_gd)

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

            batch_size = fut.shape[1]
            total_loss += loss_g.item() * batch_size
            total_samples += batch_size

    avg_val_loss = total_loss / total_samples if total_samples > 0 else 0
    return avg_val_loss

def main():
    # Directories
    base_path = args["path"]
    log_dir = os.path.join(base_path, "logs")
    tensorboard_dir = os.path.join(base_path, "tensorboard")
    models_dir = os.path.join(base_path, "models")
    trained_dir = os.path.join(models_dir, "trained")
    best_dir = os.path.join(models_dir, "best")
    
    for directory in [base_path, log_dir, tensorboard_dir, models_dir, trained_dir, best_dir]:
        os.makedirs(directory, exist_ok=True)

    # Add file handler to logger now that directories exist
    file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(file_handler)

    logger.info("Training for Model: %s", str(args['model']))
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Initialize models
    gdEncoder = model.GDEncoder(args).to(device)
    generator = model.Generator(args).to(device)
    
    # Optimizers
    optimizer_gd = optim.Adam(gdEncoder.parameters(), lr=args['learning_rate'])
    optimizer_g = optim.Adam(generator.parameters(), lr=args['learning_rate'])
    scheduler_gd = ReduceLROnPlateau(optimizer_gd, mode='min', factor=0.6, patience=2)
    scheduler_g = ReduceLROnPlateau(optimizer_g, mode='min', factor=0.6, patience=2)

    # Load dataset
    if args['dataset'] == "ngsim":
        train_dataset = lo.NgsimDataset(args['train_set'], enc_size=args['encoder_size'])
        val_dataset = lo.NgsimDataset(args['val_set'], enc_size=args['encoder_size'])
    else:
        train_dataset = lo.HighdDataset(args['highd_train_set'], enc_size=args['encoder_size'])
        val_dataset = lo.HighdDataset(args['highd_val_set'], enc_size=args['encoder_size'])
    
    trainDataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True,
                                 num_workers=args['num_worker'], collate_fn=train_dataset.collate_fn)
    valDataloader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False,
                               num_workers=args['num_worker'], collate_fn=val_dataset.collate_fn)

    best_val_loss = float('inf')
    global_step = 0
    patience_counter = 0

    # Check for existing best model to resume (optional)
    # best_gd_path = os.path.join(best_dir, "epochbest_gd.tar")
    # best_g_path = os.path.join(best_dir, "epochbest_g.tar")
    # if os.path.exists(best_gd_path) and os.path.exists(best_g_path):
    #     gdEncoder.load_state_dict(t.load(best_gd_path))
    #     generator.load_state_dict(t.load(best_g_path))
    #     logger.info("Resumed from best model.")

    for epoch in range(args['epoch']):
        gdEncoder.train()
        generator.train()
        epoch_loss = 0.0
        
        logger.info("Epoch: %d, LR: %.6f", epoch+1, optimizer_gd.param_groups[0]['lr'])
        
        progress_bar = tqdm(trainDataloader, desc=f"Epoch {epoch+1}")
        for idx, data in enumerate(progress_bar):

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

            # Forward
            values = gdEncoder(hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls)
            g_out, lat_pred, lon_pred = generator(values, lat_enc, lon_enc)
            
            # Loss calculation
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

            # Backward
            optimizer_g.zero_grad()
            optimizer_gd.zero_grad()
            loss_g.backward()
            
            t.nn.utils.clip_grad_norm_(generator.parameters(), 10)
            t.nn.utils.clip_grad_norm_(gdEncoder.parameters(), 10)
            
            optimizer_g.step()
            optimizer_gd.step()

            epoch_loss += loss_g.item()
            writer.add_scalar('Train/Loss', loss_g.item(), global_step)
            global_step += 1
            
            progress_bar.set_postfix({'loss': loss_g.item()})

        avg_epoch_loss = epoch_loss / len(trainDataloader)
        logger.info("Epoch %d Training Loss: %.4f", epoch+1, avg_epoch_loss)
        
        # Save checkpoint
        save_model(str(epoch+1), gdEncoder, generator, base_path, save_best=False)

        # Validation
        avg_val_loss = validate_model(epoch, gdEncoder, generator, valDataloader)
        writer.add_scalar('Val/Loss', avg_val_loss, epoch+1)
        logger.info("Epoch %d Validation Loss: %.4f", epoch+1, avg_val_loss)

        # Check best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_model('best', gdEncoder, generator, base_path, save_best=True)
            logger.info("New best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= 10:
                logger.info("Early stopping triggered.")
                break

        scheduler_gd.step(avg_val_loss)
        scheduler_g.step(avg_val_loss)

    writer.close()

if __name__ == '__main__':
    main()
