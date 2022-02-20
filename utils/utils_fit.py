import torch
from tqdm import tqdm

from utils.utils import get_lr
from config import depth_img
        
def fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda):
    loss        = 0
    val_loss    = 0
    ll_loss = 0
    da_loss = 0
    ll_val_loss = 0
    da_val_loss = 0

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            if depth_img:
                images, targets, ll_gt, da_gt, Dimages = batch[0], batch[1], batch[2], batch[3], batch[4]
            else:
                images, targets, ll_gt, da_gt = batch[0], batch[1], batch[2], batch[3]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                    ll_gt = torch.from_numpy(ll_gt).type(torch.FloatTensor).cuda()
                    da_gt = torch.from_numpy(da_gt).type(torch.FloatTensor).cuda()
                    if depth_img:
                        Dimages = torch.from_numpy(Dimages).type(torch.FloatTensor).cuda()
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                    ll_gt = torch.from_numpy(ll_gt).type(torch.FloatTensor)
                    da_gt = torch.from_numpy(da_gt).type(torch.FloatTensor)
                    if depth_img:
                        Dimages = torch.from_numpy(Dimages).type(torch.FloatTensor).cuda()
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            if depth_img:
                outputs = model_train(images,Dimages)
            else:
                outputs         = model_train(images)

            loss_value_all  = 0
            num_pos_all     = 0
            ll_loss_item = 0
            da_loss_item = 0
            #----------------------#
            #   计算损失
            #----------------------#
            # print('len_outputs',len(outputs))
            for l in range(len(outputs)):
                if l == 3:  # ll_output
                    loss_item, num_pos = yolo_loss(l, outputs[l], None, ll_gt, None)
                    loss_value += loss_item
                    ll_loss_item = loss_item.item()
                elif l == 4: # da_output
                    loss_item, num_pos = yolo_loss(l, outputs[l], None, None, da_gt)
                    loss_value += loss_item
                    da_loss_item = loss_item.item()
                else:
                    loss_item, num_pos = yolo_loss(l, outputs[l], targets, None, None)
                    loss_value_all  += loss_item
                    num_pos_all     += num_pos
                    if l == 2:
                        loss_value = loss_value_all / num_pos_all
                        # print(loss_value)

            #----------------------#
            #   反向传播
            #----------------------#
            loss_value.backward()
            optimizer.step()

            loss += loss_value.item()
            ll_loss += ll_loss_item
            da_loss += da_loss_item
            
            pbar.set_postfix(**{'det_loss'  : (loss-ll_loss-da_loss) / (iteration + 1),
                                'll_loss': ll_loss / (iteration + 1),
                                'da_loss': da_loss / (iteration + 1),
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break

            if depth_img:
                images, targets, Dimages = batch[0], batch[1], batch[4]
            else:
                images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                    if depth_img:
                        Dimages = torch.from_numpy(Dimages).type(torch.FloatTensor).cuda()
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                    if depth_img:
                        Dimages = torch.from_numpy(Dimages).type(torch.FloatTensor).cuda()

                #----------------------#
                #   清零梯度
                #----------------------#
                optimizer.zero_grad()
                #----------------------#
                #   前向传播
                #----------------------#
                if depth_img:
                    outputs = model_train(images, Dimages)
                else:
                    outputs = model_train(images)

                loss_value_all  = 0
                num_pos_all     = 0
                ll_loss_item = 0
                da_loss_item = 0
                #----------------------#
                #   计算损失
                #----------------------#
                for l in range(len(outputs)):
                    if l == 3:  # ll_output
                        loss_item, num_pos = yolo_loss(l, outputs[l], None, ll_gt, None)
                        loss_value += loss_item
                        ll_loss_item = loss_item.item()
                    elif l == 4:  # da_output
                        loss_item, num_pos = yolo_loss(l, outputs[l], None, None, da_gt)
                        loss_value += loss_item
                        da_loss_item = loss_item.item()
                    else:
                        loss_item, num_pos = yolo_loss(l, outputs[l], targets, None, None)
                        loss_value_all += loss_item
                        num_pos_all += num_pos
                        if l == 2:
                            loss_value = loss_value_all / num_pos_all
                            # print(loss_value)

            val_loss += loss_value.item()
            ll_val_loss += ll_loss_item
            da_val_loss += da_loss_item

            pbar.set_postfix(**{'val_loss': (val_loss-ll_val_loss-da_val_loss) / (iteration + 1),
                                'll_loss': ll_val_loss / (iteration + 1),
                                'da_loss': da_val_loss / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    
    loss_history.append_loss(loss / epoch_step, val_loss / epoch_step_val)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    torch.save(model.state_dict(), 'G:\yoloR\logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))
