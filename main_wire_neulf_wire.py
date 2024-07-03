import torch
import argparse

from nerf.neulf.provider import NeuLFDataset , UVXYDataset , UVSTDataset
from nerf.gui import NeRFGUI
from nerf.utils import *

from functools import partial
from loss import huber_loss
from nerf.neulf.network import NeuLFWireNetwork,NeuLFNetwork,NeuLFWireNetwork2

#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--debug', action='store_true', help="debug mode")
    parser.add_argument('--render_only' , action='store_true', help="render only mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=200000, help="training iters")
    parser.add_argument('--lr', type=float, default=4e-5, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=1, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    ### jw 
    parser.add_argument('--depth', type=int, default=6 )
    parser.add_argument('--width', type=int, default=256 )
    parser.add_argument('--exp_sc', action='store_true')
    parser.add_argument('--jw_test', type=str, default='in')
    parser.add_argument('--whole_epoch', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--loss_coeff', type=int, default=1)

    parser.add_argument('--skip_mode2', action='store_true')
    parser.add_argument('--no_skips', action='store_true')

    parser.add_argument('--neulf', action='store_true')

    parser.add_argument('--LF_mode', type=str, default='vec')


    
    parser.add_argument('--sigma', type=int, default=40 )
    parser.add_argument('--omega', type=int, default=40 )

    


    opt = parser.parse_args()

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True
    
    if opt.patch_size > 1:
        opt.error_map = False # do not use error_map if use patch-based training
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."


    input_dim = 0
    Dataset = None

    if opt.LF_mode == "vec":
        input_dim = 6
        Dataset = NeuLFDataset
    elif opt.LF_mode == "uvxy":
        input_dim = 4
        Dataset = UVXYDataset
    elif opt.LF_mode == "uvst":
        input_dim = 4
        Dataset = UVSTDataset



    # if opt.ff:
    #     opt.fp16 = True
    #     assert opt.bg_radius <= 0, "background model is not implemented for --ff"
    #     from nerf.network_ff import NeRFNetwork
    # elif opt.tcnn:
    #     opt.fp16 = True
    #     assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
    #     from nerf.network_tcnn import NeRFNetwork
    # else:
    #     from nerf.network import NeRFNetwork

    print(opt)
    
    seed_everything(opt.seed)
    if opt.no_skips:
        skips = []
    else:
        skips = [4]

    
    model = NeuLFWireNetwork( num_layers=opt.depth ,hidden_dim=opt.width , input_dim=input_dim, skips=skips ,  sigma = opt.sigma , omega = opt.omega)
    lr = opt.lr
    
    # if opt.skip_mode2:
    #     skips = [3,7,11,15,19]
    # else:
    #     skips = [4,8,12,16,20]


    if opt.neulf:
        
        model = NeuLFNetwork( num_layers=opt.depth ,hidden_dim=opt.width , input_dim=input_dim, skips=skips , sigma = opt.sigma , omega = opt.omega)
        lr = 5e-4
        lr = 5e-04

    print(model)

    criterion = torch.nn.MSELoss(reduction='none')
    #criterion = partial(huber_loss, reduction='none')
    #criterion = torch.nn.HuberLoss(reduction='none', beta=0.1) # only available after torch 1.10 ?

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   

        

    if opt.test:
        
        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            test_loader = Dataset(opt, device=device, type='test').dataloader()

            if test_loader.has_gt:
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
    
            trainer.test(test_loader, write_video=True) # test and save video
            
            #trainer.save_mesh(resolution=256, threshold=10)
    

    elif opt.render_only:
        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            render_loader = Dataset(opt, device=device, type='render').dataloader()

            # if render_loader.has_gt:
            #     trainer.evaluate(render_loader) # blender has gt, so evaluate it.
    
            trainer.test(render_loader, write_video=True , name= "_render_") # test and save video
            
            #trainer.save_mesh(resolution=256, threshold=10)

    else:
        
        optimizer = lambda model: torch.optim.Adam(model.parameters(), lr = lr , betas=(0.9, 0.999))
        train_loader = Dataset(opt, device=device, type='train_neulf').dataloader()

        # decay to 0.1 * init_lr at last iter step
   
        
        
        if opt.neulf:
            scheduler = lambda optimizer: optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995) 
            
        else:    
            scheduler = lambda optimizer: optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995) 
            #scheduler = lambda optimizer: optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97) 
        
        
        #scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))
        logger_path = os.path.join(opt.workspace , 'log')
        logger = PSNRLogger(logger_path,opt.workspace)
        logger.set_metadata("depth",opt.depth)
        logger.set_metadata("width",opt.width)
        logger.set_metadata("LF_mode",opt.LF_mode)
        logger.set_metadata("datadir",opt.path)
        logger.set_metadata("lr",opt.lr)
        dataset_name = opt.path.split('/')[-1]
        logger.set_metadata("dataset_name",dataset_name)


        logger.load_results()

        metrics = [PSNRMeterWithLogger(logger), LPIPSMeter(device=device) ]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=None, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=False, metrics=metrics, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval , loss_coeff=1000)

        if opt.gui: 
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()
        
        else:
            valid_loader = Dataset(opt, device=device, type='val', downscale=1).dataloader()
            test_loader = Dataset(opt, device=device, type='test').dataloader()

            max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            max_epoch = opt.whole_epoch
            trainer.train(train_loader, test_loader, max_epoch)

            logger.save_results()

            # also test
            
            if test_loader.has_gt:
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
            
            trainer.test(test_loader, write_video=True) # test and save video
            
            ##trainer.save_mesh(resolution=256, threshold=10)