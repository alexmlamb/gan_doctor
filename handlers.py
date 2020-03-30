import torch
import pickle
import os
import numpy as np
from torchvision.utils import save_image
from utils import (compute_fid,
                   compute_is,
                   compute_fid_tf,
                   compute_is_tf)

def is_handler(gan,
               n_samples,
               batch_size,
               use_tf=False,
               eval_every=5):
    '''Callback for computing Inception Score.

    Params
    ------

    '''
    def _is_handler(losses, batch, outputs, kwargs):
        if kwargs['iter'] == 1 and \
           kwargs['mode'] == 'train' and \
           kwargs['epoch'] % eval_every == 0:
            gan._eval()
            use_tf = False
            if not use_tf:
                print('no tf inception score')
                with torch.no_grad():
                    return compute_is(n_samples=n_samples,
                                      gan=gan,
                                      batch_size=batch_size)
            else:
                print('tf inception score')
                return compute_is_tf(n_samples=n_samples,
                                     gan=gan,
                                     batch_size=batch_size)
        return {}
    return _is_handler

def fid_handler(gan,
                cls,
                loader,
                n_samples,
                batch_size=None,
                use_tf=False,
                eval_every=5):
    '''Callback for computing FID score.

    Params
    ------

    '''

    #if batch_size is None:
    #    batch_size = loader.batch_size
    # Extract the loader here...

    # Extract `n_samples` from the training set
    # here.
    print('computing fid!')
    real_samples = []
    for b, (x_batch, _) in enumerate(loader):
        real_samples.append(x_batch.numpy())
    real_samples = np.vstack(real_samples)

    # Must be between 0 and 1.
    real_samples = real_samples*0.5 + 0.5

    def _fid_handler(losses, batch, outputs, kwargs):
        if kwargs['iter'] == 1 and \
           kwargs['mode'] == 'train' and \
           kwargs['epoch'] % eval_every == 0:
            print('fid handler!')
            use_tf = False
            gan._eval()
            if not use_tf:
                with torch.no_grad():
                    return compute_fid(train_samples=real_samples,
                                       n_gan_samples=n_samples,
                                       gan=gan,
                                       batch_size=batch_size,
                                       cls=cls)
            else:
                return compute_fid_tf(n_gan_samples=n_samples,
                                      gan=gan,
                                      batch_size=batch_size)
        return {}
    return _fid_handler

def dump_img_handler(gan,
                     dest_dir,
                     batch_size,
                     eval_every=1):
    def _dump_img_handler(losses, batch, outputs, kwargs):
        if kwargs['iter'] == 1 and \
           kwargs['mode'] == 'train' and \
           kwargs['epoch'] % eval_every == 0:
            gan._eval()
            with torch.no_grad():
                gen_imgs = gan.sample(batch_size)
                if type(gen_imgs) == list:
                    # Special case to handle MSG-GAN
                    for j in range(len(gen_imgs)):
                        save_image(gen_imgs[j]*0.5 + 0.5,
                                   "%s/%i_res%i.png" % (dest_dir, kwargs['epoch'], j),
                                   nrow=20)
                else:
                    save_image(gen_imgs*0.5 + 0.5,
                               "%s/%i.png" % (dest_dir, kwargs['epoch']),
                               nrow=20)
        return {}
    return _dump_img_handler
