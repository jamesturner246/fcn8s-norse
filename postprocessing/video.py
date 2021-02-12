import torch
import pytorch_lightning as pl
import h5py

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import norse_dvs

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def extract_scene(scene_id, filename):
    with h5py.File(filename, "r") as f:
        # List all groups
        print(f"Keys: {f.keys()}, scene: {scene_id}")
        key_list = list(f.keys())
        dvs = f[key_list[0]]
        lbl = f[key_list[4]]
        rgb = f[key_list[6]]
        
        return dvs[scene_id], lbl[scene_id], rgb[scene_id]

def show_animation(r, interval=50):
    fig = plt.figure()
    ims = []
    for i in range(len(r)):
        im = plt.imshow(r[i], animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True,
                                    repeat_delay=1000)

    return ani

def save_animation(r, filename):
    fig = plt.figure()
    ims = []
    for i in range(len(r)):
        im = plt.imshow(r[i], animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)

    ani.save(filename)

def load_model(model_file):
    m = torch.load(model_file)
    weights = m['state_dict']['loss_fn.weight']
    return norse_dvs.DVSModel.load_from_checkpoint(model_file, n_class=9, n_channels=2, height=256, width=256, iter_per_frame=2, class_weights=weights)

def train_scenes(dvs, model):
    #x = torch.tensor(dvs[:,:,:,0:2]).view(1000, 2, 512, 512)
    #x = torch.nn.functional.interpolate(x, scale_factor=0.5)
    #x = x.view(1, -1, 2, 512, 512).cuda()
    model.freeze()
    with torch.no_grad():
        model = model.cuda()
        return x, model(x)

if __name__ == "__main__":
    # Labels
    # 0 (Nothing)
    # 1 (table)
    # 2 (Human): 'eric_armL', 'eric_armR', 'eric_calfL', 'eric_calfR', 'eric_footL', 'eric_footR', 'eric_forearmL', 'eric_forearmR', 'eric_handL', 'eric_handR', 'eric_head', 'eric_legL', 'eric_legR', 'eric_torso', 
    # 3-8 (tool): 'hammer' (3), 'spanner' (4), 'screwdriver' (5), 'sphere' (6), 'box' (7), 'cylinder' (8)
    class_lbl = 3
    scene_id = 0
    
    filename = "high_speed2.h5"
    dvs, labels, rgb = extract_scene(scene_id, filename)
    save_animation(rgb, 'out.mp4')
    model = load_model("model_hs.ckpt")
    x, y = train_scenes(dvs, model)
    torch.save((x, y), 'trained.dat')
    print(x.shape, y.shape)

