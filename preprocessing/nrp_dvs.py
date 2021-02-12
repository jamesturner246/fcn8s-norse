import h5py
import torch

import os

if __name__ == '__main__':
    if len(os.argv) > 1:
        filename = os.argv[1]
    else:
        filename = "dvs_60.h5"

    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        key_list = list(f.keys())
    
        # Get the data
        dvs = f[key_list[0]]
        lbl = f[key_list[4]]
        rgb = f[key_list[6]]
        
        scenes = []
        i = 0
        for events, labels in zip(dvs, lbl):
            x = torch.as_tensor(events[6:])
            # Remove third unused dimension
            x = x[:,:,:,0:2] 
            y = torch.as_tensor(labels[6:])
            # A bug shows weird labels in the first column
            y[:,0] = 0 
            # Merge human class to 1
            human_mask = torch.bitwise_and(y > 1, y < 16)         # Identify human 1 < y < 16
            y = torch.where(human_mask, torch.ones_like(y) * 2, y) # Replace human with 2
            y = torch.where(y > 2, y - 13, y)                      # Subtract 13 from tool classes
            y = torch.where(y > 8, torch.zeros_like(y), y)         # Remove weirdly large labels
            scenes.append((x, y))
            if i > 100:
                break
            i += 1
    torch.save(scenes, f'{filename}.dat')
