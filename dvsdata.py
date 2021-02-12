import gzip
import io
import multiprocessing
import pathlib
import subprocess

import torch
from torchvision.datasets.utils import download_and_extract_archive

class DVSNRPDataset(torch.utils.data.Dataset):
    """
    An event-based vision dataset generated using the Neurorobotics Platform (NRP).
    The dataset features 512x512 event-based for 2000 timesteps (shape 2000, 512, 512, 3)
    along with image labels (shape 2000, 512, 512) containing a single integer classifying
    9 different classes:
      0: Nothing
      1: Table
      2: Human
      3: Hammer
      4: Spanner
      5: Screwdriver
      6: Sphere
      7: Box
      8: Cylinder

    Parameters:
        root (str): The root folder to read from/download the data to.
        window_lenght (int): The number of timesteps to break the data into. If set to 0, one
            file = one window = 2000 timesteps. Defaults to 0.
        scale_factor (float): The scaling factor of the x,y dimensions. Defaults to 1.
        download (bool): Should the data be downloaded if it does not exist? Defaults to False.
    """
    
    FILE_LENGTH_TOTAL = 1992
    URL = "https://kth.box.com/shared/static/53aj33uu4j5x1yer8wmmem5p2f2lng8e.gz"
    MD5 = "9f16bc54ae1f7b09bcbbe89625c53f80"

    def __init__(self, root, window_length=0, scale_factor=1, download=False):
        super(DVSNRPDataset, self).__init__()
        self.root = pathlib.Path(root)
        assert self.root.is_dir(), "Given root folder is not a folder"
        if not self.verify() and download:
            self.download()

        self.dat_files = [path.absolute() for path in self.root.glob('*.dat')]
        self.files = [path.absolute().parent / f"{path.stem}.pt" for path in self.root.glob('*.dat')]

        self.scale_factor = scale_factor
        self.window_length = window_length
        if window_length == 0:
            self.windows_per_file = 1
            self.length = len(self.files)
        else:
            self.windows_per_file = self.FILE_LENGTH_TOTAL // window_length
            self.length = len(self.files) * self.windows_per_file

        
    def __getitem__(self, index):
        if self.window_length == 0:
            data, labels = torch.load(self.files[index])
        else:
            file_index = index // self.windows_per_file
            data, labels = torch.load(self.files[file_index])
            file_offset = (index % self.windows_per_file) * self.window_length
            data = data[file_offset:file_offset + self.window_length]
            labels = labels[file_offset:file_offset + self.window_length]
        return self.scale(data), self.scale(labels).long()

    def _extract_file(self, file_path):
        file_target = f"{file_path.parent}/{file_path.stem}.pt"
        with open(file_target, 'wb') as fp:
            gzcat = subprocess.Popen(["gunzip", "-c", file_path], stdout=subprocess.PIPE)
            gzbytes = io.BytesIO(gzcat.communicate()[0])
            fp.write(gzbytes.getbuffer())

    def download(self):
        # Step 1: Download
        download_and_extract_archive(url=self.URL, root=self.root, md5=self.MD5)
        # Step 2: Extract
        with multiprocessing.Pool() as pool:
            pool.map(self._extract_file, self.dat_files)

    def scale(self, data):
        is_label = len(data.shape) < 4
        # We need to re-size the data to prepare interpolation
        if is_label: # Labels
            image = data.view(1, *data.shape)
        else:                   # Inputs
            # Before:  T, H, W, C
            # We need: T, C, H, W
            image = data.view(*data.shape).permute(0, 3, 1, 2).float()
        scaled = torch.nn.functional.interpolate(image, scale_factor=self.scale_factor)
        return (scaled.view(*scaled.shape[1:]) if is_label else scaled)
    
    def verify(self):
        pt_files = list(self.root.glob("*.pt"))
        for f in pt_files:
            if not f.exists():
                return False
        return len(pt_files) > 0

    def __len__(self):
        return self.length

if __name__ == "__main__":
    d = DVSNRPDataset(root='/home/jens/scenes_hs/', download=True)
    x, y = d[0]
    print(x.shape, y.shape)