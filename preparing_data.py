import shutil
from pathlib import Path
import numpy as np
from matplotlib import image
from PIL import Image
from matplotlib import pyplot as plt
from scipy import ndimage

def augment_data(src, folder_name):
    # first we copy the data that we want to augment
    dest = 'augmented_data/'+ folder_name
    shutil.copytree(src,dest)

    p = Path(dest)
    _iterfile(p)    

def _modification(name='rot',num=1):
    if name == 'rot':
        return lambda img: (np.rot90(img,num),'rot'+str(num))
    elif name == 'flip':
        return lambda img: (np.flip(img,num),'flip'+str(num))

modifications = [_modification(name='rot',num=1),
                _modification(name='rot',num=2),
                _modification(name='rot',num=3),
                _modification(name='flip',num=0),
                _modification(name='flip',num=1)]


def _iterfile(path:Path):
    for i in path.iterdir():
        if i.is_dir():
            _iterfile(i)
        elif i.is_file():
            img = Image.open(str(i))
            mod_elem = []
            mod_elem.append(ndimage.rotate(img,90))
            mod_elem.append(ndimage.rotate(img,180))
            mod_elem.append(ndimage.rotate(img,270))
            mod_elem.append(np.fliplr(img))
            mod_elem.append(np.flipud(img))
            mod_elem.append(np.fliplr(mod_elem[0]))
            mod_elem.append(np.flipud(mod_elem[0]))
            
            for mod,name in zip(mod_elem,['rot90','rot180','rot270','flip1','flip2','flip3','flip4']):
                file_path = str(path)+'/'+ i.name[:-4] + name+'.png'
                plt.imsave(file_path,mod)
                
#             for mod in modifications:
#                 copy,name = mod(img)
#                 mod_elem.append(copy)
#                 file_path = str(path)+'/'+ i.name[:-4] + name+'.png'
#                 print(file_path)
#                 plt.imsave(file_path,copy)

