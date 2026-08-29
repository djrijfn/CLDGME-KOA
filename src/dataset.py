import os 
from torch.utils.data import Dataset 
from PIL import Image 
 
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff') 
CLASS_MAP = {'0':0, '1':1, '2':2, '3':3, '4':4, 'normal':0, 'doubtful':1, 'minimal':2, 'mild':2, 'moderate':3, 'severe':4} 
SPLIT_ALIASES = {'train':['train'], 'val':['val', 'valid', 'validation'], 'test':['test']} 
 
def _find_split_root(image_root, split): 
    for name in SPLIT_ALIASES[split]: 
        candidate = os.path.join(image_root, name) 
        if os.path.isdir(candidate): 
            return candidate 
    raise FileNotFoundError(f'split folder not found for {split} under {image_root}') 
 
def _class_id_from_name(name): 
    key = name.strip().lower() 
    if key in CLASS_MAP: 
        return CLASS_MAP[key] 
    raise ValueError(f'unsupported class folder: {name}') 
 
class KoaFolderDataset(Dataset): 
    def __init__(self, image_root, split, transform=None): 
        self.transform = transform 
        root = _find_split_root(image_root, split) 
        self.samples = [] 
        for class_name in os.listdir(root): 
            class_dir = os.path.join(root, class_name) 
            if not os.path.isdir(class_dir): 
                continue 
            class_id = _class_id_from_name(class_name) 
            for base, _, files in os.walk(class_dir): 
                for fname in files: 
                    if fname.lower().endswith(IMG_EXTS): 
                        self.samples.append((os.path.join(base, fname), class_id)) 
        if len(self.samples) == 0: 
            raise RuntimeError(f'no images found in {root}') 
        self.labels = [x[1] for x in self.samples] 
 
    def __len__(self): 
        return len(self.samples) 
 
    def __getitem__(self, idx): 
        path, label = self.samples[idx] 
        image = Image.open(path).convert('RGB') 
        if self.transform is not None: 
            image = self.transform(image) 
        return image, label, path 
 
    def get_labels(self): 
        return self.labels
