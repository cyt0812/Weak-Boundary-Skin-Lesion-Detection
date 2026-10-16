"""ISIC2017 Mask-CN reproduction without the extra 1320 images.

Keeps upstream architectures/crops/losses. Modern AMP, smaller microbatches,
gradient accumulation, seeded aligned augmentation and resumable checkpoints
are explicitly recorded adaptations. Test labels are used only for final evaluation.
"""
import argparse
import ast
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from net.models import deeplabv3plus, Xception_dilation
from net.loss import dice_loss

OUT = ROOT / 'outputs' / 'isic2017_no_extra'
OUT.mkdir(parents=True, exist_ok=True)
DL = ROOT / 'downloads'
DL.mkdir(exist_ok=True)
DEVICE = torch.device('cuda')
CLASS_NAMES = ['nevus', 'melanoma', 'seborrheic_keratosis']
SPECS = {
    'train': ('Training', 'ISIC-2017_Training_Data.zip', 6229496702),
    'val': ('Validation', 'ISIC-2017_Validation_Data.zip', 920895119),
    'test': ('Test_v2', 'ISIC-2017_Test_v2_Data.zip', 5826753548),
}

def status(stage, **extra):
    data = {'stage': stage, 'time': time.strftime('%Y-%m-%d %H:%M:%S'), **extra}
    tmp = OUT / 'status.tmp'
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    tmp.replace(OUT / 'status.json')
    print(json.dumps(data), flush=True)

def download(url, path, expected=None):
    valid = path.exists() and (path.stat().st_size == expected if expected else zipfile.is_zipfile(path))
    if valid:
        return
    status('downloading', file=path.name, expected_bytes=expected)
    for attempt in range(20):
        with (DL / (path.name + '.download.log')).open('a') as log:
            completed = subprocess.run(['curl.exe', '-f', '-L', '--connect-timeout', '30',
                                       '--speed-limit', '100', '--speed-time', '120',
                                       '--retry', '5', '--retry-delay', '10', '--continue-at', '-',
                                       '-o', str(path), url], stdout=log, stderr=log)
        if completed.returncode == 0:
            break
        status('download_retry', file=path.name, attempt=attempt+1, exit_code=completed.returncode)
        time.sleep(min(60, 10*(attempt+1)))
    else:
        raise RuntimeError(f'Download failed after retries: {path.name}')
    if expected and path.stat().st_size != expected:
        raise RuntimeError(f'Incomplete download: {path}')

def atomic_save(obj, path):
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)

def labels(split):
    prefix = SPECS[split][0]
    path = ROOT / 'dataset' / 'data' / f'ISIC-2017_{prefix}_Part3_GroundTruth.csv'
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    return [(r['image_id'], 1 if float(r['melanoma']) else 2 if float(r['seborrheic_keratosis']) else 0) for r in rows]

def prepare_raw():
    for split, (_, name, size) in SPECS.items():
        path = DL / name
        download('https://isic-archive.s3.amazonaws.com/challenges/2017/' + name, path, size)
        target = ROOT / 'dataset' / 'raw' / split
        marker = target / '.extracted'
        if not marker.exists():
            status('extracting', split=split)
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path) as z:
                for info in z.infolist():
                    if info.is_dir():
                        continue
                    resolved = (target / info.filename).resolve()
                    if not resolved.is_relative_to(target.resolve()):
                        raise RuntimeError('Unsafe archive member')
                z.extractall(target)
            marker.touch()

def image_map(split):
    result = {}
    for p in (ROOT / 'dataset' / 'raw' / split).rglob('*.jpg'):
        if not p.stem.endswith('_superpixels'):
            if p.stem in result:
                raise RuntimeError('Duplicate image id')
            result[p.stem] = p
    needed = {name for name, _ in labels(split)}
    missing = needed - result.keys()
    if missing:
        raise RuntimeError(f'Missing {split} raw images: {len(missing)}')
    return result

def prepare_gt():
    # Existing 256x256 annotation masks are not used for classification inputs.
    for split, prefix in [('train', 'Training'), ('val', 'Validation')]:
        name = f'ISIC-2017_{prefix}_Part1_GroundTruth.zip'
        path = DL / name
        if not path.exists():
            download('https://isic-archive.s3.amazonaws.com/challenges/2017/' + name, path)
        target = ROOT / 'dataset' / 'raw' / split
        marker = target / '.gt_extracted'
        if not marker.exists():
            with zipfile.ZipFile(path) as z:
                for info in z.infolist():
                    if not (target / info.filename).resolve().is_relative_to(target.resolve()):
                        raise RuntimeError('Unsafe ground-truth archive')
                z.extractall(target)
            marker.touch()

class SegDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.images = image_map(split)
        self.ids = [n for n, _ in labels(split)]
        self.masks = {p.name.replace('_segmentation.png', ''): p for p in (ROOT/'dataset'/'raw'/split).rglob('*_segmentation.png')}
        if set(self.ids) - self.masks.keys():
            raise RuntimeError('Segmentation annotations incomplete')
        self.aug = transforms.Compose([
            transforms.RandomAffine(10, translate=(0, .1), scale=(.9, 1.1), shear=5.729578),
            transforms.RandomVerticalFlip(), transforms.RandomHorizontalFlip()])
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, i):
        n = self.ids[i]
        im = Image.open(self.images[n]).convert('RGB')
        m = Image.open(self.masks[n]).convert('L')
        if self.split == 'train' and random.random() < .5:
            w, h = im.size
            candidates = list(range(30, int((w+h)/8), 30))
            if candidates:
                p = random.choice(candidates)
                box = (p, p, w-p, h-p)
                im, m = im.crop(box), m.crop(box)
        im = im.resize((224,224), Image.Resampling.BICUBIC)
        m = m.resize((224,224), Image.Resampling.NEAREST)
        if self.split == 'train':
            rng = torch.get_rng_state()
            im = self.aug(im)
            torch.set_rng_state(rng)
            m = self.aug(m)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32).copy()/255).permute(2,0,1)
        y = torch.from_numpy((np.asarray(m)>0).astype(np.float32))
        return x, y

def seg_loss(logits, target):
    p = logits.float().softmax(1)
    d = (dice_loss(p[:,0], 1-target).mean()+dice_loss(p[:,1],target).mean())/2
    flat, gt = p[:,1].flatten(1), target.flatten(1)
    neg = (flat*(1-gt)).topk(30,dim=1).values
    pos = 1-((1-flat)*gt).topk(30,dim=1).values
    rank = (neg[:,:,None]-pos[:,None,:]+.3).clamp_min(0).mean()
    return d+.05*rank

def load_pretrained(model, path, fourth_channel=False):
    raw = torch.load(path, map_location='cpu', weights_only=True)
    state = {k.removeprefix('module.'): v for k,v in raw.items()}
    if fourth_channel:
        w = state['conv1.weight']
        state['conv1.weight'] = torch.cat([w,w.mean(1,keepdim=True)],1)
    own = model.state_dict()
    matched = {k:v for k,v in state.items() if k in own and own[k].shape == v.shape}
    if len(matched) < 100:
        raise RuntimeError(f'Pretrained weight match unexpectedly low: {len(matched)}')
    model.load_state_dict(matched, strict=False)
    print('Pretrained tensors matched', len(matched), '/', len(own), flush=True)

def restore_rng(c):
    random.setstate(c['random_state']); np.random.set_state(c['numpy_state'])
    torch.set_rng_state(c['torch_state']); torch.cuda.set_rng_state_all(c['cuda_state'])

def checkpoint(model, opt, scaler, epoch, best):
    return dict(model=model.state_dict(), optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                epoch=epoch, best=best, random_state=random.getstate(), numpy_state=np.random.get_state(),
                torch_state=torch.get_rng_state(), cuda_state=torch.cuda.get_rng_state_all())

def train_seg(epochs):
    if (OUT/'coarse_complete.json').exists():
        return
    status('coarse_segmentation_training', total_epochs=epochs)
    model = deeplabv3plus(num_classes=2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    latest, best_path = OUT/'coarse_latest.pth', OUT/'coarse_best.pth'
    start, best = 0, -1.
    if latest.exists():
        c = torch.load(latest, map_location='cpu', weights_only=False)
        model.load_state_dict(c['model']); opt.load_state_dict(c['optimizer']); scaler.load_state_dict(c['scaler'])
        start,best = c['epoch'],c['best']; restore_rng(c)
    else:
        load_pretrained(model, ROOT/'models'/'deeplabv3plus_xception_VOC2012_epoch46_all.pth')
    train = DataLoader(SegDataset('train'), batch_size=4, shuffle=True, num_workers=0)
    val = DataLoader(SegDataset('val'), batch_size=1, num_workers=0)
    for epoch in range(start, epochs):
        model.train(); total = 0.; opt.zero_grad(set_to_none=True)
        for i,(x,y) in enumerate(train):
            x,y = x.to(DEVICE),y.to(DEVICE)
            lr = 1e-4*(1-(epoch+i/len(train))/epochs)**.9
            for g in opt.param_groups: g['lr'] = lr
            with torch.autocast('cuda', dtype=torch.float16):
                loss = seg_loss(model(x), y)
            # Effective batch 16 via four microbatches.
            group = min(4, len(train)-(i//4)*4)
            scaler.scale(loss/group).backward()
            if (i+1)%4==0 or i+1==len(train):
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            total += loss.item()
            if i%50==0: print(f'coarse epoch {epoch+1}/{epochs} batch {i}/{len(train)} loss {total/(i+1):.5f}',flush=True)
        model.eval(); ious=[]
        with torch.no_grad():
            for x,y in val:
                pred = model(x.to(DEVICE)).argmax(1).bool(); gt=y.to(DEVICE).bool()
                ious.append(((pred & gt).sum()/(pred | gt).sum().clamp_min(1)).item())
        score=float(np.mean(ious))
        if score>best:
            best=score; atomic_save({'model':model.state_dict(),'epoch':epoch+1,'val_iou':score},best_path)
        atomic_save(checkpoint(model,opt,scaler,epoch+1,best),latest)
        status('coarse_segmentation_training', epoch=epoch+1,total_epochs=epochs,val_mean_iou=score,best_iou=best)
    (OUT/'coarse_complete.json').write_text(json.dumps({'epochs':epochs,'best_iou':best}))
    del model,opt; torch.cuda.empty_cache()

def upstream_crop_functions(split):
    file = 'extractPatch_cls_train.py' if split=='train' else 'extractPatch_cls_val_test.py'
    tree = ast.parse((ROOT/'dataset'/file).read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    scope = {'np':np, 'Image':Image, 'img_size':224}
    exec(compile(tree,file,'exec'),scope)
    return scope

def prepare_crops():
    for split in SPECS:
        dest=ROOT/'dataset'/'classification_crops'/split
        manifest=dest/'manifest.csv'
        if manifest.exists(): continue
        status('classification_crops', split=split)
        dest.mkdir(parents=True,exist_ok=True)
        functions=upstream_crop_functions(split); images=image_map(split); rows=[]
        for i,(name,label) in enumerate(labels(split)):
            im=Image.open(images[name]).convert('RGB')
            fn=name+'.png'; im.resize((224,224),Image.Resampling.BICUBIC).save(dest/fn)
            rows.append([fn,label,name])
            crop=functions['data_arg_15' if split=='train' and label!=0 else 'data_arg_9']
            names,labs=crop(*im.size,im,label,str(dest)+os.sep,name+'.jpg')
            rows.extend([fn,int(lab),name] for fn,lab in zip(names,labs))
            if i%100==0: print(split,'cropped',i,flush=True)
        tmp=dest/'manifest.tmp'
        with tmp.open('w',newline='') as f:
            w=csv.writer(f); w.writerow(['file','label','image_id']); w.writerows(rows)
        tmp.replace(manifest)

def generate_masks():
    model=deeplabv3plus(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(OUT/'coarse_best.pth',map_location='cpu',weights_only=True)['model']); model.eval()
    for split in SPECS:
        dest=ROOT/'dataset'/'predicted_masks'/split
        dest.mkdir(parents=True,exist_ok=True)
        status('generating_predicted_masks',split=split)
        folder=ROOT/'dataset'/'classification_crops'/split
        with (folder/'manifest.csv').open() as f: rows=list(csv.DictReader(f))
        with torch.no_grad():
            for i,row in enumerate(rows):
                path=dest/row['file']
                if path.exists(): continue
                im=np.asarray(Image.open(folder/row['file']).convert('RGB'),dtype=np.float32)/255
                x=torch.from_numpy(im.copy()).permute(2,0,1).unsqueeze(0).to(DEVICE)
                pred=model(x).argmax(1)[0].byte().cpu().numpy()*255
                Image.fromarray(pred).save(path)
                if i%500==0: print(split,'masks',i,'/',len(rows),flush=True)
    del model; torch.cuda.empty_cache()

class ClassDataset(Dataset):
    def __init__(self,split):
        self.split=split; self.folder=ROOT/'dataset'/'classification_crops'/split
        self.mask=ROOT/'dataset'/'predicted_masks'/split
        with (self.folder/'manifest.csv').open() as f:self.rows=list(csv.DictReader(f))
        self.aug=transforms.Compose([transforms.RandomAffine(10,translate=(0,.1),scale=(.9,1.1),shear=5.729578),transforms.RandomVerticalFlip(),transforms.RandomHorizontalFlip()])
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; im=Image.open(self.folder/r['file']).convert('RGB'); mask=Image.open(self.mask/r['file']).convert('L')
        if self.split=='train':
            rng=torch.get_rng_state(); im=self.aug(im); torch.set_rng_state(rng); mask=self.aug(mask)
        x=torch.from_numpy(np.asarray(im,dtype=np.float32).copy()/255).permute(2,0,1)
        m=torch.from_numpy((np.asarray(mask)>0).astype(np.float32)).unsqueeze(0)
        return torch.cat([x,m]),int(r['label']),r['image_id']

def evaluate_cls(model, loader):
    model.eval(); scores={}; labels_by_id={}
    with torch.no_grad():
        for x,y,ids in loader:
            p=model(x.to(DEVICE)).softmax(1).cpu().numpy()
            for n,label,prob in zip(ids,y.tolist(),p):
                scores.setdefault(n,[]).append(prob); labels_by_id[n]=label
    ids=list(scores); y=np.array([labels_by_id[n] for n in ids]); p=np.array([np.mean(scores[n],axis=0) for n in ids])
    aucs=[roc_auc_score(y==k,p[:,k]) for k in range(3)]
    return {'images':len(ids),'accuracy':accuracy_score(y,p.argmax(1)), 'macro_f1':f1_score(y,p.argmax(1),labels=[0,1,2],average='macro',zero_division=0), 'auc_per_class':aucs,'selection_auc_mean_melanoma_sk':float(np.mean(aucs[1:])), 'confusion_matrix':confusion_matrix(y,p.argmax(1),labels=[0,1,2]).tolist(), 'classification_report':classification_report(y,p.argmax(1),labels=[0,1,2],target_names=CLASS_NAMES,output_dict=True,zero_division=0)}

def train_cls(steps):
    model=Xception_dilation(input_channel=4,num_classes=3).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=1e-4); scaler=torch.amp.GradScaler('cuda')
    latest,best_path=OUT/'classification_latest.pth',OUT/'classification_best.pth'
    start,best=0,-1.
    if latest.exists():
        c=torch.load(latest,map_location='cpu',weights_only=False)
        model.load_state_dict(c['model']); opt.load_state_dict(c['optimizer']); scaler.load_state_dict(c['scaler']); start,best=c['epoch'],c['best']; restore_rng(c)
    else:load_pretrained(model,ROOT/'models'/'xception-43020ad28.pth',fourth_channel=True)
    loader=DataLoader(ClassDataset('train'),batch_size=8,shuffle=True,num_workers=0,drop_last=True)
    val=DataLoader(ClassDataset('val'),batch_size=8,num_workers=0)
    iterator=iter(loader)
    for step in range(start,steps):
        model.train(); opt.zero_grad(set_to_none=True); loss_sum=0.
        for g in opt.param_groups:g['lr']=1e-4*(1-step/steps)**.9
        for micro in range(4):
            try:x,y,_=next(iterator)
            except StopIteration:iterator=iter(loader); x,y,_=next(iterator)
            with torch.autocast('cuda',dtype=torch.float16):loss=torch.nn.functional.cross_entropy(model(x.to(DEVICE)),y.to(DEVICE))
            scaler.scale(loss/4).backward(); loss_sum+=loss.item()/4
        scaler.step(opt); scaler.update()
        if step%20==0:print('Mask-CN step',step+1,'/',steps,'loss',loss_sum,flush=True)
        if (step>500 and step%100==0) or step+1==steps:
            metrics=evaluate_cls(model,val); score=metrics['selection_auc_mean_melanoma_sk']
            if score>best:
                best=score; atomic_save({'model':model.state_dict(),'step':step+1,'metrics':metrics,'class_names':CLASS_NAMES},best_path)
            atomic_save(checkpoint(model,opt,scaler,step+1,best),latest)
            status('classification_training',step=step+1,total_steps=steps,validation=metrics,best_auc=best)
    model.load_state_dict(torch.load(best_path,map_location='cpu',weights_only=True)['model'])
    result=evaluate_cls(model,DataLoader(ClassDataset('test'),batch_size=8,num_workers=0))
    result.update({'protocol':'ISIC2017 official splits, no extra1320; upstream architectures/crops; modern AMP; effective batches16/32 with microbatch4/8; validation selection by mean MEL/SK AUC', 'coarse_epochs':args.seg_epochs,'classification_steps':steps})
    (OUT/'test_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    status('complete',results=result)

def smoke():
    model=deeplabv3plus(num_classes=2).cuda()
    load_pretrained(model,ROOT/'models'/'deeplabv3plus_xception_VOC2012_epoch46_all.pth')
    opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    scaler=torch.amp.GradScaler('cuda')
    x=torch.rand(4,3,224,224,device='cuda'); y=(torch.rand(4,224,224,device='cuda')>.5).float()
    with torch.autocast('cuda',dtype=torch.float16): loss=seg_loss(model(x),y)
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    print('coarse batch4 AMP optimizer smoke passed',loss.item(),flush=True)
    del model,opt,scaler,x,y,loss; torch.cuda.empty_cache()
    model=Xception_dilation(input_channel=4,num_classes=3).cuda()
    opt=torch.optim.Adam(model.parameters(),lr=1e-4); scaler=torch.amp.GradScaler('cuda')
    with torch.autocast('cuda',dtype=torch.float16):
        loss=torch.nn.functional.cross_entropy(model(torch.rand(8,4,224,224,device='cuda')),torch.randint(0,3,(8,),device='cuda'))
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    print('Mask-CN batch8 AMP optimizer smoke passed',loss.item(),flush=True)
    print('Peak GPU memory GiB',torch.cuda.max_memory_allocated()/1024**3,flush=True)
    del model,opt,scaler,loss; torch.cuda.empty_cache()
    from net.loss import rank_loss
    p=torch.rand(2,16,16,device='cuda'); t=(torch.rand_like(p)>.5).float()
    flat,gt=p.flatten(1),t.flatten(1)
    rank=((flat*(1-gt)).topk(30,1).values[:,:,None]-(1-((1-flat)*gt).topk(30,1).values)[:,None,:]+.3).clamp_min(0).mean()
    assert torch.allclose(rank,rank_loss(p,t),atol=1e-6)
    print('vectorized rank loss matches upstream',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true'); parser.add_argument('--seg-epochs',type=int,default=500); parser.add_argument('--cls-steps',type=int,default=50001); args=parser.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=True
    if args.smoke:smoke(); sys.exit(0)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        download('https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-cadene/xception-43020ad28.pth',ROOT/'models'/'xception-43020ad28.pth',91675053)
        prepare_raw(); prepare_gt(); train_seg(args.seg_epochs); prepare_crops(); generate_masks(); train_cls(args.cls_steps)
    except Exception as e:
        status('failed',error=repr(e)); raise
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
