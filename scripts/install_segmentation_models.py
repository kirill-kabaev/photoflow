"""Explicit trusted downloads + CPU ONNX preparation. Never reads photographs.

Runtime uses only ONNX Runtime; PyTorch/ONNX are installation/export tools.
SCHP architecture remains upstream MIT code, fetched at a pinned commit/hash.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import urllib.request

SCHP_COMMIT = 'eb84c432cc697f494d99662a05f2335eb2f26095'
SOURCES = {
    'schp-atr.pth': ('https://drive.usercontent.google.com/download?id=1ruJg4lqR_jgQPj-9K0PP-L2vJERYOxLP&export=download&confirm=t',
                     'e9d7c91ce3b4e7133df56b599fc817b533e3439c5e8d282a59126d2fda339a2a'),
    'schp-architecture.py': (f'https://raw.githubusercontent.com/GoGoDuck912/Self-Correction-Human-Parsing/{SCHP_COMMIT}/networks/AugmentCE2P.py',
                             '8ab50f26dc9bc085619b1d5489399ca4c025e53d3d2c874b49c7bdd225bef321'),
    'SCHP-LICENSE.txt': (f'https://raw.githubusercontent.com/GoGoDuck912/Self-Correction-Human-Parsing/{SCHP_COMMIT}/LICENSE',
                         '4b6f33d1127bad303130ad839fd79541e4390c43a4c4de3e9ebbdd90978df941'),
    'face-resnet18.onnx': ('https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx',
                          '0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f'),
    'FACE-LICENSE.txt': ('https://raw.githubusercontent.com/yakhyo/face-parsing/main/LICENSE',
                         'a4f7998feb5c90555daae4329db61caf7049e6b2706991dc1d9eb2a0923847ae'),
}


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def fetch(directory,download):
    directory.mkdir(parents=True,exist_ok=True)
    for name,(url,expected) in SOURCES.items():
        path = directory/name
        if not path.exists():
            if not download:
                raise RuntimeError(f'Missing {path}; use --download for explicit official downloads')
            staging = path.with_suffix(path.suffix+'.part')
            with urllib.request.urlopen(url,timeout=120) as response,staging.open('xb') as handle:
                while data := response.read(1024*1024):
                    handle.write(data)
                handle.flush();os.fsync(handle.fileno())
            if sha(staging) != expected:
                raise RuntimeError(f'Download SHA-256 mismatch: {name}; partial file retained')
            staging.replace(path)
        if sha(path) != expected:
            raise RuntimeError(f'Model/source/license checksum mismatch: {name}')


def export(directory):
    import torch
    from torch import nn
    import torch.nn.functional as F
    import numpy as np
    import onnxruntime as ort
    torch.set_num_threads(4)
    torch.manual_seed(0)

    class InferenceABN(nn.BatchNorm2d):
        """Inference formula verified against upstream modules/src/inplace_abn_cpu.cpp."""
        def __init__(self,num_features,eps=1e-5,momentum=0.1,affine=True,activation='leaky_relu',slope=0.01):
            super().__init__(num_features,eps,momentum,affine)
            self.activation,self.slope = activation,slope

        def forward(self,x):
            x = F.batch_norm(x,self.running_mean,self.running_var,self.weight.abs()+self.eps,self.bias,False,0.,self.eps)
            return F.leaky_relu(x,self.slope) if self.activation=='leaky_relu' else x

    shim = types.ModuleType('modules')
    shim.InPlaceABNSync = InferenceABN
    previous = sys.modules.get('modules')
    sys.modules['modules'] = shim
    try:
        spec = importlib.util.spec_from_file_location('schp_arch',directory/'schp-architecture.py')
        arch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(arch)  # Trusted, pinned upstream installation source; never recipe code.
    finally:
        if previous is None:
            del sys.modules['modules']
        else:
            sys.modules['modules'] = previous
    model = arch.resnet101(num_classes=18,pretrained=None).eval()
    checkpoint = torch.load(directory/'schp-atr.pth',map_location='cpu',weights_only=True)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in checkpoint['state_dict'].items()},strict=True)

    class StaticPool(nn.Module):
        """Exact adaptive mean bins, including non-divisible dimensions (32 -> 3)."""
        def __init__(self,size):
            super().__init__();self.size=(size,size) if isinstance(size,int) else size

        def forward(self,x):
            h,w=x.shape[-2:];oh,ow=self.size
            return torch.cat([torch.cat([x[:,:,i*h//oh:((i+1)*h+oh-1)//oh,j*w//ow:((j+1)*w+ow-1)//ow].mean((-2,-1),keepdim=True)
                for j in range(ow)],dim=3) for i in range(oh)],dim=2)

    def replace(module):
        for name,child in module.named_children():
            if isinstance(child,nn.AdaptiveAvgPool2d):
                setattr(module,name,StaticPool(child.output_size))
            else:
                replace(child)

    class Output(nn.Module):
        def __init__(self,model):
            super().__init__();self.model=model

        def forward(self,x):
            return F.interpolate(self.model(x)[0][-1],size=(512,512),mode='bilinear',align_corners=True)

    x = torch.randn(1,3,512,512)
    target = directory/'schp-atr.onnx'
    staging = directory/'schp-atr.partial.onnx'
    with torch.inference_mode():
        reference = Output(model)(x)
        replace(model)
        output = Output(model).eval()
        actual = output(x)
        torch.onnx.export(output,x,staging,opset_version=17,input_names=['input'],output_names=['logits'],dynamo=False)
    options = ort.SessionOptions();options.intra_op_num_threads=4
    session = ort.InferenceSession(str(staging),sess_options=options,providers=['CPUExecutionProvider'])
    predicted = session.run(None,{'input':x.numpy()})[0]
    error = float(np.abs(predicted-reference.numpy()).max())
    agreement = float((predicted.argmax(1)==reference.numpy().argmax(1)).mean())
    pool_error = float((reference-actual).abs().max())
    if error>=0.001 or agreement<0.999 or pool_error>=0.001:
        raise RuntimeError('SCHP ONNX parity verification failed; refusing installation')
    staging.replace(target)
    verification = {'pool_max_difference':pool_error,'onnx_max_difference':error,'label_agreement':agreement,
                    'torch':torch.__version__,'onnxruntime':ort.__version__}
    (directory/'export-verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('.models'))
    parser.add_argument('--download',action='store_true')
    args = parser.parse_args()
    fetch(args.directory,args.download)
    verification = export(args.directory)
    data = {'id':'schp-atr-bisenet-r18-v1','weights':{
        'body':{'filename':'schp-atr.onnx','sha256':sha(args.directory/'schp-atr.onnx'),
                'source_checkpoint_sha256':SOURCES['schp-atr.pth'][1],
                'source':'https://github.com/GoGoDuck912/Self-Correction-Human-Parsing','revision':SCHP_COMMIT,
                'license':'MIT repository; dataset terms separate'},
        'face':{'filename':'face-resnet18.onnx','sha256':SOURCES['face-resnet18.onnx'][1],
                'source':'https://github.com/yakhyo/face-parsing/releases/tag/weights',
                'license':'MIT repository; CelebAMask-HQ terms separate'}},
        'export_verification':verification,
        'input':'letterbox512; RGB sRGB; body BGR ImageNet normalization; face RGB ImageNet normalization'}
    temp = args.directory/'models.partial.json'
    temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(args.directory/'models.json')
    print(json.dumps(data,indent=2))


if __name__=='__main__':
    main()
