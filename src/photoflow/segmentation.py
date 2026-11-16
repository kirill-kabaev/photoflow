"""Local human/face parsing, source-coordinate masks and atomic reusable cache.

No downloads at inference time. Model adapter is replaceable in offline tests;
fixtures are never reported as real model inference. Caller holds Session.lock.
"""
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal

import numpy as np
from PIL import Image, ImageDraw
from pydantic import Field
from scipy.ndimage import label, uniform_filter

from . import imaging
from .models import Checksum, SemanticName, StrictModel
from .storage import FlowError, digest, json_bytes, now, read_model, sync_dir, write_json

MODEL_ID = 'schp-atr-bisenet-r18-v1'
# ATR: dress remains separate because its upper/lower split is not predicted.
ATR_LABELS = ['background','hat','hair','sunglasses','upper_clothes','skirt','pants',
              'dress','belt','left_shoe','right_shoe','face','left_leg','right_leg',
              'left_arm','right_arm','bag','scarf']
REGION_LABELS = {'background':[0], 'person':list(range(1,18)), 'face':[11],
                 'hair':[2], 'upper_clothing':[4], 'lower_clothing':[5,6], 'dress':[7]}
FACE_LABELS = ['background','skin','left_brow','right_brow','left_eye','right_eye',
               'glasses','left_ear','right_ear','earring','nose','mouth','upper_lip',
               'lower_lip','neck','necklace','cloth','hair','hat']
REGIONS = [*REGION_LABELS, 'face_skin', 'skin', 'eyes', 'lips']
COLORS = [(255,90,30),(40,170,255),(255,180,160),(255,210,60),(80,230,120),
          (190,100,255),(250,90,160),(40,230,220)]


class SegmentConfig(StrictModel):
    schema_version: Literal[1] = 1
    model: Literal['schp-atr-bisenet-r18-v1'] = MODEL_ID
    preview_edge: Annotated[int, Field(ge=512, le=2000)] = 1600
    preview_ev: Annotated[float, Field(ge=0, le=1)] = 0.7
    refine: bool = True
    guided_radius: Annotated[int, Field(ge=1, le=8)] = 3
    guided_epsilon: Annotated[float, Field(ge=0.0001, le=0.01)] = 0.001
    min_region_pixels: Annotated[int, Field(ge=1, le=1000)] = 16


class SegmentManifest(StrictModel):
    schema_version: Literal[1] = 1
    key: Checksum
    identity: dict
    config: SegmentConfig
    dimensions: list[int]
    preview_dimensions: list[int]
    transforms: dict
    regions: dict
    timings: dict
    execution: dict
    files: dict[str, Checksum]
    warnings: list[str]
    created_at: str


@dataclass(frozen=True)
class Letterbox:
    width: int
    height: int
    size: int
    resized_width: int
    resized_height: int
    left: int
    top: int

    @classmethod
    def create(cls, width, height, size=512):
        if min(width,height,size) < 1:
            raise FlowError('Empty segmentation image')
        scale = min(size/width,size/height)
        w,h = max(1,round(width*scale)),max(1,round(height*scale))
        return cls(width,height,size,w,h,(size-w)//2,(size-h)//2)

    def apply(self, image):
        canvas = Image.new('RGB',(self.size,self.size))
        canvas.paste(image.resize((self.resized_width,self.resized_height),Image.Resampling.BILINEAR),(self.left,self.top))
        return np.asarray(canvas,dtype=np.float32)/255

    def restore(self, labels):
        if labels.ndim != 2:
            raise FlowError('Expected discrete semantic label map')
        square = Image.fromarray(labels.astype(np.uint8)).resize((self.size,self.size),Image.Resampling.NEAREST)
        cropped = square.crop((self.left,self.top,self.left+self.resized_width,self.top+self.resized_height))
        return np.asarray(cropped.resize((self.width,self.height),Image.Resampling.NEAREST)).copy()


@lru_cache(maxsize=4)
def _ort_session(path, checksum, device):
    import onnxruntime as ort
    if digest(Path(path)) != checksum:
        raise FlowError('Model weights checksum mismatch')
    available = ort.get_available_providers()
    providers = ['CPUExecutionProvider']
    if device == 'auto' and 'CUDAExecutionProvider' in available:
        providers.insert(0,'CUDAExecutionProvider')
    options = ort.SessionOptions()
    options.intra_op_num_threads = min(4,os.cpu_count() or 1)
    options.inter_op_num_threads = 1
    try:
        session = ort.InferenceSession(path,sess_options=options,providers=providers)
    except Exception as exc:
        if providers == ['CPUExecutionProvider']:
            raise FlowError(f'Local model loading failed: {exc}') from exc
        session = ort.InferenceSession(path,sess_options=options,providers=['CPUExecutionProvider'])
    return session


class LocalParser:
    """SCHP ATR for body, BiSeNet R18 on automatically located facial crops."""
    def __init__(self, directory=None, device='cpu'):
        if device not in {'cpu','auto'}:
            raise FlowError('Segmentation device must be cpu or auto')
        self.directory = Path(directory or os.environ.get('PHOTOFLOW_MODEL_DIR','.models')).resolve()
        self.device = device
        manifest = self.directory/'models.json'
        if not manifest.is_file():
            raise FlowError('Local parsing models missing; run scripts/install_segmentation_models.py explicitly')
        model_data = json.loads(manifest.read_text())
        self.identity = {'id':model_data.get('id'),'weights':model_data['weights']}
        if self.identity.get('id') != MODEL_ID:
            raise FlowError('Unsupported segmentation model identity')
        self.execution = {'adapter':'real_local_onnx','requested_device':device}
        self.timings = {}

    def _session(self,name):
        item = self.identity['weights'][name]
        if Path(item['filename']).name != item['filename']:
            raise FlowError('Unsafe model filename')
        path = self.directory/item['filename']
        if digest(path) != item['sha256']:
            raise FlowError(f'Model checksum mismatch: {name}')
        started = perf_counter()
        try:
            session = _ort_session(str(path),item['sha256'],self.device)
        except ImportError as exc:
            raise FlowError('Install the optional segmentation dependencies: pip install -e ".[segmentation]"') from exc
        timing_key = f'{name}_load_seconds'
        self.timings[timing_key] = self.timings.get(timing_key,0.)+perf_counter()-started
        import onnxruntime as ort
        self.execution.update({'available_providers':ort.get_available_providers(),'onnxruntime':ort.__version__,
                               f'{name}_providers':session.get_providers()})
        return session

    def _predict(self,name,image):
        transform = Letterbox.create(*image.size)
        rgb = transform.apply(image)
        if name == 'body':
            rgb = rgb[...,::-1].copy()  # SCHP was trained with cv2 BGR.
            mean,std = [0.406,0.456,0.485],[0.225,0.224,0.229]
        else:
            mean,std = [0.485,0.456,0.406],[0.229,0.224,0.225]
        tensor = ((rgb-np.array(mean,np.float32))/np.array(std,np.float32)).transpose(2,0,1)[None].copy()
        session = self._session(name)
        started = perf_counter()
        try:
            output = session.run(None,{session.get_inputs()[0].name:tensor})[0]
        except Exception as exc:
            raise FlowError(f'Local {name} inference failed: {exc}') from exc
        if output.ndim != 4 or output.shape[0] != 1 or not np.isfinite(output).all():
            raise FlowError('Invalid neural model output')
        classes = 18 if name == 'body' else 19
        if output.shape[1] != classes:
            raise FlowError('Neural model label schema mismatch')
        self.timings[f'{name}_inference_seconds'] = self.timings.get(f'{name}_inference_seconds',0.)+perf_counter()-started
        return transform.restore(output[0].argmax(0).astype(np.uint8)), transform.__dict__

    def predict(self,image):
        body,transform = self._predict('body',image)
        face_map = np.zeros(body.shape,np.uint8)
        components,count = label(body==11)
        boxes = []
        for index in range(1,count+1):
            yy,xx = np.where(components==index)
            if len(xx) < 64:
                continue
            boxes.append((len(xx),int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)))
        face_records = []
        for _,x1,y1,x2,y2 in sorted(boxes,reverse=True)[:4]:
            cx,cy = (x1+x2)/2,(y1+y2)/2
            side = max(x2-x1,y2-y1)*1.65
            box = (max(0,int(cx-side/2)),max(0,int(cy-side/2)),min(image.width,int(cx+side/2)),min(image.height,int(cy+side/2)))
            # Face parsing was trained predominantly upright. Estimate roll from
            # the model-predicted face component, never from user coordinates.
            yy,xx = np.where(body[y1:y2,x1:x2]==11)
            angle = face_roll(xx,yy)
            crop = image.crop(box)
            upright = crop.rotate(-angle,resample=Image.Resampling.BILINEAR)
            parsed,face_transform = self._predict('face',upright)
            parsed = np.asarray(Image.fromarray(parsed).rotate(angle,resample=Image.Resampling.NEAREST)).copy()
            x1,y1,x2,y2 = box
            target = face_map[y1:y2,x1:x2]
            valid = body[y1:y2,x1:x2]==11
            target[valid] = parsed[valid]
            reliable = int(np.isin(parsed,[4,5]).sum())>=8 and int(np.isin(parsed,[12,13]).sum())>=8
            if not reliable:
                target[valid] = 0
            face_records.append({'box_in_preview':list(box),'letterbox':face_transform,
                                 'roll_degrees':angle,'facial_features_detected':reliable})
        self.execution['face_crops'] = len(face_records)
        self.execution['unprocessed_face_components'] = max(0,len(boxes)-4)
        return body,face_map,{'body_letterbox':transform,'face_crops':face_records}


def face_roll(xx,yy):
    """PCA on predicted face support; a geometric roll estimate, not landmarks."""
    if len(xx)<3:
        return 0.
    values,vectors = np.linalg.eigh(np.cov(np.stack([xx,yy])))
    if values[0]<=0 or values[1]/values[0]<1.15:
        return 0.
    vector = vectors[:,-1]
    if vector[1]<0:
        vector = -vector
    angle = float(np.degrees(np.arctan2(vector[0],vector[1])))
    return angle if abs(angle)<=65 else 0.


def region_support(body,face,name):
    if name in REGION_LABELS:
        return np.isin(body,REGION_LABELS[name])
    if name == 'face_skin':
        return (body==11)&np.isin(face,[1,10])
    if name == 'skin':
        return ((body==11)&np.isin(face,[1,7,8,10,14])) | np.isin(body,[12,13,14,15])
    if name == 'eyes':
        return (body==11)&np.isin(face,[4,5])
    if name == 'lips':
        return (body==11)&np.isin(face,[12,13])
    raise FlowError(f'Unsupported semantic region: {name}')


def refine_mask(support,guide,config):
    """Scalar guided filter, clipped INSIDE predicted labels: no outward color spill."""
    p = support.astype(np.float32)
    if not config.refine:
        return p
    size = 2*config.guided_radius+1
    mean_i,mean_p = uniform_filter(guide,size),uniform_filter(p,size)
    covariance = uniform_filter(guide*p,size)-mean_i*mean_p
    variance = np.maximum(uniform_filter(guide*guide,size)-mean_i*mean_i,0)
    a = covariance/(variance+config.guided_epsilon)
    b = mean_p-a*mean_i
    return (np.clip(uniform_filter(a,size)*guide+uniform_filter(b,size),0,1)*p).astype(np.float32)


def _save_array(path,value):
    with path.open('xb') as handle:
        np.save(handle,value,allow_pickle=False)
        handle.flush();os.fsync(handle.fileno())


def _preview(decoded,config):
    h,w = decoded.pixels.shape[:2]
    scale = min(1,config.preview_edge/max(h,w))
    size = (max(1,round(w*scale)),max(1,round(h*scale)))
    small = np.stack([np.asarray(Image.fromarray(decoded.pixels[...,c]).resize(size,Image.Resampling.LANCZOS)) for c in range(3)],-1)
    return Image.fromarray(np.uint8(np.rint(imaging.encode_srgb(small*np.float32(2**config.preview_ev))*255)))


class SemanticCache:
    def __init__(self,session,photo,key):
        self.session,self.photo,self.key = session,photo,key
        self.folder = session.work(photo.id,'segmentation',key)
        self.manifest = read_model(self.folder/'manifest.json',SegmentManifest)
        if (self.manifest.key != key or self.manifest.identity.get('source_sha256') != photo.source_sha256
                or hashlib.sha256(json_bytes(self.manifest.identity)).hexdigest() != key
                or self.manifest.dimensions != self.manifest.identity.get('oriented_dimensions')
                or self.manifest.config.model_dump(mode='json') != self.manifest.identity.get('config')):
            raise FlowError('Segmentation cache identity mismatch')
        required = {'body-labels.npy','face-labels.npy',*[f'{name}.npy' for name in REGIONS]}
        if not required <= self.manifest.files.keys():
            raise FlowError('Segmentation cache manifest is incomplete')
        session.source(photo)
        for relative,checksum in self.manifest.files.items():
            path = session.work(photo.id,'segmentation',key,relative)
            if digest(path) != checksum:
                raise FlowError(f'Segmentation cache artifact changed: {relative}')
        self.body = np.load(self.folder/'body-labels.npy',mmap_mode='r',allow_pickle=False)
        self.face = np.load(self.folder/'face-labels.npy',mmap_mode='r',allow_pickle=False)
        expected = tuple(reversed(self.manifest.preview_dimensions))
        if self.body.shape != expected or self.face.shape != expected or self.body.dtype != np.uint8 or self.face.dtype != np.uint8:
            raise FlowError('Segmentation cache label dimensions invalid')

    def mask(self,name,shape):
        if name not in self.manifest.regions or not self.manifest.regions[name]['available']:
            raise FlowError(f'Semantic region unavailable: {name}; no ellipse fallback')
        h,w = shape[:2]
        if h<1 or w<1:
            raise FlowError('Invalid semantic target dimensions')
        values = np.load(self.folder/f'{name}.npy',mmap_mode='r',allow_pickle=False)
        if values.shape != self.body.shape or values.dtype != np.float32 or not np.isfinite(values).all() or values.min()<0 or values.max()>1:
            raise FlowError('Invalid cached semantic float mask')
        # Continuous interpolation for alpha; discrete labels never use bilinear.
        mask = np.asarray(Image.fromarray(values).resize((w,h),Image.Resampling.BILINEAR)).copy()
        support = np.asarray(Image.fromarray(region_support(self.body,self.face,name).astype(np.uint8)).resize((w,h),Image.Resampling.NEAREST))
        return np.clip(mask,0,1)*support

    def summary(self):
        return {'cache_key':self.key,'model':self.manifest.config.model,'regions':self.manifest.regions,
                'warnings':self.manifest.warnings,'diagnostics':str(self.folder/'report.md'),
                'overlays':str(self.folder/'overlays.jpg'),'execution':self.manifest.execution}


def segment(workflow,identifier,config=None,parser=None,device='cpu'):
    photo = workflow.photo(identifier)
    config = config or SegmentConfig()
    parser = parser or LocalParser(device=device)
    decoded = workflow.decoded(photo,imaging.RawSettings())
    identity = {'source_sha256':photo.source_sha256,'oriented_dimensions':decoded.metadata['oriented_dimensions'],
                'orientation_flip':decoded.metadata.get('orientation_flip',0),'decoder':decoded.metadata,
                'model':parser.identity,'config':config.model_dump(mode='json'),'pipeline_version':2}
    key = hashlib.sha256(json_bytes(identity)).hexdigest()
    folder = workflow.session.work(photo.id,'segmentation',key)
    if folder.exists():
        cache = SemanticCache(workflow.session,photo,key)
        if cache.manifest.identity != identity:
            raise FlowError('Segmentation cache key mismatch')
        return {**cache.summary(),'cache_hit':True}
    staging = workflow.session.work(photo.id,'segmentation',f'.staging-{uuid.uuid4().hex}')
    staging.mkdir(parents=True)
    started = perf_counter()
    preview = _preview(decoded,config)
    body,face,transforms = parser.predict(preview)
    expected = (preview.height,preview.width)
    if (body.shape != expected or face.shape != expected or body.dtype != np.uint8 or face.dtype != np.uint8
            or body.max()>17 or face.max()>18):
        raise FlowError('Local parser returned invalid labels/dimensions; staging preserved')
    _save_array(staging/'body-labels.npy',body);_save_array(staging/'face-labels.npy',face)
    preview.save(staging/'input.jpg',quality=94,icc_profile=imaging.srgb_profile())
    refine_started = perf_counter()
    guide = imaging.luminance(np.asarray(preview,dtype=np.float32)/255)
    regions = {}
    for name in REGIONS:
        support = region_support(body,face,name)
        available = int(support.sum())>=config.min_region_pixels
        regions[name] = {'available':available,'pixels':int(support.sum()),'coverage':float(support.mean()),
                         'derivation':'ATR labels '+str(REGION_LABELS[name]) if name in REGION_LABELS else 'BiSeNet facial labels gated by ATR face; skin also ATR bare arms/legs',
                         'quality':'unreviewed_model_prediction' if available else 'absent_or_too_small'}
        _save_array(staging/f'{name}.npy',refine_mask(support,guide,config))
    refinement_seconds = perf_counter()-refine_started
    warnings = ['Human parsing is coarse semantic segmentation, not strand-level matting.',
                'Face region includes coarse facial/neck area; face_skin excludes predicted eyes/brows/lips.',
                'ATR arm/leg labels can miss hands; no separate hand class.',
                'Dress is separate: no invented upper/lower split.',
                'At most four sufficiently large face components are parsed; inspect execution diagnostics.']
    make_diagnostics(staging,preview,body,face,regions)
    workflow.session.source(photo)
    files = {p.name:digest(p) for p in staging.iterdir() if p.is_file()}
    manifest = SegmentManifest(key=key,identity=identity,config=config,dimensions=decoded.metadata['oriented_dimensions'],
        preview_dimensions=list(preview.size),transforms=transforms,regions=regions,
        timings={**parser.timings,'refinement_seconds':refinement_seconds,'total_before_publish_seconds':perf_counter()-started},
        execution=parser.execution,files=files,warnings=warnings,created_at=now())
    write_json(staging/'manifest.json',manifest)
    os.rename(staging,folder);sync_dir(folder.parent)
    cache = SemanticCache(workflow.session,photo,key)
    return {**cache.summary(),'cache_hit':False,'timings':manifest.timings}


def make_diagnostics(folder,preview,body,face,regions):
    from .reporting import text
    names = ['person','background','face','face_skin','hair','skin','upper_clothing','lower_clothing']
    image = np.asarray(preview,dtype=np.float32)
    thumb = preview.copy();thumb.thumbnail((400,600))
    sheet = Image.new('RGB',(thumb.width*4,(thumb.height+28)*2),(24,24,24))
    lines = []
    for index,name in enumerate(names):
        alpha = np.load(folder/f'{name}.npy',allow_pickle=False)[...,None]*0.48
        overlay = Image.fromarray(np.uint8(np.rint(image*(1-alpha)+np.array(COLORS[index])*alpha)))
        overlay.save(folder/f'{name}-overlay.jpg',quality=93,icc_profile=imaging.srgb_profile())
        small = overlay.resize(thumb.size,Image.Resampling.LANCZOS)
        x,y = (index%4)*thumb.width,(index//4)*(thumb.height+28)
        sheet.paste(small,(x,y+28));ImageDraw.Draw(sheet).text((x+8,y+8),name,fill='white')
        lines.append(f'## {name}\n\nCoverage: {regions[name]["coverage"]:.5f}. Available: {regions[name]["available"]}.\n\n![{name}]({name}-overlay.jpg)')
    sheet.save(folder/'overlays.jpg',quality=94,icc_profile=imaging.srgb_profile())
    import colorsys
    palette = np.array([(0,0,0),*[tuple(round(255*c) for c in colorsys.hsv_to_rgb((i*0.61803398875)%1,.7,1)) for i in range(1,18)]],dtype=np.uint8)
    painted = Image.fromarray(palette[body])
    Image.blend(preview,painted,0.45).save(folder/'labels-overlay.jpg',quality=94,icc_profile=imaging.srgb_profile())
    painted.save(folder/'labels.png')
    Image.fromarray(face).save(folder/'face-labels.png')
    text(folder/'report.md','# Реальная локальная сегментация\n\nЦвета overlays — визуализация масок, не изменение цвета фотографии.\n'
         '\n![Все регионы](overlays.jpg)\n\n![ATR labels](labels-overlay.jpg)\n\n'+ '\n\n'.join(lines)+
         '\n\nМодель может ошибаться; численная точность без размеченного ground truth не заявляется.\n')


def compare_masks(workflow,identifier,version,key,mapping):
    """AI supplies explicit ellipse IDs to compare; overlap is not ground-truth IoU."""
    photo = workflow.photo(identifier)
    _,recipe,_ = workflow.validate_version(photo,version)
    cache = SemanticCache(workflow.session,photo,key)
    root = workflow.session.work(photo.id,'mask_comparisons',f'{version}-{key}')
    if root.exists():
        raise FlowError('Mask comparison exists; refusing overwrite')
    root.mkdir(parents=True)
    image = Image.open(cache.folder/'input.jpg').convert('RGB')
    preview = image.copy();preview.thumbnail((450,650))
    sheet = Image.new('RGB',(preview.width*2,(preview.height+30)*len(mapping)),(24,24,24))
    measurements = {}
    known = {m.id:m for m in recipe.masks}
    for row,(name,ids) in enumerate(mapping.items()):
        if not ids or any(mask_id not in known for mask_id in ids):
            raise FlowError('Comparison needs existing explicitly selected ellipse IDs')
        ellipse = np.zeros((image.height,image.width),np.float32)
        for mask_id in ids:
            ellipse = np.maximum(ellipse,imaging.ellipse_mask((*ellipse.shape,3),known[mask_id]))
        semantic = cache.mask(name,(*ellipse.shape,3))
        measurements[name] = {'ellipse_ids':ids,'ellipse_alpha_outside_predicted_semantic_fraction':
            float((ellipse*(semantic==0)).sum()/max(float(ellipse.sum()),1e-8)),
            'note':'Agreement with model predictions, NOT accuracy against manually labeled ground truth'}
        for col,(title,mask,color) in enumerate([('ELLIPSE',ellipse,(255,80,25)),('SEMANTIC',semantic,(30,230,150))]):
            alpha = mask[...,None]*0.48
            base = np.asarray(image,dtype=np.float32)
            overlay = Image.fromarray(np.uint8(np.rint(base*(1-alpha)+np.array(color)*alpha)))
            overlay.save(root/f'{name}-{title.lower()}.jpg',quality=94,icc_profile=imaging.srgb_profile())
            sheet.paste(overlay.resize(preview.size,Image.Resampling.LANCZOS),(col*preview.width,row*(preview.height+30)+30))
            ImageDraw.Draw(sheet).text((col*preview.width+8,row*(preview.height+30)+8),f'{name}: {title}',fill='white')
    sheet.save(root/'comparison.jpg',quality=94,icc_profile=imaging.srgb_profile())
    from .reporting import text
    text(root/'report.md',f'# Эллипсы {version} и реальные semantic masks\n\nЦвет означает маску, не цветокоррекцию.\n\n![Сравнение](comparison.jpg)\n\n```json\n{json.dumps(measurements,indent=2)}\n```\n\nЭто сравнение локализации с предсказаниями, не ground-truth accuracy.\n')
    write_json(root/'manifest.json',{'version':version,'segmentation_key':key,'mapping':mapping,'measurements':measurements,
        'files':{p.name:digest(p) for p in root.iterdir() if p.is_file()},'created_at':now()})
    return {'report':str(root/'report.md'),'comparison':str(root/'comparison.jpg'),'measurements':measurements}


def boundary_diagnostics(workflow,identifier,version,key):
    """Show model alphas on native RAW crops; alpha resolution remains model-limited."""
    photo = workflow.photo(identifier)
    _,recipe,_ = workflow.validate_version(photo,version)
    cache = SemanticCache(workflow.session,photo,key)
    root = workflow.session.work(photo.id,'semantic_details',f'{version}-{key}')
    if root.exists():
        raise FlowError('Semantic detail report exists; refusing overwrite')
    root.mkdir(parents=True)
    source = workflow.decoded(photo,imaging.RawSettings()).pixels
    h,w = source.shape[:2]
    relevant = {'face_skin':['face'],'hair':['hair'],'upper_clothing':['collar','waist'],
                'skin':['hand_left','hand_right'],'lower_clothing':['waist']}
    by_id = {region.id:region for region in recipe.detail_regions}
    records = {}
    for name,regions in relevant.items():
        alpha = cache.mask(name,source.shape)
        for region_id in regions:
            if region_id not in by_id:
                continue
            bounds = by_id[region_id].bounds
            x1,y1,x2,y2 = int(w*bounds.left),int(h*bounds.top),int(w*bounds.right),int(h*bounds.bottom)
            if x2<=x1 or y2<=y1:
                raise FlowError('Empty native semantic detail')
            rgb = imaging.encode_srgb(source[y1:y2,x1:x2])
            weight = alpha[y1:y2,x1:x2,None]*0.45
            overlay = Image.fromarray(np.uint8(np.rint(np.clip(rgb*(1-weight)+np.array([.1,.95,.45])*weight,0,1)*255)))
            filename = f'{name}-{region_id}.jpg'
            overlay.save(root/filename,quality=95,icc_profile=imaging.srgb_profile())
            records[filename] = {'source_box':[x1,y1,x2,y2],'dimensions':list(overlay.size),'display_scale':1.0,
                                 'inference_size':[512,512],'note':'Native raster mapping, not native neural detail'}
        del alpha
    import colorsys
    palette = np.array([(0,0,0),*[tuple(round(255*c) for c in colorsys.hsv_to_rgb((i*.61803398875)%1,.7,1)) for i in range(1,18)]],np.uint8)
    preview = Image.open(cache.folder/'input.jpg').convert('RGB')
    combined = Image.blend(preview,Image.fromarray(palette[cache.body]),.45)
    canvas = Image.new('RGB',(combined.width+260,combined.height),(25,25,25))
    canvas.paste(combined,(0,0));draw = ImageDraw.Draw(canvas)
    for index,name in enumerate(ATR_LABELS):
        draw.rectangle((combined.width+10,10+index*34,combined.width+30,30+index*34),fill=tuple(int(v) for v in palette[index]))
        draw.text((combined.width+40,15+index*34),f'{index}: {name}',fill='white')
    canvas.save(root/'labels-with-legend.jpg',quality=94,icc_profile=imaging.srgb_profile())
    from .reporting import text
    text(root/'report.md','# Семантические границы на RAW-кропах\n\nЦвет — визуализация alpha, не цветокоррекция. Кропы без увеличения; исходный neural input 512x512.\n\n![ATR labels](labels-with-legend.jpg)\n\n'+
         '\n\n'.join(f'## {name}\n\n![{name}]({name})' for name in records))
    write_json(root/'manifest.json',{'cache_key':key,'version':version,'details':records,
        'files':{p.name:digest(p) for p in root.iterdir() if p.is_file()},'created_at':now()})
    return {'report':str(root/'report.md'),'labels':str(root/'labels-with-legend.jpg'),'details':records}
