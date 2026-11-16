"""Local suspected-spot proposals and explicit per-photo consent. Never a diagnosis."""
import hashlib
import json
from typing import Annotated, Literal

import numpy as np
from PIL import Image, ImageDraw
from pydantic import Field
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, label, maximum_filter

from . import imaging
from .models import Checksum, Identifier, SpotHealing, StrictModel, Text, Version
from .storage import FlowError, digest, json_bytes, now, read_model, write_json

ALGORITHM = 'red-local-contrast-v3'
QUESTION = 'Potential pimples detected. Remove them? Yes / No'


class Spot(StrictModel):
    id: Identifier
    x: Annotated[int, Field(ge=0)]
    y: Annotated[int, Field(ge=0)]
    radius: Annotated[int, Field(ge=2, le=8)]
    status: Literal['suspected'] = 'suspected'
    red_contrast: Annotated[float, Field(gt=0, le=1)]
    confidence: Annotated[float, Field(ge=0, le=1)] = 0.0


class Proposal(StrictModel):
    id: Annotated[str, Field(pattern=r'^[0-9a-f]{32}$')]
    photo_id: Identifier
    source_sha256: Checksum
    version: Version
    recipe_sha256: Checksum
    cache_key: Checksum
    dimensions: Annotated[list[int], Field(min_length=2, max_length=2)]
    source_box: Annotated[list[int], Field(min_length=4, max_length=4)]
    detail_id: Identifier
    input_sha256: Checksum
    overlay_sha256: Checksum
    spots: Annotated[list[Spot], Field(max_length=8)]
    algorithm: Literal['red-local-contrast-v1','red-local-contrast-v2','red-local-contrast-v3'] = ALGORITHM
    created_at: Text
    reviewed_from: Annotated[str, Field(pattern=r'^[0-9a-f]{32}$')] | None = None
    review_reason: str = ''
    note: Text = ('Yellow circles = suspected red spots, NOT confirmed defects or a medical diagnosis. '
                  'Permanent features cannot be reliably classified. No pixels have been healed.')


class Decision(StrictModel):
    request_id: Annotated[str, Field(pattern=r'^[0-9a-f]{32}$')]
    photo_id: Identifier
    proposal_sha256: Checksum
    shown_overlay_sha256: Checksum
    answer: Literal['Yes', 'No']
    by: Text
    user_message: Text
    created_at: Text


def detect(rgb, skin):
    """Conservative redness maxima at native resolution, excluding dark marks/edges.

    rgb is display-encoded sRGB, used ONLY for detection, never as a RAW render source.
    """
    if (rgb.ndim != 3 or rgb.shape[2] != 3 or skin.shape != rgb.shape[:2]
            or not np.isfinite(rgb).all() or not np.isfinite(skin).all()
            or rgb.min()<0 or rgb.max()>1 or skin.min()<0 or skin.max()>1):
        raise FlowError('Invalid face/skin detection input')
    if min(skin.shape)<21:
        return []
    lum = rgb @ np.array([.2126, .7152, .0722], np.float32)
    red = rgb[..., 0] - (rgb[..., 1] + rgb[..., 2]) * .5
    score = gaussian_filter(red, 1.5) - gaussian_filter(red, 8)
    persistent = gaussian_filter(red, 3)-gaussian_filter(red, 10)
    dark = lum - gaussian_filter(lum, 5) < -.065
    edge_y, edge_x = np.gradient(lum)
    protected = binary_dilation(dark | (np.hypot(edge_x, edge_y) > .055), iterations=3)
    safe = binary_erosion(skin >= .95, iterations=15) & ~protected
    peaks = (safe & (score > .04) & (persistent > .025)
             & (score == maximum_filter(score, size=17)) & (lum > .15) & (lum < .92))
    ys, xs = np.where(peaks)
    selected = []
    for i in np.argsort(score[ys, xs])[::-1]:
        x, y = int(xs[i]), int(ys[i])
        patch = score[y-14:y+15, x-14:x+15]
        components, _ = label(patch > score[y,x]*.5)
        component = np.argwhere(components == components[14,14])
        extent = component.max(0)-component.min(0)+1
        if min(extent) < 5 or max(extent) > 15 or max(extent)/min(extent) > 1.5 or not 15 <= len(component) <= 150:
            continue  # Reject extended folds/colour boundaries, not compact spots.
        yy,xx = np.mgrid[-14:15,-14:15]
        distance = np.hypot(xx,yy)
        ring = (distance>=9)&(distance<=14)
        center = distance<=2
        local_red = red[y-14:y+15,x-14:x+15]
        baseline = np.median(local_red[ring])
        contrast = float(np.median(local_red[center])-baseline)
        noise = max(.005,float(1.4826*np.median(np.abs(local_red[ring]-baseline))))
        local_lum = lum[y-14:y+15,x-14:x+15]
        if contrast < .045 or contrast/noise < 4 or np.median(local_lum[center])-np.median(local_lum[ring]) < -.02:
            continue  # Reject low contrast/noise, freckles and dark pigmentation.
        compactness = len(component)/float(np.prod(extent))
        confidence = .45*min(1,contrast/.065)+.3*min(1,contrast/noise/8)+.25*compactness
        if confidence < .8:
            continue  # Heuristic evidence score, NOT a calibrated acne probability.
        if any((x-p['x'])**2+(y-p['y'])**2 < 20**2 for p in selected):
            continue
        selected.append({'id': f'spot{len(selected)+1:02d}', 'x': x, 'y': y, 'radius': 5,
                         'status': 'suspected', 'red_contrast': float(score[y, x]), 'confidence':float(confidence)})
        if len(selected) == 8:
            break
    return selected


def eligible_skin(cache, shape):
    """Strict guard zones around eyes/eyelids/brows/lips/nose/hair.

    Conservative omission of the nose avoids mistaking nostrils/creases for acne.
    Label maps are restored to the FULL oriented frame before any crop.
    """
    h,w = shape[:2]
    labels = np.asarray(Image.fromarray(cache.face).resize((w,h),Image.Resampling.NEAREST))
    yy,xx = np.where(np.isin(cache.face,[1,2,3,4,5,10,11,12,13]))
    if not len(xx):
        return np.zeros((h,w),np.float32)
    face_extent = max((xx.max()-xx.min()+1)*w/cache.face.shape[1],
                      (yy.max()-yy.min()+1)*h/cache.face.shape[0])
    margin = max(12,round(face_extent*.05))
    forbidden = maximum_filter((labels != 1).astype(np.uint8),size=2*margin+1,mode='constant',cval=1)>0
    return cache.mask('face_skin',shape)*(~forbidden)


def directory(workflow, photo, request_id):
    # Validate before composing a filesystem path, including public CLI calls.
    if len(request_id) != 32 or any(c not in '0123456789abcdef' for c in request_id):
        raise FlowError('Invalid healing request ID')
    return workflow.session.work(photo.id, 'spot_healing', request_id)


def load_proposal(workflow, photo, request_id):
    folder = directory(workflow, photo, request_id)
    proposal = read_model(folder / 'proposal.json', Proposal)
    if (proposal.id != request_id or proposal.photo_id != photo.id or proposal.source_sha256 != photo.source_sha256):
        raise FlowError('Healing proposal identity mismatch')
    version, recipe, manifest = workflow.validate_version(photo, proposal.version)
    if proposal.recipe_sha256 != manifest['recipe_sha256'] or not recipe.segmentation or recipe.segmentation.cache_key != proposal.cache_key:
        raise FlowError('Healing proposal recipe/segmentation changed')
    detail = version / 'details' / proposal.detail_id
    region = json.loads((detail / 'region.json').read_text())
    dims = json.loads((version / 'decoder.json').read_text())['oriented_dimensions']
    if (region['source_box'] != proposal.source_box or region['scale'] != 1 or dims != proposal.dimensions
            or digest(detail / 'after.jpg') != proposal.input_sha256
            or digest(folder / 'suspected.jpg') != proposal.overlay_sha256):
        raise FlowError('Healing proposal visual input or geometry changed')
    x1, y1, x2, y2 = proposal.source_box
    if not (0 <= x1 < x2 <= dims[0] and 0 <= y1 < y2 <= dims[1]) or any(
            not (x1 <= s.x-s.radius and s.x+s.radius < x2 and y1 <= s.y-s.radius and s.y+s.radius < y2)
            for s in proposal.spots):
        raise FlowError('Healing spot outside native face crop')
    workflow.session.source(photo)
    if proposal.reviewed_from:
        if proposal.reviewed_from == proposal.id:
            raise FlowError('Recursive healing review reference')
        parent = read_model(directory(workflow,photo,proposal.reviewed_from)/'proposal.json',Proposal)
        if parent.reviewed_from or (parent.photo_id,parent.version,parent.input_sha256) != (photo.id,proposal.version,proposal.input_sha256):
            raise FlowError('Invalid healing candidate review parent')
        if not proposal.review_reason.strip() or any(s not in parent.spots for s in proposal.spots):
            raise FlowError('AI review can only reject candidates, never invent new targets')
    return folder, proposal


def scan(workflow, identifier, version=None):
    """Detection from an existing native preview only. No RAW decode/model inference."""
    photo = workflow.photo(identifier)
    version = version or photo.current_version
    if not version:
        return {'status': 'unavailable', 'reason': 'No existing version/native face crop'}
    folder, recipe, _ = workflow.validate_version(photo, version)
    if not recipe.segmentation:
        return {'status': 'unavailable', 'reason': 'No existing semantic face_skin cache; no ellipse fallback'}
    from .segmentation import SemanticCache
    cache = SemanticCache(workflow.session, photo, recipe.segmentation.cache_key)
    if not cache.manifest.regions.get('face_skin', {}).get('available'):
        return {'status': 'unavailable', 'reason': 'Semantic face_skin unavailable'}
    # Explicit face detail ID convention; do not guess clothing/eyes to be facial skin.
    faces = [r for r in recipe.detail_regions if r.id in {'face', 'face_skin', 'skin_face'}]
    if not faces:
        return {'status': 'unavailable', 'reason': 'No native face detail region; detection not performed'}
    detail = folder / 'details' / faces[0].id
    region = json.loads((detail / 'region.json').read_text())
    dims = cache.manifest.dimensions
    x1, y1, x2, y2 = region['source_box']
    with Image.open(detail / 'after.jpg') as source:
        face = source.convert('RGB')
    if region['scale'] != 1 or face.size != (x2-x1, y2-y1):
        raise FlowError('Detection requires an actual native 100% face crop')
    full_skin = eligible_skin(cache, (dims[1], dims[0], 3))
    skin = full_skin[y1:y2, x1:x2].copy()
    del full_skin
    spots = detect(np.asarray(face, dtype=np.float32)/255, skin)
    identity = {'photo': photo.id, 'version': version, 'input': digest(detail/'after.jpg'),
                'cache': cache.key, 'algorithm': ALGORITHM, 'overlay_version':2}
    request_id = hashlib.sha256(json_bytes(identity)).hexdigest()[:32]
    target = directory(workflow, photo, request_id)
    if not target.exists():
        target.mkdir(parents=True)
        canvas = Image.new('RGB', (face.width, face.height+60), (25, 25, 25))
        allowed = skin>=.95
        boundary = allowed & ~binary_erosion(allowed)
        annotated = np.asarray(face).copy()
        annotated[boundary] = [60,210,240]
        canvas.paste(Image.fromarray(annotated), (0, 60))
        draw = ImageDraw.Draw(canvas)
        draw.text((10,7),f'VALID CANDIDATES: {len(spots)}. Confirmed defects: NONE. No healing.',fill='white')
        draw.text((10,25),'CYAN: safe-zone boundary; YELLOW: machine candidates pending visual review. 1:1.',fill=(60,210,240))
        for spot in spots:
            x, y, r = spot['x'], spot['y']+60, spot['radius']+6
            draw.ellipse((x-r, y-r, x+r, y+r), outline='yellow', width=2)
            draw.text((x+r+2, y-r), spot['id'], fill='yellow')
            spot['x'] += x1
            spot['y'] += y1
        canvas.save(target/'suspected.jpg', quality=96, subsampling=0, icc_profile=imaging.srgb_profile())
        proposal = Proposal(id=request_id, photo_id=photo.id, source_sha256=photo.source_sha256,
            version=version, recipe_sha256=digest(folder/'recipe.json'), cache_key=cache.key,
            dimensions=dims, source_box=region['source_box'], detail_id=faces[0].id,
            input_sha256=identity['input'], overlay_sha256=digest(target/'suspected.jpg'),
            spots=spots, created_at=now())
        write_json(target/'proposal.json', proposal)
    target, proposal = load_proposal(workflow, photo, request_id)
    return {'status': 'needs_visual_review' if proposal.spots else 'none_detected', 'photo_id': photo.id,
            'request_id': request_id, 'count': len(proposal.spots), 'confirmed_defects': 0,
            'overlay': str(target/'suspected.jpg'), 'overlay_sha256': proposal.overlay_sha256,
            'question': None, 'note': proposal.note,
            'instruction': 'Inspect real native crop and overlay first. Reject every uncertain mark, including all candidates. Do not propose healing until AI visual review. Notify user of possible spots; no automatic healing.'}


def review_candidates(workflow, identifier, request_id, keep, reason, inspected_checksum):
    """AI may discard uncertain candidates. This action never grants consent."""
    photo = workflow.photo(identifier)
    original, proposal = load_proposal(workflow,photo,request_id)
    if proposal.reviewed_from or (original/'decision.json').exists() or inspected_checksum != proposal.overlay_sha256:
        raise FlowError('Review requires an undecided original proposal and actual inspected overlay')
    if not reason.strip() or not set(keep) <= {s.id for s in proposal.spots}:
        raise FlowError('Review needs a reason and existing candidate IDs only')
    identity = {'parent':request_id,'keep':sorted(set(keep)),'reason':reason}
    new_id = hashlib.sha256(json_bytes(identity)).hexdigest()[:32]
    folder = directory(workflow,photo,new_id)
    link = original/'candidate_review.json'
    if link.exists() and json.loads(link.read_text()) != identity|{'request_id':new_id}:
        raise FlowError('Candidate review is immutable; cannot replace a shown selection')
    if not folder.exists():
        folder.mkdir()
        detail = workflow.session.work(photo.id,proposal.version,'details',proposal.detail_id,'after.jpg')
        with Image.open(detail) as source:
            face = source.convert('RGB')
        canvas = Image.new('RGB',(face.width,face.height+60),(25,25,25)); canvas.paste(face,(0,60))
        draw = ImageDraw.Draw(canvas)
        draw.text((10,7),'YELLOW: SUSPECTED only; confirmed defects: NONE',fill='yellow')
        draw.text((10,25),'AI rejected uncertain marks. No healing. Native crop 1:1.',fill='white')
        kept = [s for s in proposal.spots if s.id in keep]
        x1,y1,_,_ = proposal.source_box
        for s in kept:
            x,y,r = s.x-x1,s.y-y1+60,s.radius+6
            draw.ellipse((x-r,y-r,x+r,y+r),outline='yellow',width=2)
            draw.text((x+r+2,y-r),s.id,fill='yellow')
        canvas.save(folder/'suspected.jpg',quality=96,subsampling=0,icc_profile=imaging.srgb_profile())
        revised = proposal.model_copy(update={'id':new_id,'spots':kept,'reviewed_from':proposal.id,
            'review_reason':reason,'overlay_sha256':digest(folder/'suspected.jpg'),'created_at':now()})
        write_json(folder/'proposal.json',revised)
    write_json(link,identity|{'request_id':new_id})
    return current_check(workflow,photo,{'request_id':request_id,'status':'suspected'})


def current_check(workflow, photo, check):
    if not check.get('request_id'):
        return check
    folder = directory(workflow,photo,check['request_id'])
    _,original=load_proposal(workflow,photo,check['request_id'])
    if original.algorithm!=ALGORITHM:
        return scan(workflow,photo.id,photo.current_version)
    link = folder/'candidate_review.json'
    if not link.exists():
        return check
    child_id = json.loads(link.read_text())['request_id']
    child, proposal = load_proposal(workflow,photo,child_id)
    if proposal.reviewed_from != check['request_id']:
        raise FlowError('Candidate review identity mismatch')
    return {'status':'suspected' if proposal.spots else 'none_detected','photo_id':photo.id,
        'request_id':child_id,'count':len(proposal.spots),'confirmed_defects':0,'overlay':str(child/'suspected.jpg'),
        'overlay_sha256':proposal.overlay_sha256,'question':QUESTION if proposal.spots else None,
        'note':proposal.note,'review_reason':proposal.review_reason}


def decide(workflow, identifier, request_id, answer, actor, message, shown_checksum):
    photo = workflow.photo(identifier)
    folder, proposal = load_proposal(workflow, photo, request_id)
    if not proposal.reviewed_from or proposal.algorithm != ALGORITHM:
        raise FlowError('Actual AI visual review of current detections is required before consent')
    if (folder/'candidate_review.json').exists():
        raise FlowError('Proposal superseded by AI candidate review; show the reviewed overlay')
    if not proposal.spots or photo.current_version != proposal.version:
        raise FlowError('No current suspected spots available for this decision')
    if shown_checksum != proposal.overlay_sha256:
        raise FlowError('Consent requires acknowledgement of the actual shown overlay')
    decision = Decision(request_id=request_id, photo_id=photo.id, proposal_sha256=digest(folder/'proposal.json'),
        shown_overlay_sha256=shown_checksum, answer=answer, by=actor, user_message=message, created_at=now())
    path = folder/'decision.json'
    if path.exists():
        old = read_model(path, Decision)
        if (old.answer, old.by, old.user_message) != (answer, actor, message):
            raise FlowError('Healing decision is immutable; cannot overwrite consent')
    else:
        write_json(path, decision)
    return {'photo_id': photo.id, 'answer': answer, 'request_id': request_id,
            'decision_sha256': digest(path), 'healed': False, 'approved_for_export': False}


def authorization(workflow, photo, recipe):
    """Validate before render reservation and again on export. No recipe self-consent."""
    if recipe.spot_healing is None:
        return None
    folder, proposal = load_proposal(workflow, photo, recipe.spot_healing.request_id)
    if (proposal.algorithm != ALGORITHM or not proposal.reviewed_from
            or (folder/'candidate_review.json').exists() or recipe.parent_version is None):
        raise FlowError('Healing consent is stale or missing its parent version')
    path = folder/'decision.json'
    decision = read_model(path, Decision)
    if (digest(path) != recipe.spot_healing.decision_sha256 or decision.answer != 'Yes'
            or decision.photo_id != photo.id or decision.request_id != proposal.id
            or decision.proposal_sha256 != digest(folder/'proposal.json')
            or decision.shown_overlay_sha256 != proposal.overlay_sha256 or not proposal.spots
            or any(s.confidence<.8 for s in proposal.spots)):
        raise FlowError('Spot Healing requires explicit Yes bound to this photo and preview')
    # A consent grants these coordinates and photographic development only. Revisions
    # may retain the same binding but cannot apply it to a different grade/crop.
    _, parent, _ = workflow.validate_version(photo, proposal.version)
    for field in ['raw', 'global_adjustments', 'masks', 'semantic_masks', 'segmentation', 'local_adjustments', 'natural_skin']:
        if getattr(parent, field) != getattr(recipe, field):
            raise FlowError('Healing consent is stale for changed development settings')
    if recipe.parent_version != proposal.version:
        _, previous, _ = workflow.validate_version(photo, recipe.parent_version)
        if previous.spot_healing != recipe.spot_healing:
            raise FlowError('Healing consent cannot be replayed on another branch')
    return proposal, decision


def _protected(rgb):
    lum = rgb @ np.array([.2126, .7152, .0722], np.float32)
    gy, gx = np.gradient(lum)
    return binary_dilation((lum-gaussian_filter(lum, 4) < -.065) | (np.hypot(gx, gy) > .055), iterations=2)


def heal_pixels(image, proposal, skin, folder=None):
    """Tiny feathered texture clones on developed linear pixels. No broad smoothing.

    Called only by the validated provider. All donors are planned before mutation.
    """
    if [image.shape[1], image.shape[0]] != proposal.dimensions or skin.shape != image.shape[:2]:
        raise FlowError('Healing source/mask dimensions mismatch')
    x1, y1, x2, y2 = proposal.source_box
    face = image[y1:y2, x1:x2]
    rgb = imaging.encode_srgb(face)
    protected = _protected(rgb)
    skin = skin[y1:y2, x1:x2]
    native_before = face.copy()
    planned = []
    for spot in proposal.spots:
        x, y, r = spot.x-x1, spot.y-y1, spot.radius
        def patch(a, b):
            return np.s_[b-r:b+r+1, a-r:a+r+1]
        def safe(a, b):
            return (r <= a < face.shape[1]-r and r <= b < face.shape[0]-r
                    and np.all(skin[patch(a,b)] >= .95) and not protected[patch(a,b)].any())
        if not safe(x, y):
            raise FlowError(f'{spot.id}: unsafe target near dark mark/feature or mask edge; no healing applied')
        choices = []
        for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
            a, b = x+round(4*r*np.cos(angle)), y+round(4*r*np.sin(angle))
            if not safe(a,b) or any((a-(s.x-x1))**2+(b-(s.y-y1))**2 < (r+s.radius+2)**2 for s in proposal.spots):
                continue
            donor = face[patch(a,b)].copy()
            target = face[patch(x,y)]
            difference = np.abs(np.median(donor, axis=(0,1))-np.median(target, axis=(0,1))).mean()
            if difference < .06:
                choices.append((float(difference), a, b, donor))
        if not choices:
            raise FlowError(f'{spot.id}: no safe nearby skin texture donor; no fallback smoothing')
        _, a, b, donor = min(choices, key=lambda v:v[0])
        planned.append((spot, x, y, a, b, donor))
    records = []
    for spot, x, y, a, b, donor in planned:
        r = spot.radius
        yy, xx = np.mgrid[-r:r+1, -r:r+1]
        distance = np.sqrt(xx*xx+yy*yy)/r
        weight = np.clip((1-distance)/.45, 0, 1)[..., None]*.8
        patch = face[y-r:y+r+1, x-r:x+r+1]
        # Match surrounding illumination using annulus, preserving donor texture.
        ring = distance >= .8
        offset = np.median(patch[ring],axis=0)-np.median(donor[ring],axis=0)
        patch += weight*(np.maximum(donor+np.clip(offset,-.025,.025),0)-patch)
        records.append({'spot_id':spot.id, 'target':[spot.x,spot.y], 'radius':r,
                        'donor':[a+x1,b+y1], 'maximum_blend':.8})
    if folder is not None:
        folder.mkdir(parents=True)
        imaging.save_previews(native_before, face, folder, max_edge=max(face.shape[:2]))
        write_json(folder/'proposal.json',proposal)
        write_json(folder/'operation.json',{'algorithm':'local-texture-clone-v1','source_box':proposal.source_box,
            'scale':1.0,'source':'developed full-resolution linear RGB before JPEG encoding',
            'before':'same grade immediately before healing','spots':records})
    return image


class HealingProvider:
    """Workflow-created authorization, not a boolean supplied by a recipe."""
    def __init__(self, verified, semantic_cache):
        self.proposal, self.decision = verified
        self.cache = semantic_cache
        if self.cache is None or self.cache.key != self.proposal.cache_key:
            raise FlowError('Healing requires the original semantic cache')

    def apply(self, pixels, folder=None):
        mask = eligible_skin(self.cache,pixels.shape)
        result = heal_pixels(pixels,self.proposal,mask,folder)
        if folder:
            write_json(folder/'decision.json',self.decision)
        return result


def apply_approved(workflow, identifier, request_id):
    """Create one budgeted RAW revision with unchanged grade, only after explicit Yes."""
    photo = workflow.photo(identifier)
    folder, proposal = load_proposal(workflow, photo, request_id)
    if photo.current_version != proposal.version or photo.state != 'review':
        raise FlowError('Healing requires the unchanged current version on review')
    if photo.active_attempt_count >= photo.active_pass_limit:
        raise FlowError('No remaining full-resolution pass for healing; no budget reset allowed')
    from .models import Plan
    original, recipe, _ = workflow.validate_version(photo, proposal.version)
    binding = SpotHealing(request_id=request_id,decision_sha256=digest(folder/'decision.json'))
    recipe = recipe.model_copy(deep=True)
    recipe.parent_version, recipe.spot_healing = proposal.version, binding
    authorization(workflow,photo,recipe)  # Validate before changing revision state.
    request = folder/'feedback.md'
    from .reporting import text
    text(request, f'Explicit per-photo Yes: {request_id}. Apply only the displayed suspected spots; preserve grade and permanent features.')
    workflow.request_revision(photo.id,feedback_file=request)
    recipe.feedback_applied = 'Explicit user Yes for displayed suspected spots only; no photographic approval/export.'
    recipe.feedback_sha256 = digest(workflow.feedback_path(photo))
    plan = Plan.model_validate_json(json.dumps({'analysis':json.loads((original/'analysis.json').read_text()),
                                               'recipe':recipe.model_dump(mode='json')}))
    path = folder/'approved-plan.json'
    write_json(path,plan)
    workflow.import_plan(photo.id,path)
    return workflow.process(photo.id)
