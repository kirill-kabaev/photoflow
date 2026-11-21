"""AI visual composition handoff and approval-bound, exact integer RAW crops."""
import hashlib
import json
import math
from typing import Annotated, Literal

from PIL import Image
from pydantic import Field, model_validator

from . import imaging
from .models import (ApprovedCrop, Checksum, Crop, DetailRegion, Identifier,
                     Recipe, StrictModel, Text, Unit, Version, VersionManifest)
from .storage import FlowError, digest, json_bytes, now, read_model, write_json

Ratio = Literal['16:9','9:16']
QUESTION = 'Which crop do you approve: 16:9, 9:16, or keep original?'
CATEGORIES = {'subject_pose','face_eyes','headroom','body_hands','background','balance'}


class Composition(StrictModel):
    center: Annotated[list[Unit],Field(min_length=2,max_length=2)]
    scale: Annotated[float,Field(ge=.5,le=1)] = 1.0
    rationale: Text


class CropAnalysis(StrictModel):
    photo_id: Identifier
    source_sha256: Checksum
    version: Version | None = None
    preview_sha256: Checksum
    provider: Literal['codex','test_fixture']
    model: Text
    visually_inspected: Literal[True]
    observations: dict[str,Text]
    protected_regions: Annotated[list[DetailRegion],Field(min_length=1,max_length=16)]
    landscape: Composition
    portrait: Composition
    recommended: Literal['16:9','9:16','keep original']

    @model_validator(mode='after')
    def coverage(self):
        if CATEGORIES-self.observations.keys():
            raise ValueError('Composition analysis must cover pose, face/eyes, headroom, body/hands, background and balance')
        return self


def inputs(workflow, photo, version=None):
    workflow.session.source(photo)
    if version:
        folder,recipe,_=workflow.validate_version(photo,version)
        if recipe.crop != Crop() or recipe.approved_crop:
            raise FlowError('Choose an existing uncropped version for full-frame crop proposals')
        path=folder/'after.jpg'
        dims=json.loads((folder/'decoder.json').read_text())['oriented_dimensions']
    else:
        path=workflow.session.work(photo.id,'analysis_preview.jpg')
        dims=json.loads(workflow.session.work(photo.id,'context.json').read_text())['metadata']['oriented_dimensions']
    return {'photo_id':photo.id,'source_sha256':photo.source_sha256,'version':version,
        'preview':str(path),'preview_sha256':digest(path),'dimensions':dims,
        'instruction':'Inspect this ACTUAL oriented full-frame image before choosing composition. Protect full head/chin, hands and important details. Do not force an impossible crop.'}


def geometry(dimensions, ratio, composition, protected):
    w,h=dimensions; a,b=map(int,ratio.split(':'))
    maximum=min(w//a,h//b)
    if maximum<1:
        raise FlowError('RAW dimensions too small for requested ratio')
    boxes=[[math.floor(r.bounds.left*w),math.floor(r.bounds.top*h),
            math.ceil(r.bounds.right*w),math.ceil(r.bounds.bottom*h)] for r in protected]
    left=min(v[0] for v in boxes); top=min(v[1] for v in boxes)
    right=max(v[2] for v in boxes); bottom=max(v[3] for v in boxes)
    required=max(math.ceil((right-left)/a),math.ceil((bottom-top)/b))
    safe=required<=maximum
    k=min(maximum,max(round(maximum*composition.scale),required)) if safe else maximum
    cw,ch=a*k,b*k
    x=round(composition.center[0]*w-cw/2); y=round(composition.center[1]*h-ch/2)
    if safe:
        x=max(max(0,right-cw),min(min(w-cw,left),x))
        y=max(max(0,bottom-ch),min(min(h-ch,top),y))
    else:
        x=max(0,min(w-cw,x)); y=max(0,min(h-ch,y))
    box=[x,y,x+cw,y+ch]
    excluded=[r.id for r,p in zip(protected,boxes) if not (x<=p[0] and y<=p[1] and x+cw>=p[2] and y+ch>=p[3])]
    return {'aspect_ratio':ratio,'source_box':box,'safe':safe,'excluded_regions':excluded,
        'rationale':composition.rationale,'limitation':None if safe else
        'Head/hands/important protected regions cannot all fit this ratio. Diagnostic preview only; approval disabled.'}


def directory(workflow, photo, request_id):
    if len(request_id)!=32 or any(c not in '0123456789abcdef' for c in request_id):
        raise FlowError('Invalid crop request ID')
    return workflow.session.work(photo.id,'cropping',request_id)


def propose(workflow, analysis):
    photo=workflow.photo(analysis.photo_id)
    context=inputs(workflow,photo,analysis.version)
    if analysis.source_sha256!=photo.source_sha256 or analysis.preview_sha256!=context['preview_sha256']:
        raise FlowError('Composition analysis does not match actual source/preview')
    candidates={ratio:geometry(context['dimensions'],ratio,composition,analysis.protected_regions)
                for ratio,composition in [('16:9',analysis.landscape),('9:16',analysis.portrait)]}
    if analysis.recommended!='keep original' and not candidates[analysis.recommended]['safe']:
        raise FlowError('AI cannot recommend a crop that cuts protected regions')
    identity={'analysis':analysis.model_dump(mode='json'),'dimensions':context['dimensions'],'algorithm':'integer-crop-v1'}
    request_id=hashlib.sha256(json_bytes(identity)).hexdigest()[:32]
    folder=directory(workflow,photo,request_id)
    if not folder.exists():
        folder.mkdir(parents=True)
        with Image.open(context['preview']) as source:
            original=source.convert('RGB')
        original.save(folder/'original.jpg',quality=95,icc_profile=imaging.srgb_profile())
        for ratio,candidate in candidates.items():
            box=candidate['source_box']; w,h=context['dimensions']
            normalized=Crop(left=box[0]/w,top=box[1]/h,right=box[2]/w,bottom=box[3]/h)
            approved=ApprovedCrop(request_id=request_id,decision_sha256='0'*64,aspect_ratio=ratio,
                                  dimensions=context['dimensions'],source_box=box)
            pixels=imaging.crop_box((original.height,original.width,3),normalized,approved,preview=True)
            cropped=original.crop(pixels)
            cropped.save(folder/(ratio.replace(':','x')+'.jpg'),quality=95,icc_profile=imaging.srgb_profile())
            candidate['preview_box']=pixels
        record={'id':request_id,**identity,'candidates':candidates,'preview_sha256':context['preview_sha256'],
            'created_at':now(),'artifacts':{p.name:digest(p) for p in folder.iterdir() if p.is_file()}}
        write_json(folder/'proposal.json',record)
    write_json(workflow.session.work(photo.id,'cropping','current.json'),{'request_id':request_id})
    return presentation(workflow,photo,request_id)


def _pruned_source_context(workflow, photo, analysis):
    """Verify recorded source identity for existing consent after explicit pruning.

    This never supplies pixels or makes a pruned version valid for processing,
    a new visual analysis, or a new crop decision.
    """
    if not analysis.version:
        raise FlowError('Pruned crop source must name a historical version')
    folder=workflow.version_folder(photo,analysis.version)
    record=json.loads((folder/'pruned_artifacts.json').read_text())
    manifest=read_model(folder/'manifest.json',VersionManifest)
    recipe=read_model(folder/'recipe.json',Recipe)
    preview_checksum=manifest.artifacts.get('after.jpg')
    if (record.get('schema_version')!=1 or record.get('photo_id')!=photo.id
            or record.get('version')!=analysis.version or not record.get('user_request','').strip()
            or record.get('source_sha256')!=photo.source_sha256
            or record.get('manifest_sha256')!=digest(folder/'manifest.json')
            or record.get('removed_artifacts',{}).get('after.jpg')!=preview_checksum
            or not preview_checksum or preview_checksum!=analysis.preview_sha256
            or (folder/'after.jpg').exists()
            or manifest.status!='complete' or manifest.version!=analysis.version
            or manifest.source_sha256!=photo.source_sha256
            or recipe.photo_id!=photo.id or recipe.source_sha256!=photo.source_sha256
            or digest(folder/'recipe.json')!=manifest.recipe_sha256
            or digest(folder/'decoder.json')!=manifest.artifacts.get('decoder.json')
            or recipe.crop!=Crop() or recipe.approved_crop):
        raise FlowError('Pruned crop source identity or deletion record changed')
    return {'preview_sha256':preview_checksum,
            'dimensions':json.loads((folder/'decoder.json').read_text())['oriented_dimensions']}


def load(workflow, photo, request_id, *, approved_crop=None):
    folder=directory(workflow,photo,request_id)
    data=json.loads((folder/'proposal.json').read_text())
    analysis=CropAnalysis.model_validate(data['analysis'])
    if approved_crop is not None and (approved_crop.request_id!=request_id
            or digest(folder/'decision.json')!=approved_crop.decision_sha256):
        raise FlowError('Crop decision differs from existing approval')
    try:
        context=inputs(workflow,photo,analysis.version)
    except FileNotFoundError:
        if approved_crop is None:
            raise
        context=_pruned_source_context(workflow,photo,analysis)
    identity={'analysis':analysis.model_dump(mode='json'),'dimensions':context['dimensions'],'algorithm':'integer-crop-v1'}
    if (analysis.photo_id!=photo.id or analysis.source_sha256!=photo.source_sha256
            or analysis.preview_sha256!=context['preview_sha256'] or data['dimensions']!=context['dimensions']
            or data['id']!=request_id or hashlib.sha256(json_bytes(identity)).hexdigest()[:32]!=request_id):
        raise FlowError('Crop proposal identity or visual input changed')
    for ratio,composition in [('16:9',analysis.landscape),('9:16',analysis.portrait)]:
        expected=geometry(data['dimensions'],ratio,composition,analysis.protected_regions)
        if any(data['candidates'][ratio].get(key)!=value for key,value in expected.items()):
            raise FlowError('Crop proposal geometry changed')
    if set(data['artifacts'])!={'original.jpg','16x9.jpg','9x16.jpg'}:
        raise FlowError('Crop proposal is missing required previews')
    for name,checksum in data['artifacts'].items():
        if digest(workflow.session.work(photo.id,'cropping',request_id,name))!=checksum:
            raise FlowError('Crop preview artifact changed')
    return folder,data


def presentation(workflow, photo, request_id):
    folder,data=load(workflow,photo,request_id)
    return {'action':'crop_approval','photo_id':photo.id,'request_id':request_id,
        'recommended':data['analysis']['recommended'],'question':QUESTION,
        'original':str(folder/'original.jpg'),'16:9':str(folder/'16x9.jpg'),'9:16':str(folder/'9x16.jpg'),
        'candidates':data['candidates'],'proposal_sha256':digest(folder/'proposal.json'),
        'instruction':'Show all three real previews in Codex chat, highlight recommendation and limitations. Wait for explicit crop choice. No recipe change, full render, approval or export yet.'}


def decide(workflow, identifier, request_id, choice, actor, message, shown_sha256):
    photo=workflow.photo(identifier); folder,data=load(workflow,photo,request_id)
    if choice not in {'16:9','9:16','keep original'}:
        raise FlowError('Only 16:9, 9:16 or keep original are supported')
    if choice!='keep original' and not data['candidates'][choice]['safe']:
        raise FlowError('This crop cannot preserve protected regions; choose another ratio or keep original')
    if not actor.strip() or not message.strip() or shown_sha256!=digest(folder/'proposal.json'):
        raise FlowError('Crop choice requires explicit user provenance and the shown proposal checksum')
    value={'photo_id':photo.id,'request_id':request_id,'choice':choice,'by':actor,'user_message':message,
           'proposal_sha256':shown_sha256,'created_at':now()}
    path=folder/'decision.json'
    if path.exists():
        old=json.loads(path.read_text())
        if any(old[k]!=value[k] for k in ['photo_id','request_id','choice','by','user_message','proposal_sha256']):
            raise FlowError('Crop decision is immutable; cannot overwrite approval')
    else:
        write_json(path,value)
    selected=selection(workflow,photo,request_id)
    write_json(folder/'approved_crop.json',selected)
    if data['analysis']['version']:
        # Concrete pending recipe, outside immutable version history. Preparing
        # these parameters does not request a render or approve the photograph.
        from .models import Plan
        base,recipe,_=workflow.validate_version(photo,data['analysis']['version'])
        recipe=recipe.model_copy(deep=True)
        recipe.parent_version=data['analysis']['version']
        recipe.feedback_applied='Explicit user crop choice: '+choice+'. '+message
        recipe.feedback_sha256=digest(path)
        plan=Plan.model_validate_json(json.dumps({'analysis':json.loads((base/'analysis.json').read_text()),
                                                  'recipe':recipe.model_dump(mode='json')}))
        pending=apply_selection(plan,selected).recipe
        write_json(folder/'approved_recipe.json',pending)
        selected['pending_recipe']=str(folder/'approved_recipe.json')
    return selected


def selection(workflow, photo, request_id, *, approved_crop=None):
    folder,data=load(workflow,photo,request_id,approved_crop=approved_crop)
    decision=json.loads((folder/'decision.json').read_text())
    choice=decision['choice']
    if (decision['photo_id']!=photo.id or decision['request_id']!=request_id
            or decision['proposal_sha256']!=digest(folder/'proposal.json')
            or not decision['by'].strip() or not decision['user_message'].strip()
            or choice not in {'16:9','9:16','keep original'}):
        raise FlowError('Crop consent provenance mismatch')
    if choice=='keep original':
        return {'action':'crop_selected','approved_crop':None,'crop':Crop().model_dump(),'choice':choice}
    candidate=data['candidates'][choice]
    if not candidate['safe']:
        raise FlowError('Unsafe crop is not eligible for approval')
    approved=ApprovedCrop(request_id=request_id,decision_sha256=digest(folder/'decision.json'),
        aspect_ratio=choice,dimensions=data['dimensions'],source_box=candidate['source_box'])
    x1,y1,x2,y2=approved.source_box; w,h=approved.dimensions
    return {'action':'crop_selected','approved_crop':approved.model_dump(mode='json'),
        'crop':Crop(left=x1/w,top=y1/h,right=x2/w,bottom=y2/h).model_dump(),'choice':choice}


def authorization(workflow, photo, recipe):
    if not recipe.approved_crop:
        return None
    folder=directory(workflow,photo,recipe.approved_crop.request_id)
    approved=selection(workflow,photo,recipe.approved_crop.request_id,approved_crop=recipe.approved_crop)
    if (approved['approved_crop']!=recipe.approved_crop.model_dump(mode='json')
            or approved['crop']!=recipe.crop.model_dump()):
        raise FlowError('Recipe crop differs from explicit user approval')
    return folder


def checkpoint(workflow, photo):
    base=workflow.revision_parent(photo)
    if base:
        _,recipe,_=workflow.validate_version(photo,base)
        if recipe.approved_crop:
            authorization(workflow,photo,recipe)
            return {'action':'crop_selected','approved_crop':recipe.approved_crop.model_dump(mode='json'),
                    'crop':recipe.crop.model_dump(),'choice':recipe.approved_crop.aspect_ratio}
    pointer=workflow.session.work(photo.id,'cropping','current.json')
    if pointer.exists():
        request_id=json.loads(pointer.read_text())['request_id']
        folder,data=load(workflow,photo,request_id)
        if data['analysis']['version']==base:
            return selection(workflow,photo,request_id) if (folder/'decision.json').exists() else presentation(workflow,photo,request_id)
    return {'action':'crop_visual_analysis','photo_id':photo.id,**inputs(workflow,photo,base),
            'analysis_schema':CropAnalysis.model_json_schema()}


def apply_selection(plan, selected):
    """Persist the user's choice in a new plan; trim inspection ROIs to that crop."""
    from .models import Plan
    data=plan.model_dump(mode='json')
    recipe=data['recipe']
    recipe['crop']=selected['crop']
    recipe['approved_crop']=selected['approved_crop']
    bounds=selected['crop']
    details=[]
    for region in recipe['detail_regions']:
        box=region['bounds']
        box={k:(max(box[k],bounds[k]) if k in {'left','top'} else min(box[k],bounds[k])) for k in box}
        if box['left']<box['right'] and box['top']<box['bottom']:
            region['bounds']=box
            details.append(region)
    recipe['detail_regions']=details
    return Plan.model_validate_json(json.dumps(data))
