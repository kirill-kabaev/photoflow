"""Synthetic composition handoff; no fixture is claimed as real visual AI."""
import json

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from conftest import make_plan,process_first,import_review
from test_director import complete_plan
from photoflow import cropping,imaging
from photoflow.cli import main
from photoflow.director import Director,DirectorConfig
from photoflow.models import ApprovedCrop,Crop,Recipe
from photoflow.storage import FlowError,digest,write_json


def analysis(workflow,photo,version=None,protect_full=False):
    context=cropping.inputs(workflow,photo,version)
    return cropping.CropAnalysis.model_validate({
        'photo_id':photo.id,'source_sha256':photo.source_sha256,'version':version,
        'preview_sha256':context['preview_sha256'],'provider':'test_fixture','model':'synthetic-no-real-AI',
        'visually_inspected':True,'observations':{k:'Explicit synthetic visual-analysis fixture' for k in cropping.CATEGORIES},
        'protected_regions':[{'id':'head_hands','bounds':{} if protect_full else
            {'left':.35,'top':.3,'right':.65,'bottom':.7},'description':'Synthetic protected head/hands'}],
        'landscape':{'center':[.5,.5],'rationale':'Synthetic landscape framing'},
        'portrait':{'center':[.5,.5],'rationale':'Synthetic portrait framing'},
        'recommended':'keep original' if protect_full else '9:16'})


def approve(workflow,photo,proposal,choice='9:16'):
    return cropping.decide(workflow,photo.id,proposal['request_id'],choice,'synthetic-user',choice,proposal['proposal_sha256'])


@pytest.mark.parametrize('dimensions',[[4168,6240],[6240,4168],[121,81],[81,121]])
@pytest.mark.parametrize('ratio',['16:9','9:16'])
def test_integer_geometry_handles_orientation_odd_sizes_without_stretch(dimensions,ratio):
    from photoflow.models import DetailRegion
    protected=[DetailRegion(id='head',bounds=Crop(left=.4,top=.4,right=.6,bottom=.6),description='Synthetic')]
    candidate=cropping.geometry(dimensions,ratio,cropping.Composition(center=[.6,.4],rationale='Synthetic'),protected)
    x1,y1,x2,y2=candidate['source_box']; a,b=map(int,ratio.split(':'))
    assert (x2-x1)*b==(y2-y1)*a and 0<=x1<x2<=dimensions[0] and 0<=y1<y2<=dimensions[1]
    assert candidate['safe']


def test_visual_proposals_never_change_recipe_queue_or_render_budget(workflow):
    photo=process_first(workflow)
    old=workflow.session.path('queue.json').read_bytes()
    source=workflow.session.work(photo.id,'v001','recipe.json').read_bytes()
    result=cropping.propose(workflow,analysis(workflow,photo,'v001'))
    assert result['question']==cropping.QUESTION
    for ratio in ['16:9','9:16']:
        with Image.open(result[ratio]) as preview:
            a,b=map(int,ratio.split(':')); assert preview.width*b==preview.height*a
    assert workflow.session.path('queue.json').read_bytes()==old
    assert workflow.session.work(photo.id,'v001','recipe.json').read_bytes()==source
    assert photo.active_attempt_count==1
    assert not (workflow.session.work(photo.id,'cropping',result['request_id'])/'decision.json').exists()


def test_impossible_ratio_is_diagnostic_and_cannot_be_approved(workflow):
    photo=process_first(workflow)
    result=cropping.propose(workflow,analysis(workflow,photo,'v001',protect_full=True))
    assert all(not v['safe'] for v in result['candidates'].values())
    with pytest.raises(FlowError,match='cannot preserve'): approve(workflow,photo,result)
    assert approve(workflow,photo,result,'keep original')['approved_crop'] is None


def test_unapproved_crop_plan_is_refused_before_render(workflow):
    photo=workflow.queue.photos[0]; workflow.prepare(photo.id)
    path=make_plan(workflow,photo)
    data=json.loads(path.read_text()); data['recipe']['crop']={'left':.1,'right':.9}
    write_json(path,data)
    with pytest.raises(FlowError,match='explicit user approval'): workflow.import_plan(photo.id,path)
    assert photo.active_attempt_count==0


def test_approved_ratio_flows_to_raw_version_native_details_and_export(workflow):
    photo=workflow.queue.photos[0]; workflow.prepare(photo.id)
    proposal=cropping.propose(workflow,analysis(workflow,photo))
    selected=approve(workflow,photo,proposal)
    path=make_plan(workflow,photo)
    from photoflow.models import Plan
    plan=cropping.apply_selection(Plan.model_validate_json(path.read_bytes()),selected)
    write_json(path,plan); workflow.import_plan(photo.id,path); workflow.process(photo.id)
    folder,recipe,manifest=workflow.validate_version(photo,'v001')
    assert recipe.approved_crop.aspect_ratio=='9:16'
    assert {'cropping/proposal.json','cropping/decision.json'}<=manifest['artifacts'].keys()
    result=imaging.process_pixels(workflow.decoded(photo,recipe.raw).pixels,recipe)
    assert result.shape[1]*16==result.shape[0]*9
    with pytest.raises(FlowError,match='approved'): workflow.export(photo.id)
    import_review(workflow,photo); workflow.approve(photo.id,'v001','synthetic-user')
    exported=workflow.export(photo.id)
    assert exported['validation']['dimensions'][0]*16==exported['validation']['dimensions'][1]*9
    assert photo.active_attempt_count==1


def test_pixel_crop_is_exact_slice_not_resampling(pixels):
    approved=ApprovedCrop(request_id='a'*32,decision_sha256='b'*64,aspect_ratio='16:9',
        dimensions=[120,80],source_box=[4,8,116,71])
    crop=Crop(left=4/120,top=8/80,right=116/120,bottom=71/80)
    actual=imaging.crop_image(pixels,crop,approved)
    assert np.array_equal(actual,pixels[8:71,4:116]) and actual.shape==(63,112,3)
    with pytest.raises(FlowError,match='dimensions'): imaging.crop_image(pixels[:40],crop,approved)


def test_wrong_photo_tampered_preview_or_coordinates_cannot_be_approved(workflow):
    photo=process_first(workflow); result=cropping.propose(workflow,analysis(workflow,photo,'v001'))
    with pytest.raises(FlowError,match='provenance'):
        cropping.decide(workflow,photo.id,result['request_id'],'9:16','synthetic','9:16','0'*64)
    folder,_=cropping.load(workflow,photo,result['request_id'])
    data=json.loads((folder/'proposal.json').read_text()); data['candidates']['9:16']['source_box'][0]+=1
    write_json(folder/'proposal.json',data)
    with pytest.raises(FlowError,match='geometry'): approve(workflow,photo,result)
    with pytest.raises(ValidationError):
        ApprovedCrop(request_id='a'*32,decision_sha256='b'*64,aspect_ratio='4:5',dimensions=[120,80],source_box=[0,0,120,80])


def test_director_requires_visual_analysis_then_choice_before_first_full_pass(workflow):
    director=Director(workflow); photo=workflow.queue.photos[0]
    action=director.start('Synthetic explicit smart crop request',photo.id,DirectorConfig(smart_crop=True))
    assert action['action']=='crop_visual_analysis' and photo.active_attempt_count==0
    path=complete_plan(workflow,photo)
    with pytest.raises(FlowError,match='wait for explicit'): workflow.import_plan(photo.id,path)
    result=cropping.propose(workflow,analysis(workflow,photo))
    assert director.advance()['action']=='crop_approval'
    selected=approve(workflow,photo,result)
    assert director.advance()['action']=='analyze_and_plan'
    workflow.import_plan(photo.id,path)
    assert director.advance()['action']=='visual_review'
    assert photo.active_attempt_count==1
    _,recipe,_=workflow.validate_version(photo,'v001')
    assert recipe.approved_crop.source_box==selected['approved_crop']['source_box']
    for region in recipe.detail_regions:
        assert region.bounds.left>=recipe.crop.left and region.bounds.right<=recipe.crop.right


def test_keep_original_and_override_are_independent_of_ai_recommendation(workflow):
    photo=process_first(workflow)
    proposal=cropping.propose(workflow,analysis(workflow,photo,'v001'))
    chosen=approve(workflow,photo,proposal,'16:9')
    assert chosen['approved_crop']['aspect_ratio']=='16:9' and proposal['recommended']=='9:16'
    pending=Recipe.model_validate_json(open(chosen['pending_recipe']).read())
    assert pending.approved_crop.aspect_ratio=='16:9' and pending.parent_version=='v001'
    with pytest.raises(FlowError,match='immutable'): approve(workflow,photo,proposal,'keep original')
    assert photo.current_version=='v001' and photo.approved_version is None


def test_detail_crop_boundary_uses_approved_integers_not_float_truncation(tmp_path):
    from photoflow.models import DetailRegion
    w,h=731,1111
    approved=ApprovedCrop(request_id='a'*32,decision_sha256='b'*64,aspect_ratio='9:16',
        dimensions=[w,h],source_box=[73,67,613,1027])
    crop=Crop(left=73/w,top=67/h,right=613/w,bottom=1027/h)
    recipe=Recipe(photo_id='synthetic',source_sha256='c'*64,style='synthetic',rationale='Synthetic boundary',
        crop=crop,approved_crop=approved,detail_regions=[DetailRegion(id='face',bounds=crop,description='Synthetic full crop')])
    pixels=np.full((960,540,3),.2,np.float32)
    details=imaging.save_details(pixels,pixels,(h,w,3),recipe,tmp_path)
    assert details['face']['source_box']==approved.source_box
    assert details['face']['preview_dimensions']==[540,960]


def prune_crop_source(workflow, photo):
    """Synthetic explicit deletion, retaining the original manifest and metadata."""
    folder=workflow.session.work(photo.id,'v001')
    record={'schema_version':1,'photo_id':photo.id,'version':'v001',
        'user_request':'Synthetic user explicitly deletes old photo versions',
        'source_sha256':photo.source_sha256,'manifest_sha256':digest(folder/'manifest.json'),
        'removed_artifacts':{'after.jpg':digest(folder/'after.jpg')}}
    write_json(folder/'pruned_artifacts.json',record)
    (folder/'after.jpg').unlink()
    return folder


def test_existing_crop_consent_survives_explicit_source_pruning_and_exports(workflow):
    photo=process_first(workflow)
    proposal=cropping.propose(workflow,analysis(workflow,photo,'v001'))
    selected=approve(workflow,photo,proposal)
    workflow.request_revision(photo.id)
    path=make_plan(workflow,photo,parent='v001',feedback=workflow.feedback_path(photo))
    from photoflow.models import Plan
    write_json(path,cropping.apply_selection(Plan.model_validate_json(path.read_bytes()),selected))
    workflow.import_plan(photo.id,path); workflow.process(photo.id)
    _,recipe,_=workflow.validate_version(photo,'v002')
    folder=prune_crop_source(workflow,photo)
    queue_before=workflow.session.path('queue.json').read_bytes()
    assert cropping.authorization(workflow,photo,recipe).name==proposal['request_id']
    assert workflow.session.path('queue.json').read_bytes()==queue_before
    # The exception only verifies existing consent; it never admits a missing
    # image as a complete version, a visual input, or a new crop decision.
    with pytest.raises(FileNotFoundError): workflow.validate_version(photo,'v001')
    with pytest.raises(FileNotFoundError): cropping.inputs(workflow,photo,'v001')
    with pytest.raises(FileNotFoundError): approve(workflow,photo,proposal)
    import_review(workflow,photo); workflow.approve(photo.id,'v002','synthetic-user')
    result=workflow.export(photo.id)
    width,height=result['validation']['dimensions']
    assert width*16==height*9 and photo.active_attempt_count==2
    assert not (folder/'after.jpg').exists()


@pytest.mark.parametrize('tamper',[
    'missing_record','record_photo','record_source','record_manifest','record_preview',
    'recipe','decoder','proposal_preview','decision','existing_preview','raw',
])
def test_pruned_source_does_not_bypass_crop_integrity(workflow,tamper):
    photo=process_first(workflow)
    proposal=cropping.propose(workflow,analysis(workflow,photo,'v001'))
    selected=approve(workflow,photo,proposal)
    recipe=Recipe.model_validate_json(open(selected['pending_recipe']).read())
    folder=prune_crop_source(workflow,photo)
    record_path=folder/'pruned_artifacts.json'
    record=json.loads(record_path.read_text())
    if tamper=='missing_record': record_path.unlink()
    elif tamper.startswith('record_'):
        key={'record_photo':'photo_id','record_source':'source_sha256',
             'record_manifest':'manifest_sha256','record_preview':'removed_artifacts'}[tamper]
        record[key]={'after.jpg':'0'*64} if key=='removed_artifacts' else '0'*64
        write_json(record_path,record)
    elif tamper in {'recipe','decoder'}:
        (folder/(tamper+'.json')).write_text('{}')
    elif tamper=='proposal_preview':
        (cropping.directory(workflow,photo,proposal['request_id'])/'original.jpg').write_bytes(b'tampered')
    elif tamper=='decision':
        decision=cropping.directory(workflow,photo,proposal['request_id'])/'decision.json'
        data=json.loads(decision.read_text()); data['user_message']='changed';write_json(decision,data)
    elif tamper=='existing_preview': (folder/'after.jpg').write_bytes(b'tampered')
    elif tamper=='raw': workflow.session.path('raw',photo.filename).write_bytes(b'tampered synthetic source')
    with pytest.raises((FlowError,FileNotFoundError,ValidationError)):
        cropping.authorization(workflow,photo,recipe)
