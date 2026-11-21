import json

import pytest
from PIL import Image

from conftest import process_first
from photoflow.cli import main
from photoflow.manual_export import export_saved
from photoflow.storage import FlowError,write_json


def test_manual_exports_saved_native_pixels_without_review_or_processing(workflow,monkeypatch):
    photo=process_first(workflow)
    snapshot=photo.model_dump()
    monkeypatch.setattr(workflow,'decoder',lambda *_:pytest.fail('No RAW decode allowed'))
    from photoflow import imaging
    monkeypatch.setattr(imaging,'process_pixels',lambda *_ ,**kw:pytest.fail('No reprocessing allowed'))
    result=export_saved(workflow,photo.id,'v001','user','Explicit delivery authorization')
    with Image.open(result['files'][0]) as jpeg:
        assert jpeg.size==(120,80) and jpeg.info['icc_profile']
    assert photo.model_dump()==snapshot
    with pytest.raises(FlowError,match='overwrite'):
        export_saved(workflow,photo.id,'v001','user','Explicit delivery authorization')
    with pytest.raises(FlowError,match='approved'):
        workflow.export(photo.id)


def test_legacy_result_unavailable_never_uses_preview_or_raw(workflow):
    photo=process_first(workflow)
    folder=workflow.session.work(photo.id,'v001')
    data=json.loads((folder/'manifest.json').read_text())
    del data['artifacts']['full_result.npy']
    write_json(folder/'manifest.json',data)
    with pytest.raises(FlowError,match='unavailable'):
        export_saved(workflow,photo.id,'v001','user','Explicit delivery authorization')
    assert not list(workflow.session.path('final').iterdir())


def test_cli_manual_flag_requires_provenance_and_preserves_normal_gate(workflow,capsys):
    photo=process_first(workflow)
    args=['export',str(workflow.session.root),photo.id]
    assert main(args)==1
    assert main(args+['--force','--version','v001'])==1
    assert main(args+['--force','--version','v001','--by','user','--reason','Explicit manual export'])==0


def test_corrupt_full_result_is_rejected(workflow):
    photo=process_first(workflow)
    workflow.session.work(photo.id,'v001','full_result.npy').write_bytes(b'corrupt')
    with pytest.raises(FlowError,match='modified'):
        export_saved(workflow,photo.id,'v001','user','Explicit manual export')


def test_explicit_recipe_replay_renders_once_without_touching_version_or_queue(workflow,monkeypatch):
    photo=process_first(workflow)
    folder=workflow.session.work(photo.id,'v001')
    data=json.loads((folder/'manifest.json').read_text())
    del data['artifacts']['full_result.npy'];write_json(folder/'manifest.json',data)
    snapshot={p.name:p.read_bytes() for p in folder.iterdir() if p.is_file()}
    queue=workflow.session.path('queue.json').read_bytes()
    decoder=workflow.decoder;calls=[]
    def counted(path,settings):
        calls.append(settings.model_dump())
        return decoder(path,settings)
    monkeypatch.setattr(workflow,'decoder',counted)
    result=export_saved(workflow,photo.id,'v001','user','Explicit saved recipe render',True)
    assert len(calls)==1
    assert {p.name:p.read_bytes() for p in folder.iterdir() if p.is_file()}==snapshot
    assert workflow.session.path('queue.json').read_bytes()==queue
    with Image.open(result['files'][0]) as jpeg:assert jpeg.size==(120,80)
    record=json.loads(workflow.session.path('final',photo.id+'.manual-export.json').read_text())
    assert record['raw_reprocessed'] and not record['artistically_approved']
    with pytest.raises(FlowError,match='overwrite'):
        export_saved(workflow,photo.id,'v001','user','Explicit saved recipe render',True)
    assert len(calls)==1


def test_replay_cli_requires_force(workflow):
    photo=process_first(workflow)
    assert main(['export',str(workflow.session.root),photo.id,'--render-saved-recipe'])==1


def test_separate_delivery_folder_preserves_previous_jpeg(workflow):
    from photoflow.storage import digest
    photo=process_first(workflow)
    old=export_saved(workflow,photo.id,'v001','user','Original delivery')['files'][0]
    from pathlib import Path
    checksum=digest(Path(old))
    new=export_saved(workflow,photo.id,'v001','user','New delivery',output_subdir='raw_sync_test')
    assert Path(new['files'][0]).parent.name=='raw_sync_test'
    assert digest(Path(old))==checksum
    with pytest.raises(FlowError,match='overwrite'):
        export_saved(workflow,photo.id,'v001','user','New delivery',output_subdir='raw_sync_test')
    with pytest.raises(FlowError,match='identifier'):
        export_saved(workflow,photo.id,'v001','user','New delivery',output_subdir='../escape')


def test_max_quality_cli_preserves_recipe_queue_and_prior_delivery(workflow,monkeypatch):
    from pathlib import Path
    from PIL.JpegImagePlugin import get_sampling
    from photoflow import imaging
    from photoflow.storage import digest
    photo=process_first(workflow)
    original=Path(export_saved(workflow,photo.id,'v001','user','Original delivery')['files'][0])
    old_hash=digest(original)
    folder=workflow.session.work(photo.id,'v001')
    snapshot={p.name:p.read_bytes() for p in folder.iterdir() if p.is_file()}
    queue=workflow.session.path('queue.json').read_bytes()
    monkeypatch.setattr(imaging,'decode_raw',lambda *_:pytest.fail('No RAW decode allowed'))
    monkeypatch.setattr(imaging,'process_pixels',lambda *_,**kw:pytest.fail('No reprocessing allowed'))
    args=['export',str(workflow.session.root),photo.id,'--jpeg-quality','100']
    assert main(args)==1  # Normal approval requirements remain in force.
    assert main(args+['--force','--version','v001','--by','user','--reason','Maximum quality delivery','--output-subdir','maximum_quality'])==0
    output=workflow.session.path('final','maximum_quality',photo.id+'.jpg')
    with Image.open(output) as jpeg:
        jpeg.load()
        assert jpeg.size==(120,80) and get_sampling(jpeg)==0
        assert all(value==1 for table in jpeg.quantization.values() for value in table)
        imaging.validate_srgb_profile(jpeg.info['icc_profile'])
    record=json.loads(output.with_suffix('.manual-export.json').read_text())
    assert record['jpeg_quality']==100 and record['recipe_jpeg_quality']==95
    assert not record['raw_reprocessed'] and not record['artistically_approved']
    assert {p.name:p.read_bytes() for p in folder.iterdir() if p.is_file()}==snapshot
    assert workflow.session.path('queue.json').read_bytes()==queue
    assert digest(original)==old_hash
    with pytest.raises(FlowError,match='overwrite'):
        export_saved(workflow,photo.id,'v001','user','Maximum quality delivery',output_subdir='maximum_quality',jpeg_quality=100)


@pytest.mark.parametrize('quality',[89,101,95.5,True])
def test_manual_quality_override_rejects_invalid_values(workflow,quality):
    photo=process_first(workflow)
    with pytest.raises(FlowError,match='JPEG quality'):
        export_saved(workflow,photo.id,'v001','user','Explicit delivery',jpeg_quality=quality)
    assert not list(workflow.session.path('final').iterdir())
