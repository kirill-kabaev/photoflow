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
