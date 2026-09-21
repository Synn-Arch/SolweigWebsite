"""Parser contract tests; forwarding mocks do not verify numerical execution."""
import argparse
import json
from pathlib import Path
import sys

import pytest
from solweig_light import __version__
from solweig_light import cli

ROOT=Path(__file__).resolve().parents[2]


def test_all_pinned_flags_defaults_types_and_requirements():
    snapshot=json.loads((ROOT/'reports/contract_snapshot.json').read_text())['cli']['flags']
    parser=cli.build_parser()
    actions={flag:action for action in parser._actions for flag in action.option_strings if flag!='--help' and flag!='-h'}
    expected_flags={flag for record in snapshot for flag in record['flags']}
    assert set(actions)==expected_flags
    assert len(expected_flags)==27
    for record in snapshot:
        action=actions[record['flags'][0]]
        if record['action']=='version':
            assert isinstance(action,argparse._VersionAction)
            assert action.version==f'solweig_light {__version__}'
            continue
        assert action.default==record['default']
        assert action.required==bool(record['required'])
        expected_type={'int':int,'str2bool':cli.str2bool,None:None}[record['type']]
        assert action.type is expected_type


@pytest.mark.parametrize('value',[True,'yes','true','t','1','YES','True'])
def test_true_spellings(value): assert cli.str2bool(value) is True


@pytest.mark.parametrize('value',[False,'no','false','f','0','NO','False'])
def test_false_spellings(value): assert cli.str2bool(value) is False


@pytest.mark.parametrize('value',['','on','off','2',' true ','y'])
def test_invalid_boolean(value):
    with pytest.raises(argparse.ArgumentTypeError,match=r'Boolean value expected \(True/False\)'):cli.str2bool(value)


def invoke(monkeypatch,args):
    calls=[]
    monkeypatch.setattr(cli,'thermal_comfort',lambda **kwargs:calls.append(kwargs))
    monkeypatch.setattr(sys,'argv',['solweig-light',*args])
    result=cli.main()
    assert result is None
    return calls


def test_exact_default_forwarding(monkeypatch,tmp_path):
    met=tmp_path/'met.txt';met.write_text('Parser-only mock forcing file.\n')
    result=invoke(monkeypatch,['--base_path',str(tmp_path),'--date','2020-07-18','--own_metfile',str(met)])
    assert result==[dict(base_path=str(tmp_path),selected_date_str='2020-07-18',building_dsm_filename='Building_DSM.tif',dem_filename='DEM.tif',trees_filename='Trees.tif',landcover_filename=None,ERA_5_z0_find=False,tile_size=3600,overlap=20,use_own_met=True,own_met_file=str(met),data_source_type=None,data_folder=None,start_time=None,end_time=None,use_uhi=True,save_tmrt=True,save_svf=False,save_kup=False,save_kdown=False,save_lup=False,save_ldown=False,save_shadow=False,save_wbgt=False,save_ta=False,save_wind=False)]


def test_all_explicit_flag_forwarding(monkeypatch,tmp_path):
    met=tmp_path/'met.txt';met.write_text('Parser-only mock forcing file.\n')
    args=['--base_path',str(tmp_path),'--date','2020-07-18','--building_dsm','b.tif','--dem','d.tif','--trees','t.tif','--landcover','lc.tif','--tile_size','64','--overlap','0','--use_own_met','yes','--own_metfile',str(met),'--data_source_type','ERA5','--data_folder',str(tmp_path),'--start','start','--end','end','--era5_z0_find','false','--use_uhi','false']
    for name in ['tmrt','svf','kup','kdown','lup','ldown','shadow','wbgt','ta','wind']:args.extend(['--save_'+name,'true'])
    call=invoke(monkeypatch,args)[0]
    assert call['building_dsm_filename']=='b.tif' and call['dem_filename']=='d.tif' and call['trees_filename']=='t.tif' and call['landcover_filename']=='lc.tif'
    assert call['tile_size']==64 and call['overlap']==0
    assert call['start_time']=='start' and call['end_time']=='end'
    assert call['data_source_type']=='ERA5' and call['data_folder']==str(tmp_path)
    assert call['ERA_5_z0_find'] is False and call['use_uhi'] is False
    assert all(call['save_'+name] is True for name in ['tmrt','svf','kup','kdown','lup','ldown','shadow','wbgt','ta','wind'])


def test_roughness_default_tracks_data_folder(monkeypatch,tmp_path):
    met=tmp_path/'met.txt';met.write_text('Parser-only mock forcing file.\n')
    call=invoke(monkeypatch,['--base_path',str(tmp_path),'--date','2020-07-18','--own_metfile',str(met),'--data_folder',str(tmp_path)])[0]
    assert call['ERA_5_z0_find'] is True


@pytest.mark.parametrize('extra,message',[
    ([], '--own_metfile is required when --use_own_met=True'),
    (['--own_metfile','/nonexistent/met.txt'], 'File not found: /nonexistent/met.txt'),
    (['--use_own_met','false'], '--data_source_type is required when --use_own_met=False'),
    (['--use_own_met','false','--data_source_type','ERA5'], '--data_folder is required when --use_own_met=False'),
    (['--use_own_met','false','--data_source_type','ERA5','--data_folder','/nonexistent/met'], 'Directory not found: /nonexistent/met'),
    (['--use_own_met','false','--data_source_type','ERA5','--data_folder','{folder}'], '--start and --end are required when using --data_source_type'),
    (['--own_metfile','{met}','--era5_z0_find','true'], '--era5_z0_find=True requires --data_folder containing data_stream-oper_stepType-instant.nc'),
])
def test_validation_errors(monkeypatch,tmp_path,capsys,extra,message):
    met=tmp_path/'met.txt';met.write_text('Parser-only mock forcing file.\n')
    args=['--base_path',str(tmp_path),'--date','2020-07-18',*[v.format(folder=str(tmp_path),met=str(met)) for v in extra]]
    monkeypatch.setattr(sys,'argv',['solweig-light',*args])
    with pytest.raises(SystemExit) as error:cli.main()
    assert error.value.code==2
    assert message in capsys.readouterr().err


def test_version_is_truthful(monkeypatch,capsys):
    monkeypatch.setattr(sys,'argv',['solweig-light','--version'])
    with pytest.raises(SystemExit) as error:cli.main()
    assert error.value.code==0
    assert capsys.readouterr().out==f'solweig_light {__version__}\n'
