"""Mission TF direction, rotating lever arm and hostile archive contracts."""
import io
import json
import tarfile

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from inertial_benchmark.data.converters.grandtour import TarArrays, compose_reference


def edge(name, rotation, translation):
    q=rotation.as_quat()
    return {"base_frame_id":"box_base","child_frame_id":name,
            "rotation":dict(zip(("x","y","z","w"),q)),
            "translation":dict(zip(("x","y","z"),translation))}


def test_noncommuting_sensor_mounts_and_rotating_translation():
    rbc=Rotation.from_euler('xyz',[23,-15,31],degrees=True)
    rba=Rotation.from_euler('xyz',[-11,17,54],degrees=True)
    rwc=Rotation.from_euler('xyz',[[4,8,0],[7,-3,90]],degrees=True)
    pbc=np.array([.2,-.1,.4]);pba=np.array([-.3,.6,.1])
    world=np.array([[1,2,3],[-2,4,6.]])
    tf={'cpt7_imu':edge('cpt7_imu',rbc,pbc),'adis16475_imu':edge('adis16475_imu',rba,pba)}
    position,q,lever=compose_reference(world,rwc.as_quat(),tf)
    # Independent homogeneous-matrix composition catches inverse/order errors.
    bc=np.eye(4);bc[:3,:3]=rbc.as_matrix();bc[:3,3]=pbc
    ba=np.eye(4);ba[:3,:3]=rba.as_matrix();ba[:3,3]=pba
    for i in range(2):
        wc=np.eye(4);wc[:3,:3]=rwc[i].as_matrix();wc[:3,3]=world[i]
        wa=wc@np.linalg.inv(bc)@ba
        np.testing.assert_allclose(position[i],wa[:3,3],atol=1e-12)
        np.testing.assert_allclose(Rotation.from_quat(q[i,[1,2,3,0]]).as_matrix(),wa[:3,:3],atol=1e-12)
    assert not np.allclose(position[0]-world[0],position[1]-world[1])
    np.testing.assert_allclose(lever,(np.linalg.inv(bc)@ba)[:3,3],atol=1e-12)


@pytest.mark.parametrize('parent',['base','unknown'])
def test_different_or_unknown_tf_parent_refused(parent):
    e=edge('cpt7_imu',Rotation.identity(),[0,0,0]);e['base_frame_id']=parent
    tf={'cpt7_imu':e,'adis16475_imu':edge('adis16475_imu',Rotation.identity(),[0,0,0])}
    with pytest.raises(ValueError,match='common box_base'):
        compose_reference(np.zeros((1,3)),np.array([[0,0,0,1]]),tf)


@pytest.mark.parametrize('name,link',[('../escape',False),('/absolute',False),('adis_imu/x',True)])
def test_tar_traversal_and_links_refused(tmp_path,name,link):
    p=tmp_path/'unsafe.tar'
    with tarfile.open(p,'w') as t:
        m=tarfile.TarInfo(name)
        if link:m.type=tarfile.SYMTYPE;m.linkname='/etc/passwd';t.addfile(m)
        else:m.size=1;t.addfile(m,io.BytesIO(b'x'))
    with pytest.raises(ValueError,match='unsafe tar'):
        TarArrays(p)


def test_numeric_padded_chunk_is_clipped_to_declared_array(tmp_path):
    numcodecs=pytest.importorskip('numcodecs')
    p=tmp_path/'numeric.tar';values=np.arange(24,dtype='<f8').reshape(8,3)
    meta={'shape':[3,3],'chunks':[8,3],'dtype':'<f8','order':'C','zarr_format':2,
          'filters':None,'compressor':{'id':'blosc','cname':'lz4','clevel':5,'shuffle':1,'blocksize':0}}
    payloads={'imu/timestamp/.zarray':json.dumps(meta).encode(),
              'imu/timestamp/0.0':numcodecs.get_codec(meta['compressor']).encode(values)}
    with tarfile.open(p,'w') as t:
        for name,data in payloads.items():
            m=tarfile.TarInfo(name);m.size=len(data);t.addfile(m,io.BytesIO(data))
    with TarArrays(p) as data:np.testing.assert_array_equal(data.array('imu','timestamp'),values[:3])


def test_object_array_not_deserialized(tmp_path):
    p=tmp_path/'object.tar';meta={'shape':[1],'chunks':[1],'dtype':'|O','order':'C','zarr_format':2,'filters':None}
    b=json.dumps(meta).encode()
    with tarfile.open(p,'w') as t:
        m=tarfile.TarInfo('imu/x/.zarray');m.size=len(b);t.addfile(m,io.BytesIO(b))
    with TarArrays(p) as data:
        with pytest.raises(ValueError,match='unsafe numeric'):data.array('imu','x')


def test_missing_numeric_chunk_is_not_replaced_with_zero_input(tmp_path):
    p=tmp_path/'missing.tar'
    meta={'shape':[3],'chunks':[3],'dtype':'<f8','order':'C',
          'zarr_format':2,'filters':None,'compressor':None}
    b=json.dumps(meta).encode()
    with tarfile.open(p,'w') as t:
        m=tarfile.TarInfo('imu/x/.zarray');m.size=len(b);t.addfile(m,io.BytesIO(b))
    with TarArrays(p) as data:
        with pytest.raises(KeyError):data.array('imu','x')
