"""Scientific 200 Hz geometry/gap/provenance tests, independent of Torch."""
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from inertial_benchmark.data.converters import _kitti_raw100hz as r
from inertial_benchmark.data.converters import kitti as k
from inertial_benchmark.data.convert import raw_to_sequence
from inertial_benchmark.data.format import load_sequence,save_sequence,validate
from inertial_benchmark.data.resample import resample_streams
from inertial_benchmark.data.views import SequenceView,ViewConfig,window_valid_mask


def synthetic(n=401):
    p=np.zeros((n,30));p[:,0:3]=[49,8,100];p[:,13]=9.80655;p[:,25:]=[4,10,6,6,6]
    c=r.analyze(p,np.arange(n,dtype=np.int64)*10_000_000,np.arange(n,dtype=np.int64))
    c.position=np.stack((np.arange(n)*.03,np.arange(n)*.01,np.arange(n)*.002),1)
    return c


def as_raw(c,root):
    name=r.SPLITS['train'];folder=root/'candidate';folder.mkdir(exist_ok=True)
    r.write_candidate(folder/(name+'.h5'),c,{'engineering_synthetic':True})
    (root/'quality-manifest.json').write_text(json.dumps({'engineering_synthetic':True}))
    return k.make_raw(c,name,0,len(c.packet),name+'_clock00',root)


class TestKittiNative(unittest.TestCase):
    def test_backjump_splits_without_dropping_and_positive_gap_stays_masked(self):
        c=synthetic(6);c.clock_ns=np.array([0,10,5,15,200_000_000,210_000_000],np.int64)
        c=r.analyze(c.packet,c.clock_ns,c.source_index)
        self.assertEqual(k.branches(c),[(0,2),(2,6)])
        self.assertEqual(sum(b-a for a,b in k.branches(c)),6)
        self.assertTrue(c.clock_breaks[3]&2)

    def test_shared200Hz_linear_position_and_q_SLERP_nonplanar_geometry(self):
        c=synthetic();yaw=np.linspace(0,.4,len(c.packet));c.packet[:,5]=yaw
        _,c.quaternion=r.pose_enu(c.packet)
        t=np.arange(len(c.packet))*.01
        x=resample_streams(t,c.packet[:,17:20],c.packet[:,11:14],t,c.position,c.quaternion,rate=200)
        self.assertTrue(x.valid_imu.all());self.assertTrue(x.valid_pose.all())
        np.testing.assert_allclose(x.arrays['position'][1],(c.position[0]+c.position[1])/2,atol=1e-12)
        self.assertAlmostEqual(float(x.arrays['orientation'][1,3]),math.sin(yaw[1]/4),places=12)
        self.assertEqual(x.info['imu']['method'],'resample_poly')
        self.assertEqual(len(x.timestamp),801)

    def test_long_gap_masks_prevent_valid_window_crossing_and_FIR_support(self):
        c=synthetic();t=np.arange(len(c.packet))*.01;t[200:]+=1.
        x=resample_streams(t,c.packet[:,17:20],c.packet[:,11:14],t,c.position,c.quaternion,rate=200)
        gap=(x.timestamp>1.99)&(x.timestamp<3.)
        self.assertFalse(x.valid_imu[gap].any());self.assertFalse(x.valid_pose[gap].any())
        starts=np.arange(len(x.timestamp)-400)
        crossing=(x.timestamp[starts]<2.)&(x.timestamp[starts+399]>3.)
        self.assertTrue(crossing.any())
        self.assertFalse(window_valid_mask(x.valid_imu&x.valid_pose,starts,400)[crossing].any())
        self.assertGreater(np.sum(~x.valid_imu),np.sum(~x.valid_pose))

    def test_short_clock_jitter_is_supported_without_time_sorting(self):
        c=synthetic();t=np.arange(len(c.packet))*.01;t[100]+=.003
        x=resample_streams(t,c.packet[:,17:20],c.packet[:,11:14],t,c.position,c.quaternion,rate=200)
        self.assertTrue(x.valid_imu.all());self.assertTrue(x.valid_pose.all())
        self.assertEqual(x.info['dropped_timestamps'],{'imu':0,'pose':0})

    def test_categorical_mode_transition_never_becomes_numeric_interpolated5(self):
        c=synthetic(3);c.packet[0,27:]=4;c.packet[1:,27:]=6
        p=k.grid_provenance(c,0,3,np.array([0.,.005,.01]))
        self.assertEqual(p['categorical_flags_left'][1,2],4)
        self.assertEqual(p['categorical_flags_right'][1,2],6)
        self.assertEqual(p['linear_reference_alpha'][1],.5)
        self.assertFalse(np.any(p['categorical_flags_left'][:,2:]==5))

    def test_source_endpoint_indices_and_exact_civil_ns_provenance(self):
        c=synthetic(3);c.clock_ns+=1_317_000_000_123_456_789
        p=k.grid_provenance(c,0,3,np.array([0.,.005,.01]))
        np.testing.assert_array_equal(p['source_left_index'],[0,0,1])
        np.testing.assert_array_equal(p['source_right_index'],[0,1,1])
        self.assertEqual(int(p['source_left_civil_ns'][1]),int(c.clock_ns[0]))
        self.assertEqual(p['is_between_original_samples'].tolist(),[False,True,False])

    def test_unknown_vehicle_leverarm_is_metadata_not_invented_or_blanket_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=as_raw(synthetic(),Path(tmp));seq,rec=raw_to_sequence(raw,k.NAME,k.NAME+'@'+k.VERSION,k.LICENSE,
                                      {'rate':200.,'gap_threshold':.05,'min_duration':2.})
            self.assertEqual(rec['status'],'accepted');self.assertTrue(validate(seq).ok)
            self.assertTrue(math.isnan(seq.attrs['start_time_unix']))
            self.assertIn('unknown',seq.attrs['car_to_IMU_lever_arm'])
            self.assertFalse(seq.attrs['reference_is_independent_ground_truth'])
            self.assertEqual(seq.group_id,r.SPLITS['train'])

    def test_native_serialization_XYZ_targets_and_same_prediction_XY_slicing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=as_raw(synthetic(),root)
            seq,rec=raw_to_sequence(raw,k.NAME,k.NAME+'@'+k.VERSION,k.LICENSE,{'rate':200.,'gap_threshold':.05,'min_duration':2.})
            path=save_sequence(root/'native.h5',seq);reload=load_sequence(path,validate=True)
            np.testing.assert_array_equal(seq.position,reload.position);np.testing.assert_array_equal(seq.orientation,reload.orientation)
            xyz=SequenceView(reload,ViewConfig(dims=3,target='avg_velocity'))
            xy=SequenceView(reload,ViewConfig(dims=2,target='avg_velocity'))
            starts=np.array([0,200],np.int64)
            np.testing.assert_array_equal(xyz.targets(starts)[:,:2],xy.targets(starts))
            np.testing.assert_allclose(xyz.targets(starts)[:,2],.2,atol=1e-7)

    def test_upstream_interpolated_packets_are_invalid_on_derived_grid(self):
        c=synthetic();c.packet[100:110,27:]=-1;c=r.analyze(c.packet,c.clock_ns,c.source_index)
        t=np.arange(len(c.packet))*.01
        x=resample_streams(t,c.packet[:,17:20],c.packet[:,11:14],t,c.position,c.quaternion,
                            imu_valid=c.input_eligible,pose_valid=c.reference_candidate,rate=200)
        interval=(x.timestamp>=1.)&(x.timestamp<=1.09)
        self.assertFalse(x.valid_imu[interval].any());self.assertFalse(x.valid_pose[interval].any())

    def test_empty_derived_output_or_bad_admission_has_explicit_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'quality-manifest.json').write_text(json.dumps({'state':'not_audited','rows':41547}))
            with self.assertRaises(ValueError):k.load_candidate(root,r.SPLITS['test'])
            with self.assertRaises(FileExistsError):k.convert_derived(root,root)


if __name__=='__main__':unittest.main()
