"""NumPy/h5py-only candidate science checks; no training or dataset registration."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from inertial_benchmark.data.converters import _kitti_raw100hz as k


def packets(n):
    p = np.zeros((n, 30), dtype=np.float64)
    p[:, :3] = [49, 8, 120]
    p[:, 13] = 9.80655
    p[:, 25:] = [4, 10, 6, 6, 6]
    return p


class TestKittiRaw100Hz(unittest.TestCase):
    def test_nanoseconds_and_civil_clock_keep_original_backjump(self):
        a = k.civil_nanoseconds("2011-09-30 12:00:00.123456789")
        b = k.civil_nanoseconds("2011-09-30 12:00:00.123456790")
        self.assertEqual(b-a, 1)
        self.assertLess(k.civil_nanoseconds("2011-09-30 11:59:59.9"), a)
        with self.assertRaises(ValueError):
            k.civil_nanoseconds("2011-09-30 12:00:00.123456789Z")

    def test_body_field_slices_preserve_SI_without_surface_axis_substitution(self):
        p = packets(2)
        p[:,11:14] = [1,2,9.8]
        p[:,14:17] = [44,45,46]
        p[:,17:20] = [.1,.2,.3]
        p[:,20:23] = [77,78,79]
        c = k.analyze(p,np.array([0,10_000_000],np.int64),np.arange(2,dtype=np.int64))
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/"candidate.h5"
            k.write_candidate(target,c,{"engineering_synthetic":True})
            with h5py.File(target) as h5:
                np.testing.assert_array_equal(h5["imu/gyro_rad_s"][:],p[:,17:20])
                np.testing.assert_array_equal(h5["imu/acc_m_s2"][:],p[:,11:14])

    def test_ENU_velocity_and_positive_yaw_front_to_north(self):
        p = packets(2)
        p[:,[7,6,10]] = [5,6,7]
        p[:,5] = np.pi/2
        _,q = k.pose_enu(p)
        R = k.rotation(q)
        np.testing.assert_allclose(R[0]@[1,0,0],[0,1,0],atol=1e-14)
        np.testing.assert_allclose(R[0]@[0,0,1],[0,0,1],atol=1e-14)

    def test_mercator_world_is_not_rotated_to_first_heading(self):
        p=packets(3)
        p[:,5]=np.pi/2
        p[1,1]+=1e-5
        p[2,0]+=1e-5
        position,_=k.pose_enu(p)
        self.assertGreater(position[1,0],0)
        self.assertAlmostEqual(position[1,1],0)
        self.assertGreater(position[2,1],0)

    def test_negative_modes_reject_upstream_interpolation_but_mode0_is_distinct(self):
        p=packets(4)
        p[1,27:]=-1
        p[2,29]=0
        c=k.analyze(p,np.arange(4,dtype=np.int64)*10_000_000,np.arange(4,dtype=np.int64))
        self.assertFalse(c.input_eligible[1])
        self.assertTrue(c.reference_candidate[2])
        self.assertFalse(c.gnss_supported[2])
        self.assertTrue(c.reasons[1]&16)
        self.assertFalse(c.reasons[2]&16)

    def test_clock_breaks_keep_every_source_row_without_crossing_gaps(self):
        ns=np.array([0,10_000_000,5_000_000,100_000_000,110_000_000],np.int64)
        idx=np.array([0,1,2,3,5],np.int64)
        c=k.analyze(packets(5),ns,idx)
        np.testing.assert_array_equal(c.clock_ns,ns)
        np.testing.assert_array_equal(c.source_index,idx)
        self.assertEqual(k.spans(c.input_eligible,c.clock_breaks),[(0,2),(2,3),(3,4),(4,5)])
        np.testing.assert_array_equal(c.clock_breaks,[0,1,2,4])

    def test_no_sorting_or_duplicate_source_index_repair(self):
        for idx in ([0,0],[1,0]):
            with self.assertRaises(ValueError):
                k.analyze(packets(2),np.array([0,10_000_000],np.int64),np.array(idx,np.int64))

    def test_pairs_use_released_start_rule_path_threshold_and_source_offsets(self):
        p=np.zeros((2000,3));p[:,0]=np.arange(2000)*.5
        pairs=k.distance_pairs(p,100,1900)
        self.assertEqual(pairs[0].tolist(),[100,300,100,100])
        self.assertTrue(np.all(pairs[:,0]%100==0))
        self.assertEqual(int(pairs[:,0].max()),1600)
        self.assertTrue(np.all(pairs[:,1]<1900))
        self.assertTrue(np.any(pairs[:,2]==800))

    def test_short_or_still_data_never_substitutes_loss_objective(self):
        self.assertEqual(len(k.distance_pairs(np.zeros((6000,3)),0,6000)),0)
        self.assertEqual(len(k.distance_pairs(np.zeros((50,3)),0,50)),0)

    def test_proof_bound_raw_loader_and_hdf_keep_original_strings_flags_and_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);name=k.SPLITS["train"];data=root/name[:10]/name/"oxts"/"data"
            data.mkdir(parents=True)
            raw=" ".join(str(x) for x in packets(1)[0])+"\n"
            for i in range(2):
                (data/f"{i:010d}.txt").write_text(raw)
            clock=data.parent/"timestamps.txt"
            clock.write_text("2011-10-03 00:00:00.123456789\n2011-10-03 00:00:00.133456790\n")
            proof={str(p.relative_to(root)): {"bytes":p.stat().st_size,"sha256":k.sha256(p)}
                   for p in [*data.glob("*.txt"),clock]}
            c=k.read_record(root,name,proof)
            out=root/"candidate.h5";k.write_candidate(out,c,{"split":"train"})
            with h5py.File(out) as h5:
                self.assertEqual(h5["raw/packet_text"][0].decode(),raw)
                self.assertEqual(int(h5["raw/civil_time_ns"][1])-int(h5["raw/civil_time_ns"][0]),10_000_001)
                self.assertFalse(h5.attrs["formal_native_ready"])
                self.assertNotIn("car_to_imu_translation",h5)
            with self.assertRaises(FileExistsError):k.write_candidate(out,c,{})
            (data/"0000000000.txt").write_text(raw+" ")
            with self.assertRaises(ValueError):k.read_record(root,name,proof)

    def test_whitelist_and_clock_alignment_reject_without_inventing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(ValueError):k.read_record(root,"other_drive",{})
            name=k.SPLITS["test"];data=root/name[:10]/name/"oxts"/"data";data.mkdir(parents=True)
            p=data/"0000000000.txt";p.write_text("0\n")
            clock=data.parent/"timestamps.txt";clock.write_text("a\nb\n")
            proof={str(clock.relative_to(root)): {"bytes":clock.stat().st_size,"sha256":k.sha256(clock)}}
            with self.assertRaises(ValueError):k.read_record(root,name,proof)

    def test_sensor_reference_quality_invalid_and_input_failure_are_explicit(self):
        p=packets(3);p[0,0]=91;p[1,25]=2;p[2,17]=np.nan
        c=k.analyze(p,np.arange(3,dtype=np.int64)*10_000_000,np.arange(3,dtype=np.int64))
        self.assertTrue(c.reasons[0]&4)
        self.assertTrue(c.reasons[1]&32)
        self.assertTrue(c.reasons[2]&2)
        self.assertFalse(c.reference_candidate.any())
        self.assertFalse(c.input_eligible[2])


if __name__ == "__main__":
    unittest.main()
