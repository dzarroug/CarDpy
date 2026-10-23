"""Numerical and saved-run checks for optional uncertainty."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from cardpy.config import DEFAULT_CONFIG
from cardpy.Data_Saving import Save_Diffusion_Image_Data
from cardpy.uncertainty import run_dataset
from cardpy.uncertainty.bootstrap import make_plan
from cardpy.uncertainty.metrics import angular_uncertainty, invariant_uncertainty
from cardpy.uncertainty.plotting import plot_maps


class UncertaintyTests(unittest.TestCase):
    def dataset(self, root, *, complex_signal=False, rejection=True):
        config = deepcopy(DEFAULT_CONFIG)
        for key in ('gibbs', 'respiratory', 'registration', 'adc_filter',
                    'denoising', 'interpolation', 'extended_matrix', 'e1'):
            config[key]['enabled'] = False
        config['rejection']['enabled'] = rejection
        config['dti']['tensor_fit'] = 'WLS'
        config['uncertainty'].update(enabled=True, iterations=4, n_jobs=1)
        config['operation_type'] = 'Complex' if complex_signal else 'Magnitude'
        vectors = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                            [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=float)
        vectors[4:] /= np.sqrt(2)
        bvals = np.array([0] + [400] * 6)
        signal = 1000 * np.exp(-bvals * np.einsum('ij,j,ij->i', vectors,
                                                [.0017, .0007, .0004], vectors))
        rng = np.random.default_rng(4)
        data = np.broadcast_to(signal[None, None, None, :, None], (8, 8, 2, 7, 3)).copy()
        data += rng.normal(0, 3, data.shape)
        if complex_signal:
            data = data * np.exp(.4j)
        if rejection:
            data[:, :, 1, 0, -1] = np.nan
        header = {axis + ' Resolution': 2.0 for axis in ('X', 'Y', 'Z')}
        stage = 'Rejected' if rejection else 'Original'
        Save_Diffusion_Image_Data(str(root / ('02_' + stage)), stage, header, data, bvals, vectors)
        return config, data, bvals, vectors

    def test_historical_statistics_and_axial_sign(self):
        stats = invariant_uncertainty(np.arange(400.))
        self.assertEqual(stats['lower'], 9)
        self.assertEqual(stats['upper'], 389)
        self.assertEqual(stats['width'], 380)
        vectors = np.array([[1., -1., 1.], [0., 0., 0.], [0., 0., 0.]])
        stats = angular_uncertainty(vectors)
        assert_allclose(stats['cone'], 0, atol=1e-6)
        self.assertEqual(stats['count'], 3)
        self.assertTrue(np.isnan(angular_uncertainty(np.zeros((3, 3)))['cone']))

    def test_plan_keeps_whole_images_and_unique_rows(self):
        from cardpy.Data_Sorting import sorted2stacked
        with tempfile.TemporaryDirectory() as folder:
            _, data, bvals, vectors = self.dataset(Path(folder))
            data, bvals, vectors = sorted2stacked(data, bvals, vectors)
            plan, counts = make_plan(data, bvals, vectors, iterations=4, seed=2)
            self.assertEqual(counts[1, 0], 2)
            self.assertFalse(np.any(plan[:, 1, 0, :] == 14))
            for slc in range(2):
                rows = plan[:, slc].transpose(0, 2, 1).reshape(-1, 7)
                self.assertEqual(len(np.unique(rows, axis=0)), 12)
            assert_allclose(plan, make_plan(data, bvals, vectors, iterations=4, seed=2)[0])
            with self.assertRaisesRegex(ValueError, 'unique composite rows'):
                make_plan(data, bvals, vectors, iterations=1000)

    def test_saved_config_run_resume_and_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, *_ = self.dataset(root)
            (root / 'processing_settings.json').write_text(json.dumps(config))
            before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            result = run_dataset(root)
            self.assertEqual(result, root.resolve() / '14_Uncertainty' / 'uncertainty.npz')
            with np.load(result) as maps:
                self.assertEqual(maps['dMD'].shape, (8, 8, 2))
                self.assertTrue(np.isfinite(maps['dE1']).all())
                assert_allclose(maps['mMD'], (.0017 + .0007 + .0004) / 3 * 1000, atol=.08)
                self.assertTrue(np.all(maps['dMD'] >= 0))
            for name in ('Invariant_Uncertainties.mat', 'Invariant_Medians.mat', 'Angular_Uncertainties.mat'):
                self.assertTrue((result.parent / name).is_file())
            sample = result.parent / 'samples' / '0000.npy'
            timestamp = sample.stat().st_mtime_ns
            run_dataset(root)
            self.assertEqual(timestamp, sample.stat().st_mtime_ns)
            with self.assertRaisesRegex(ValueError, 'manifest differs'):
                run_dataset(root, seed=10)
            self.assertEqual(timestamp, sample.stat().st_mtime_ns)
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)
            fig = plot_maps(result)
            self.assertEqual(len(fig.axes), 15)
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_legacy_complex_config_and_parallel_workers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, *_ = self.dataset(root, complex_signal=True, rejection=False)
            with self.assertRaisesRegex(ValueError, 'original processing config'):
                run_dataset(root)
            result = run_dataset(root, config=config, iterations=2, n_jobs=2, keep_images=True)
            with np.load(result) as maps:
                self.assertTrue(np.isfinite(maps['mMD']).all())
            self.assertEqual(len(list((result.parent / 'images').glob('*.npz'))), 2)
            self.assertEqual(config['uncertainty']['enabled'], True)

    def test_pipeline_config_runs_after_quantitative_results(self):
        from cardpy.Data_Sorting import sorted2stacked
        from cardpy.pipeline import process
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, data, bvals, vectors = self.dataset(root, rejection=False)
            data = np.repeat(np.repeat(data, 4, axis=0), 4, axis=1)
            config['output']['directory'] = str(root / 'processed')
            config['uncertainty']['iterations'] = 2
            theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
            contours = ((15.5 + 5 * np.cos(theta)).tolist(),
                        (15.5 + 5 * np.sin(theta)).tolist(),
                        (15.5 + 12 * np.cos(theta)).tolist(),
                        (15.5 + 12 * np.sin(theta)).tolist(),
                        [27.5, 15.5], [15.5, 27.5])
            header = {axis + ' Resolution': 2.0 for axis in ('X', 'Y', 'Z')}
            with patch('cardpy.pipeline.New_GUI', return_value=contours), \
                 patch('cardpy.pipeline._save_quantitative_maps'), \
                 patch('matplotlib.pyplot.show'):
                results = process(root, config, input_data=(*sorted2stacked(data, bvals, vectors), header))
            self.assertTrue(results['uncertainty'].is_file())
            self.assertTrue((root / 'processed/12_Quantitative_Results/Standard_DTI_Metrics.mat').is_file())
            self.assertTrue((root / 'processed/processing_settings.json').is_file())


if __name__ == '__main__':
    unittest.main()
