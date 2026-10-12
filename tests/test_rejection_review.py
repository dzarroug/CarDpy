"""Headless regression checks for review decisions and image replacement."""
import unittest
import warnings
from unittest.mock import Mock, patch

import numpy as np

from cardpy.Data_Processing.Rejection import _apply_dictionary_keep, _apply_sorted_keep, shot_rejection
from cardpy.Data_Sorting import sorted2dictionary
from cardpy.GUI_Tools.Rejection_Review import RejectionReviewWindow, review_rejection


class ReplacementTests(unittest.TestCase):
    def setUp(self):
        # Same vector at different b-values must remain separate encodings.
        self.matrix = np.arange(2 * 2 * 2 * 3 * 2).reshape(2, 2, 2, 3, 2).astype(complex)
        self.bvals = np.array([0, 100, 200])
        self.bvecs = np.tile([1, 0, 0], (3, 1))

    def test_sorted_decisions_follow_repetition_major_order(self):
        keep = np.ones((2, 6), dtype=int)
        keep[0, 1] = 0
        keep[1, 3] = 0
        before = self.matrix.copy()
        accepted, stats, decisions = _apply_sorted_keep(self.matrix, keep)
        np.testing.assert_array_equal(accepted[:, :, 0, 1, 0], before[:, :, 0, 1, 1])
        np.testing.assert_array_equal(accepted[:, :, 1, 0, 1], before[:, :, 1, 0, 0])
        expected = before.copy()
        expected[:, :, 0, 1, 0] = before[:, :, 0, 1, 1]
        expected[:, :, 1, 0, 1] = before[:, :, 1, 0, 0]
        np.testing.assert_array_equal(accepted, expected)
        np.testing.assert_array_equal(self.matrix, before)
        np.testing.assert_array_equal(decisions.reshape(2, -1), [[1, 1, 0, 1, 1, 1], [1, 0, 1, 1, 1, 1]])
        np.testing.assert_array_equal(stats, [[100, 50, 100], [50, 100, 100]])
        self.assertEqual(accepted.dtype, before.dtype)

    def test_all_rejected_and_singleton_retain_first_source(self):
        accepted, stats, _ = _apply_sorted_keep(self.matrix, np.zeros((2, 6)))
        np.testing.assert_array_equal(accepted[..., 1], self.matrix[..., 0])
        np.testing.assert_array_equal(stats, 0)
        singleton = self.matrix[..., :1]
        accepted, stats, _ = _apply_sorted_keep(singleton, np.zeros((2, 3)))
        np.testing.assert_array_equal(accepted, singleton)
        np.testing.assert_array_equal(stats, 0)

    def test_dictionary_unequal_repetitions(self):
        items = [('first', {'images': self.matrix[:, :, :, 0, :], 'bval': 0, 'bvec': self.bvecs[0]}, 2),
                 ('second', {'images': self.matrix[:, :, :, 1, :1], 'bval': 100, 'bvec': self.bvecs[1]}, 1)]
        before = [item[1]['images'].copy() for item in items]
        keep = np.array([[0, 1, 0], [1, 0, 1]])
        accepted, stats, decisions = _apply_dictionary_keep(items, keep)
        np.testing.assert_array_equal(accepted['first']['images'][:, :, 0, 0], before[0][:, :, 0, 1])
        np.testing.assert_array_equal(accepted['first']['images'][:, :, 1, 1], before[0][:, :, 1, 0])
        np.testing.assert_array_equal(accepted['second']['images'], before[1])
        np.testing.assert_array_equal(stats['first'], [50, 50])
        np.testing.assert_array_equal(stats['second'], [0, 100])
        np.testing.assert_array_equal(decisions['second'], keep[:, 2:])
        for item, original in zip(items, before):
            np.testing.assert_array_equal(item[1]['images'], original)

    def test_invalid_decisions_raise(self):
        for keep in (np.ones((2, 5)), np.full((2, 6), 2), np.full((2, 6), np.nan)):
            with self.assertRaises(ValueError):
                _apply_sorted_keep(self.matrix, keep)
        with self.assertRaises(ValueError):
            _apply_dictionary_keep([], np.ones((2, 1)))


class ReviewTests(unittest.TestCase):
    def make_model(self):
        app = RejectionReviewWindow.__new__(RejectionReviewWindow)
        app.automatic = np.array([[1, 0], [0, 1]])
        app.keep = app.automatic.copy()
        app.slc = 0
        app.tiles = [Mock(), Mock()]
        app.update_tile = Mock()
        app.update_status = Mock()
        app.window = Mock()
        app.result = None
        return app

    def test_toggle_and_reset_preserve_other_slices_and_automatic(self):
        app = self.make_model()
        app.toggle(1)
        np.testing.assert_array_equal(app.keep, [[1, 1], [0, 1]])
        np.testing.assert_array_equal(app.automatic, [[1, 0], [0, 1]])
        app.reset_slice()
        np.testing.assert_array_equal(app.keep, app.automatic)

    def test_apply_returns_copy_and_cancel_aborts(self):
        app = self.make_model()
        app.apply()
        self.assertFalse(np.shares_memory(app.result, app.keep))
        app.window.destroy.assert_called_once()
        with patch('cardpy.GUI_Tools.Rejection_Review.RejectionReviewWindow', return_value=app), patch('cardpy.GUI_Tools.Rejection_Review.wait_window'):
            np.testing.assert_array_equal(review_rejection(None, None, None, None), app.keep)
            app.cancel()
            with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                review_rejection(None, None, None, None)

    def test_invalid_metadata_does_not_create_window(self):
        with patch('cardpy.GUI_Tools.Rejection_Review.create_window') as create:
            with self.assertRaises(ValueError):
                RejectionReviewWindow(np.zeros((8, 8, 1, 2)), [0], [[0, 0, 0]], [[1, 0]])
            create.assert_not_called()

    def test_labels_distinguish_shells_and_repetitions(self):
        app = self.make_model()
        app.bvals = np.array([0, 100, 0, 100])
        app.bvecs = np.tile([1, 0, 0], (4, 1))
        labels = app.acquisition_labels()
        self.assertIn('Encoding 1 · Rep 2', labels[2])
        self.assertIn('Encoding 2 · Rep 2', labels[3])


class IntegrationTests(unittest.TestCase):
    def test_review_changes_replacement_and_statistics_in_both_layouts(self):
        matrix = np.random.default_rng(17).uniform(0.9, 1.1, (8, 8, 1, 2, 2))
        original = matrix.copy()
        bvals = np.array([0., 100.])
        bvecs = np.tile([1., 0., 0.], (2, 1))
        dictionary = sorted2dictionary(matrix, bvals, bvecs)
        for input_images in (matrix, dictionary):
            with self.subTest(dictionary=isinstance(input_images, dict)):
                # Sorted stacking is repetition-major; dictionary stacking is encoding-major.
                reviewed = np.array([[1, 0, 0, 1]])
                with warnings.catch_warnings(), patch(
                        'cardpy.GUI_Tools.Rejection_Review.review_rejection', return_value=reviewed) as review:
                    warnings.simplefilter('ignore')
                    result = shot_rejection(input_images, bvals, bvecs, zoom='OFF',
                                            manual_review=True, save_diagnostics=False,
                                            NRMSE_threshold=0, NSSIM_threshold=0)
                review.assert_called_once()
                np.testing.assert_array_equal(review.call_args.args[3], np.ones((1, 4)))
                if isinstance(input_images, dict):
                    keys = list(dictionary)
                    np.testing.assert_array_equal(result[0][keys[0]]['images'][..., 1], original[:, :, :, 0, 0])
                    np.testing.assert_array_equal(result[0][keys[1]]['images'][..., 0], original[:, :, :, 1, 1])
                    for stats in result[4].values():
                        np.testing.assert_array_equal(stats, [50])
                else:
                    np.testing.assert_array_equal(result[0][:, :, :, 0, 1], original[:, :, :, 0, 0])
                    np.testing.assert_array_equal(result[0][:, :, :, 1, 0], original[:, :, :, 1, 1])
                    np.testing.assert_array_equal(result[4], [[50, 50]])
                    np.testing.assert_array_equal(result[5], [[[1, 0], [0, 1]]])
                np.testing.assert_array_equal(matrix, original)


if __name__ == '__main__':
    unittest.main()
