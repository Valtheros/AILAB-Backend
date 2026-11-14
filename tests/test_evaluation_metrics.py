import unittest

import numpy as np

from evaluation_metrics import classification_report, detection_report, semantic_report


class EvaluationMetricsTests(unittest.TestCase):
    def test_classification_reports_undefined_classes_without_inventing_scores(self):
        report = classification_report([[3, 1, 0], [1, 1, 0], [0, 0, 0]], ['cat', 'dog', 'absent'])
        self.assertAlmostEqual(report['accuracy'], 4 / 6)
        self.assertEqual(report['classes'][0]['support'], 4)
        self.assertAlmostEqual(report['classes'][0]['precision'], 3 / 4)
        self.assertIsNone(report['classes'][2]['f1'])

    def test_semantic_iou_and_dice_include_only_present_classes_in_mean(self):
        report = semantic_report([[8, 2, 0], [1, 9, 0], [0, 0, 0]], ['background', 'road', 'absent'])
        self.assertAlmostEqual(report['pixelAccuracy'], .85)
        self.assertAlmostEqual(report['classes'][1]['iou'], .75)
        self.assertAlmostEqual(report['classes'][1]['dice'], 18 / 21)
        self.assertIsNone(report['classes'][2]['iou'])

    def test_coco_ap_and_fixed_threshold_ignore_crowd_and_count_errors(self):
        images = [{'id': 1, 'width': 100, 'height': 100}]
        categories = [{'id': 2, 'name': 'dog'}, {'id': 1, 'name': 'cat'}, {'id': 3, 'name': 'absent'}]
        annotations = [
            {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 10, 10], 'area': 100, 'iscrowd': 0},
            {'id': 2, 'image_id': 1, 'category_id': 2, 'bbox': [20, 20, 10, 10], 'area': 100, 'iscrowd': 0},
            {'id': 3, 'image_id': 1, 'category_id': 1, 'bbox': [40, 40, 10, 10], 'area': 100, 'iscrowd': 1},
        ]
        predictions = [
            {'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 10, 10], 'score': .9},
            {'image_id': 1, 'category_id': 1, 'bbox': [60, 60, 10, 10], 'score': .8},
            {'image_id': 1, 'category_id': 1, 'bbox': [40, 40, 10, 10], 'score': .7},
        ]
        report, images_report = detection_report(images, annotations, categories, predictions)
        classes = {item['name']: item for item in report['classes']}
        self.assertAlmostEqual(classes['cat']['ap50'], 1.0)
        self.assertAlmostEqual(classes['cat']['precision'], .5)
        self.assertEqual(classes['dog']['recall'], 0)
        self.assertIsNone(classes['absent']['ap'])
        self.assertEqual(images_report[1], {'tp': 1, 'fp': 1, 'fn': 1})

    def test_empty_predictions_and_empty_ground_truth_are_distinct(self):
        images = [{'id': 1, 'width': 10, 'height': 10}]
        categories = [{'id': 1, 'name': 'cat'}]
        annotation = {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 4, 4], 'area': 16, 'iscrowd': 0}
        report, _ = detection_report(images, [annotation], categories, [])
        self.assertEqual(report['map50'], 0)
        empty, _ = detection_report(images, [], categories, [])
        self.assertIsNone(empty['map50'])

    def test_invalid_confusion_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            semantic_report(np.zeros((1, 2)), ['cat'])


if __name__ == '__main__':
    unittest.main()
