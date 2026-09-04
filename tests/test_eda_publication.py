"""Проверки переносимости опубликованного анализа данных."""

import hashlib
import json
import unittest
from pathlib import Path

from scripts.eda import gazetteer_analysis, style_analysis_pipeline

ROOT = Path(__file__).resolve().parents[1]


class EDAPublicationTests(unittest.TestCase):
    """Проверить пути, данные и синтаксис без сети и модельных запусков."""

    def test_paths_and_data_hashes(self):
        """Проверить совпадение reference-данных с опубликованным снимком."""
        self.assertEqual(gazetteer_analysis.ROOT, ROOT)
        self.assertEqual(style_analysis_pipeline.ROOT, ROOT)
        summary = json.loads((gazetteer_analysis.OUT / 'summary.json').read_text())
        for split, expected in summary['input_sha256'].items():
            path = style_analysis_pipeline.DATA_DIR / f'{split}.jsonl'
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_notebook_compiles(self):
        """Проверить синтаксис ячеек и отсутствие старых импортов и путей."""
        notebook = json.loads((ROOT / 'data_analysis.ipynb').read_text())
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            self.assertNotIn('brand_analytics_data', source)
            self.assertNotIn('from gazetteer_analysis import', source)
            compile(source, f'cell_{index}', 'exec')


if __name__ == '__main__':
    unittest.main()
