import os
import shutil
from tempfile import TemporaryDirectory
from unittest import TestCase

import datasets.config
from datasets.builder import GeneratorBasedBuilder
from datasets.commands.dummy_data import DummyDataGeneratorDownloadManager, MockDownloadManager
from datasets.features import Features, Value
from datasets.info import DatasetInfo
from datasets.splits import Split, SplitGenerator
from datasets.utils.download_manager import DownloadConfig
from datasets.utils.version import Version


EXPECTED_XML_DUMMY_DATA = """\
<tmx version="1.4">
  <header segtype="sentence" srclang="ca" />
  <body>
    <tu>
      <tuv xml:lang="ca"><seg>Contingut 1</seg></tuv>
      <tuv xml:lang="en"><seg>Content 1</seg></tuv>
    </tu>
    </body>
</tmx>"""


class DummyBuilder(GeneratorBasedBuilder):
    def __init__(self, tmp_test_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tmp_test_dir = tmp_test_dir

    def _info(self) -> DatasetInfo:
        return DatasetInfo(features=Features({"text": Value("string")}))

    def _split_generators(self, dl_manager):
        to_dl = {
            "train": os.path.abspath(os.path.join(self.tmp_test_dir, "train.txt")),
            "test": os.path.abspath(os.path.join(self.tmp_test_dir, "test.txt")),
        }
        downloaded_files = dl_manager.download_and_extract(to_dl)
        return [
            SplitGenerator(Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            SplitGenerator(Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath, **kwargs):
        with open(filepath, encoding="utf-8") as f:
            for i, line in enumerate(f):
                yield i, {"text": line.strip()}


class DummyDataAutoGenerationTest(TestCase):
    def test_dummy_data_autogenerate(self):
        n_lines = 5

        with TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "train.txt"), "w", encoding="utf-8") as f:
                f.write("foo\nbar\n" * 10)
            with open(os.path.join(tmp_dir, "test.txt"), "w", encoding="utf-8") as f:
                f.write("foo\nbar\n" * 10)

            class MockDownloadManagerWithCustomDatasetsScriptsDir(MockDownloadManager):
                datasets_scripts_dir = os.path.join(tmp_dir, "datasets")

            cache_dir = os.path.join(tmp_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            dataset_builder = DummyBuilder(tmp_test_dir=tmp_dir, cache_dir=cache_dir)
            mock_dl_manager = MockDownloadManagerWithCustomDatasetsScriptsDir(
                dataset_name=dataset_builder.name,
                config=None,
                version=Version("0.0.0"),
                use_local_dummy_data=True,
                cache_dir=cache_dir,
                load_existing_dummy_data=False,  # dummy data don't exist yet
            )
            download_config = DownloadConfig(cache_dir=os.path.join(tmp_dir, datasets.config.DOWNLOADED_DATASETS_DIR))
            dl_manager = DummyDataGeneratorDownloadManager(
                dataset_name=dataset_builder.name,
                mock_download_manager=mock_dl_manager,
                download_config=download_config,
            )
            dataset_builder.download_and_prepare(dl_manager=dl_manager, try_from_hf_gcs=False)
            shutil.rmtree(dataset_builder._cache_dir)

            dl_manager.auto_generate_dummy_data_folder(n_lines=n_lines)
            path_do_dataset = os.path.join(mock_dl_manager.datasets_scripts_dir, mock_dl_manager.dataset_name)
            dl_manager.compress_autogenerated_dummy_data(path_do_dataset)

            mock_dl_manager.load_existing_dummy_data = True
            dataset_builder.download_and_prepare(
                dl_manager=mock_dl_manager, ignore_verifications=True, try_from_hf_gcs=False
            )
            dataset = dataset_builder.as_dataset(split="train")
            self.assertEqual(len(dataset), n_lines)
            del dataset


def test_create_xml_dummy_data(xml_file, tmp_path):
    dst_path = tmp_path / "file.xml"
    DummyDataGeneratorDownloadManager._create_xml_dummy_data(xml_file, dst_path, "tu", n_lines=1)
    with open(dst_path) as f:
        xml_dummy_data = f.read()
    assert xml_dummy_data == EXPECTED_XML_DUMMY_DATA
