# coding=utf-8
# KFTT Dataset

# Lint as: python3
"""The Kyoto Free Translation Task (KFTT) Dataset for Japanese-English machine translation."""


import collections

import datasets


_DESCRIPTION = """\
The Kyoto Free Translation Task is a task for Japanese-English translation that focuses
on Wikipedia articles related to Kyoto. The data used was originally prepared by the
National Institute for Information and Communication Technology (NICT) and released as
the Japanese-English Bilingual Corpus of Wikipedia's Kyoto Articles (we are simply using
the data, NICT does not specifically endorse or sponsor this task).
"""

_CITATION = """\
@misc{neubig11kftt,
    author = {Graham Neubig},
    title = {The {Kyoto} Free Translation Task},
    howpublished = {http://www.phontron.com/kftt},
    year = {2011}
}
"""

_HOMEPAGE = "http://www.phontron.com/kftt/"

_LICENSE = "Creative Commons Attribution-Share-Alike License 3.0 (CC BY-SA 3.0)"

_DATA_URL = "http://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz"

# Tuple that describes a single pair of files with matching translations.
# language_to_file is the map from language (2 letter string: example 'en')
# to the file path in the extracted directory.
TranslateData = collections.namedtuple("TranslateData", ["url", "language_to_file"])


class KFTTConfig(datasets.BuilderConfig):
    """BuilderConfig for KFTT."""

    def __init__(self, language_pair=(None, None), **kwargs):
        """BuilderConfig for KFTT.

        Args:
            for the `datasets.features.text.TextEncoder` used for the features feature.
          language_pair: pair of languages that will be used for translation. Should
            contain 2-letter coded strings. First will be used at source and second
            as target in supervised mode. For example: ("ja", "en").
          **kwargs: keyword arguments forwarded to super.
        """
        super(KFTTConfig, self).__init__(
            name="%s-%s" % (language_pair[0], language_pair[1]),
            description="English-Japanese translation dataset.",
            version=datasets.Version("1.0.0", ""),
            **kwargs,
        )

        # Validate language pair.
        assert "en" in language_pair
        assert "ja" in language_pair

        self.language_pair = language_pair


class KFTT(datasets.GeneratorBasedBuilder):
    """KFTT machine translation dataset."""

    BUILDER_CONFIGS = [
        KFTTConfig(
            language_pair=("en", "ja"),
        ),
    ]

    def _info(self):
        source, target = self.config.language_pair
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {"translation": datasets.features.Translation(languages=self.config.language_pair)}
            ),
            supervised_keys=(source, target),
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        archive = dl_manager.download(_DATA_URL)

        source, target = self.config.language_pair
        path_tmpl = "kftt-data-1.0/data/orig/kyoto-{split}.{lang}"

        files = {}
        for split in ("train", "dev", "test", "tune"):
            files[split] = {
                "source_file": path_tmpl.format(split=split, lang=source),
                "target_file": path_tmpl.format(split=split, lang=target),
                "files": dl_manager.iter_archive(archive),
            }

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs=files["train"]),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs=files["dev"]),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs=files["test"]),
            datasets.SplitGenerator(name=datasets.Split("tune"), gen_kwargs=files["tune"]),
        ]

    def _generate_examples(self, files, source_file, target_file):
        """This function returns the examples in the raw (text) form."""
        source_sentences, target_sentences = None, None
        for path, f in files:
            if path == source_file:
                source_sentences = f.read().decode("utf-8").split("\n")
            elif path == target_file:
                target_sentences = f.read().decode("utf-8").split("\n")
            if source_sentences is not None and target_sentences is not None:
                break

        assert len(target_sentences) == len(source_sentences), "Sizes do not match: %d vs %d for %s vs %s." % (
            len(source_sentences),
            len(target_sentences),
            source_file,
            target_file,
        )

        source, target = self.config.language_pair
        for idx, (l1, l2) in enumerate(zip(source_sentences, target_sentences)):
            result = {"translation": {source: l1, target: l2}}
            # Make sure that both translations are non-empty.
            if all(result.values()):
                yield idx, result
