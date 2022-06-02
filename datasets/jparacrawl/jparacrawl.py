# coding=utf-8
# JParaCrawl Dataset

# Lint as: python3
"""JParaCrawl: A Large Scale Web-Based English-Japanese Parallel Corpus."""


import collections

import pandas as pd

import datasets
from datasets.features import Translation, Value


_DESCRIPTION = """\
JParaCrawl is the largest publicly available English-Japanese parallel corpus created by NTT.
It was created by largely crawling the web and automatically aligning parallel sentences.
"""

_CITATION = """\
@inproceedings{morishita-etal-2020-jparacrawl,
    title = "{JP}ara{C}rawl: A Large Scale Web-Based {E}nglish-{J}apanese Parallel Corpus",
    author = "Morishita, Makoto  and
      Suzuki, Jun  and
      Nagata, Masaaki",
    booktitle = "Proceedings of The 12th Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://www.aclweb.org/anthology/2020.lrec-1.443",
    pages = "3603--3609",
    ISBN = "979-10-95546-34-4",
}
"""

_LICENSE = """\
Terms of Use for Bilingual Data, Monolingual Data and Trained Models

Nippon Telegraph and Telephone Corporation (Hereinafter referred to as "our company".) will provide bilingual data, monolingual data and trained models (Hereinafter referred to as "this data.") subject to your acceptance of these Terms of Use. We assume that you have agreed to these Terms of Use when you start using this data (including downloads).

Article 1 (Use conditions)
This data can only be used for research purposes involving information analysis (Including, but not limited to, replication and distribution. Hereinafter the same in this article.). The same applies to the derived data created based on this data. However, this data is not available for commercial use, including the sale of translators trained using this data.

Article 2 (Disclaimer)
Our company does not warrant the quality, performance or any other aspects of this data. We shall not be liable for any direct or indirect damages caused by the use of this data. Our company shall not be liable for any damage to the system caused by the installation of this data.

Article 3 (Other).
This data may be changed in whole or in part, or provision of this data may be interrupted or stopped at our company’s discretion without prior notice.
"""

_DATA_URL = "https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/3.0/bitext/en-%s.tar.gz"

_HOMEPAGE = "https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/"

_LANGUAGE_PAIRS = [("en", lang) for lang in ["ja", "zh"]]

# Tuple that describes a single pair of files with matching translations.
# language_to_file is the map from language (2 letter string: example 'en')
# to the file path in the extracted directory.
TranslateData = collections.namedtuple("TranslateData", ["url", "language_to_file"])


class JParaCrawlConfig(datasets.BuilderConfig):
    """BuilderConfig for JParaCrawl."""

    def __init__(self, language_pair=(None, None), **kwargs):
        """BuilderConfig for JParaCrawl.

        Args:
            for the `datasets.features.text.TextEncoder` used for the features feature.
          language_pair: pair of languages that will be used for translation. Should
            contain 2-letter coded strings. First will be used at source and second
            as target in supervised mode. For example: ("en", "ja").
          **kwargs: keyword arguments forwarded to super.
        """
        name = "%s-%s" % (language_pair[0], language_pair[1])

        description = ("%s-%s Translation dataset") % (language_pair[0], language_pair[1])
        super(JParaCrawlConfig, self).__init__(
            name=name,
            description=description,
            version=datasets.Version("3.0.0", ""),
            **kwargs,
        )

        # Validate language pair.
        assert "en" in language_pair, ("Config language pair must contain `en`, got: %s", language_pair)
        source, target = language_pair
        non_en = source if target == "en" else target
        assert non_en in ["ja", "zh"], ("Invalid non-en language in pair: %s", non_en)

        self.language_pair = language_pair


class JParaCrawl(datasets.GeneratorBasedBuilder):
    """JParaCrawl machine translation dataset."""

    BUILDER_CONFIGS = [
        JParaCrawlConfig(language_pair=("en", "ja")),
        JParaCrawlConfig(language_pair=("en", "zh")),
    ]

    def _info(self):
        source, target = self.config.language_pair
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "url": Value(dtype="string", id=None),
                    "probability": Value(dtype="float32", id=None),
                    "translation": Translation(languages=self.config.language_pair),
                }
            ),
            supervised_keys=(source, target),
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        source, target = self.config.language_pair
        non_en = source if target == "en" else target
        archive = dl_manager.download(_DATA_URL % non_en)

        train_files = {
            "txt_file": f"en-{non_en}/en-{non_en}.bicleaner05.txt",
            "files": dl_manager.iter_archive(archive),
        }

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs=train_files),
        ]

    def _generate_examples(self, files, txt_file):
        """This function returns the examples in the raw form."""
        source, target = self.config.language_pair
        non_en = source if target == "en" else target
        for path, file in files:
            if path == txt_file:
                dtype = collections.OrderedDict({"url": str, "probability": float, "en": str, non_en: str})
                dfs = pd.read_csv(
                    path,
                    header=None,
                    names=dtype.keys(),
                    sep="\t",
                    encoding="utf8",
                    quoting=3,
                    keep_default_na=False,
                    na_values="",
                    dtype=dtype,
                    chunksize=1000000,
                )
                break

        for df in dfs:
            for idx, row in df.iterrows():
                result = {
                    "url": row["url"],
                    "probability": row["probability"],
                    "translation": {"en": row["en"], non_en: row[non_en]},
                }
                # Make sure that both translations are non-empty.
                if all(v is not None for v in result.values()):
                    yield idx, result
