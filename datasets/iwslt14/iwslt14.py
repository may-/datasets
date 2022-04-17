# coding=utf-8
"""The IWSLT 2014 Evaluation Campaign includes a multilingual TED Talks MT task."""


import datasets


_CITATION = """\
@inproceedings{cettoloEtAl:EAMT2012,
Address = {Trento, Italy},
Author = {Mauro Cettolo and Christian Girardi and Marcello Federico},
Booktitle = {Proceedings of the 16$^{th}$ Conference of the European Association for Machine Translation (EAMT)},
Date = {28-30},
Month = {May},
Pages = {261--268},
Title = {WIT$^3$: Web Inventory of Transcribed and Translated Talks},
Year = {2012}}
"""

_DESCRIPTION = """\
The IWSLT 2014 Evaluation Campaign includes the MT track on TED Talks. In this edition, the official language pairs are five:

  from English to French
  from English to German
  from German to English
  from English to Italian
  from Italian to English

Optional tasks are proposed with English paired in both directions with other twelve languages:

  from/to English to/from Arabic, Spanish, Farsi, Hebrew, Dutch, Polish, Portuguese-Brazil, Romanian, Russian, Slovenian, Turkish and Chinese

Submitted runs on additional pairs will be evaluated as well, in the hope to stimulate the MT community to evaluate systems on common benchmarks and to share achievements on challenging translation tasks.
"""

_URL = "https://drive.google.com/file/d/1GnBarJIbNgEIIDvUyKDtLmv35Qcxg6Ed/view"

_HOMEPAGE = ""

_LANGUAGES = ["ar", "de", "es", "fa", "he", "it", "nl", "pl", "pt-br", "ro", "ru", "sl", "tr", "zh"]
_PAIRS = [(lang, "en") for lang in _LANGUAGES] + [("en", lang) for lang in _LANGUAGES]


class IWSLT14Config(datasets.BuilderConfig):
    """BuilderConfig for IWSLT14 Dataset"""

    def __init__(self, language_pair=(None, None), **kwargs):
        """

        Args:
            language_pair: the language pair to consider. Should
            contain 2-letter coded strings. For example: ("ja", "en").
            **kwargs: keyword arguments forwarded to super.
        """
        super(IWSLT14Config, self).__init__(
            name="%s-%s" % (language_pair[0], language_pair[1]),
            description="IWSLT 2014 multilingual dataset.",
            version=datasets.Version("1.0.0", ""),
            **kwargs,
        )

        # Validate language pair.
        assert language_pair in _PAIRS

        self.language_pair = language_pair
        

class IWSLT14(datasets.GeneratorBasedBuilder):
    """The IWSLT 2014 Evaluation Campaign includes a multilingual TED Talks MT task."""

    BUILDER_CONFIGS = [IWSLT14Config(language_pair=pair) for pair in _PAIRS]

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {"translation": datasets.features.Translation(languages=self.config.language_pair)}
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://wit3.fbk.eu/2014-01",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        
        def _get_drive_url(url):
            return f"https://drive.google.com/uc?id={url.split('/')[5]}"
        
        source, target = self.config.language_pair
        pair = f"{source}-{target}"
        ex_dir = dl_manager.download_and_extract(_get_drive_url(_URL))
        dl_dir = dl_manager.extract(f"{ex_dir}/2014-01/texts/{source}/{target}/{pair}.tgz")
        path_tmpl = f"{dl_dir}/{pair}/IWSLT14.%s.{pair}.%s.xml"
        
        subsets = {"dev": ['TED.dev2010', 'TEDX.dev2012'],
                   "test": ['TED.tst2010', 'TED.tst2011', 'TED.tst2012']}
        files = {
            "train": {
                "source_files": [f"{dl_dir}/{pair}/train.tags.{pair}.{source}"],
                "target_files": [f"{dl_dir}/{pair}/train.tags.{pair}.{target}"],
                "split": "train",
            },
            "dev": {
                "source_files": [path_tmpl % (year, source) for year in subsets["dev"]],
                "target_files": [path_tmpl % (year, target) for year in subsets["dev"]],
                "split": "dev",
            },
            "test": {
                "source_files": [path_tmpl % (year, source) for year in subsets["test"]],
                "target_files": [path_tmpl % (year, target) for year in subsets["test"]],
                "split": "test",
            },
        }
        
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs=files["train"]),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs=files["dev"]),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs=files["test"]),
        ]

    def _generate_examples(self, source_files, target_files, split):
        """Yields examples."""
        id_ = 0
        source, target = self.config.language_pair
        for source_file, target_file in zip(source_files, target_files):
            with open(source_file, "r", encoding="utf-8") as sf:
                with open(target_file, "r", encoding="utf-8") as tf:
                    for source_row, target_row in zip(sf, tf):
                        source_row = source_row.strip()
                        target_row = target_row.strip()

                        if source_row.startswith("<"):
                            if source_row.startswith("<seg"):
                                # Remove <seg id="1">.....</seg>
                                # Very simple code instead of regex or xml parsing
                                part1 = source_row.split(">")[1]
                                source_row = part1.split("<")[0]
                                part1 = target_row.split(">")[1]
                                target_row = part1.split("<")[0]

                                source_row = source_row.strip()
                                target_row = target_row.strip()
                            else:
                                continue

                        yield id_, {"translation": {source: source_row, target: target_row}}
                        id_ += 1
