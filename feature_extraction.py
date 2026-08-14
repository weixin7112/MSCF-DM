import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ALLOWED_EXTENSIONS = {".txt", ".fasta", ".fa", ".fas", ".fna"}
DNA_PATTERN = re.compile(r"^[ACGTUNacgtun]+$")
BASE_TO_ID = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
CANONICAL_BASES = {"A": 0, "C": 1, "G": 2, "T": 3}

SPECIES_CATALOG = [
    {
        "canonical": "A. thaliana",
        "scientific_name": "Arabidopsis thaliana",
        "aliases": ["a_thaliana", "athaliana", "arabidopsis_thaliana"],
    },
    {
        "canonical": "C. elegans",
        "scientific_name": "Caenorhabditis elegans",
        "aliases": ["c_elegans", "celegans", "caenorhabditis_elegans"],
    },
    {
        "canonical": "C. equisetifolia",
        "scientific_name": "Casuarina equisetifolia",
        "aliases": ["c_equisetifolia", "cequisetifolia", "casuarina_equisetifolia"],
    },
    {
        "canonical": "D. melanogaster",
        "scientific_name": "Drosophila melanogaster",
        "aliases": ["d_melanogaster", "dmelanogaster", "drosophila_melanogaster"],
    },
    {
        "canonical": "F. vesca",
        "scientific_name": "Fragaria vesca",
        "aliases": ["f_vesca", "fvesca", "fragaria_vesca"],
    },
    {
        "canonical": "H. sapiens",
        "scientific_name": "Homo sapiens",
        "aliases": ["h_sapiens", "hsapiens", "homo_sapiens", "human"],
    },
    {
        "canonical": "R. chinensis",
        "scientific_name": "Rosa chinensis",
        "aliases": ["r_chinensis", "rchinensis", "rosa_chinensis"],
    },
    {
        "canonical": "S. cerevisiae",
        "scientific_name": "Saccharomyces cerevisiae",
        "aliases": ["s_cerevisiae", "scerevisiae", "saccharomyces_cerevisiae"],
    },
    {
        "canonical": "Ts. SUP5-1",
        "scientific_name": "Tolypocladium sp. SUP5-1",
        "aliases": [
            "ts_sup5_1",
            "tssup5_1",
            "tolypocladium_sp_sup5_1",
            "tolypocladium_sup5_1",
            "tolypocladium",
        ],
    },
    {
        "canonical": "T. thermophila",
        "scientific_name": "Tetrahymena thermophila",
        "aliases": [
            "t_thermophila",
            "tthermophila",
            "tetrahymena_thermophila",
            "t_thermophile",
            "tetrahymena_thermophile",
        ],
    },
    {
        "canonical": "Xoc. BLS256",
        "scientific_name": "Xanthomonas oryzae pv. oryzicola BLS256",
        "aliases": [
            "xoc_bls256",
            "xocbls256",
            "xanthomonas_oryzae_pv_oryzicola_bls256",
            "xanthomonas_oryzae_bls256",
            "bls256",
        ],
    },
    {
        "canonical": "M. musculus",
        "scientific_name": "Mus musculus",
        "aliases": ["m_musculus", "mmusculus", "mus_musculus", "mouse"],
    },
]

SPECIES_TO_ID = {item["canonical"]: index + 1 for index, item in enumerate(SPECIES_CATALOG)}
ID_TO_SPECIES = {0: "UNKNOWN", **{value: key for key, value in SPECIES_TO_ID.items()}}


def _normalize_alias(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", text.lower())
    return re.sub(r"_+", "_", normalized).strip("_")


SPECIES_ALIAS_TO_CANONICAL = {}
for species in SPECIES_CATALOG:
    aliases = set(species["aliases"] + [species["canonical"], species["scientific_name"]])
    for alias in aliases:
        normalized_alias = _normalize_alias(alias)
        if normalized_alias:
            SPECIES_ALIAS_TO_CANONICAL[normalized_alias] = species["canonical"]


def collect_sequence_files(root: Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    )


def _normalize_path_text(path: Path, root: Optional[Path] = None) -> str:
    try:
        relative_path = path.relative_to(root) if root is not None else path
    except ValueError:
        relative_path = path
    text = "_".join(relative_path.parts).lower()
    text = re.sub(r"\s*\(\d+\)(?=\.)", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def infer_binary_label(path: Path, root: Optional[Path] = None) -> Optional[int]:
    text = _normalize_path_text(path, root)
    positive_pattern = r"(^|_)(pos(?:itive)?(?:_?samples?)?\d*)(_|$)"
    negative_pattern = r"(^|_)(neg(?:ative)?(?:_?samples?)?\d*)(_|$)"
    is_positive = re.search(positive_pattern, text) is not None
    is_negative = re.search(negative_pattern, text) is not None
    if is_positive == is_negative:
        return None
    return 1 if is_positive else 0


def infer_species(path: Path, root: Optional[Path] = None) -> tuple[str, int]:
    text = _normalize_path_text(path, root)
    matches = []
    for alias, canonical in sorted(
        SPECIES_ALIAS_TO_CANONICAL.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if re.search(rf"(^|_){re.escape(alias)}(_|$)", text):
            matches.append(canonical)
    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        canonical = unique_matches[0]
        return canonical, SPECIES_TO_ID[canonical]
    if len(unique_matches) > 1:
        raise ValueError(f"Ambiguous species name in path: {path}")
    return "UNKNOWN", 0


def sanitize_sequence(raw_sequence: str) -> Optional[str]:
    sequence = re.sub(r"\s+", "", raw_sequence).upper().replace("U", "T")
    if not sequence or DNA_PATTERN.fullmatch(sequence) is None:
        return None
    return sequence


def read_sequence_records(path: Path) -> tuple[list[dict], int, str]:
    text = Path(path).read_text(encoding="utf-8-sig", errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return [], 0, "EMPTY"

    is_fasta = any(line.startswith(">") for line in lines)
    records: list[dict] = []
    invalid_count = 0

    if is_fasta:
        header: Optional[str] = None
        sequence_lines: list[str] = []
        record_number = 0

        def flush_record() -> None:
            nonlocal header, sequence_lines, invalid_count, record_number
            if not sequence_lines:
                return
            record_number += 1
            sequence = sanitize_sequence("".join(sequence_lines))
            if sequence is None:
                invalid_count += 1
            else:
                records.append(
                    {
                        "record_id": header or f"sample_{record_number}",
                        "sequence": sequence,
                        "source_file": path.name,
                    }
                )
            sequence_lines = []

        for line in lines:
            if line.startswith(">"):
                flush_record()
                header = line[1:].strip() or None
            else:
                sequence_lines.append(line)
        flush_record()
        content_type = "FASTA"
    else:
        for line_number, line in enumerate(lines, start=1):
            sequence = sanitize_sequence(line)
            if sequence is None:
                invalid_count += 1
            else:
                records.append(
                    {
                        "record_id": f"line_{line_number}",
                        "sequence": sequence,
                        "source_file": path.name,
                    }
                )
        content_type = "TXT"

    return records, invalid_count, content_type


def center_pad_crop(sequence: str, target_length: int) -> str:
    sequence_length = len(sequence)
    if sequence_length == target_length:
        return sequence
    if sequence_length > target_length:
        start = (sequence_length - target_length) // 2
        return sequence[start : start + target_length]
    total_padding = target_length - sequence_length
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding
    return "N" * left_padding + sequence + "N" * right_padding


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def kmer_to_id(kmer: str) -> int:
    value = 0
    for nucleotide in kmer:
        if nucleotide not in CANONICAL_BASES:
            return 0
        value = value * 4 + CANONICAL_BASES[nucleotide]
    return value + 1


def encode_one_sequence(
    sequence: str, target_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    token_ids = np.array([BASE_TO_ID.get(base, 5) for base in sequence], dtype=np.int64)
    k2_ids = np.zeros(target_length, dtype=np.int64)
    k3_ids = np.zeros(target_length, dtype=np.int64)
    for index in range(target_length):
        if index >= 1:
            k2_ids[index] = kmer_to_id(sequence[index - 1 : index + 1])
        if index >= 1 and index + 1 < target_length:
            k3_ids[index] = kmer_to_id(sequence[index - 1 : index + 2])
    return token_ids, k2_ids, k3_ids


def encode_sequence_list(
    sequences: list[str], target_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_count = len(sequences)
    tokens = np.empty((sample_count, target_length), dtype=np.int64)
    k2_ids = np.empty((sample_count, target_length), dtype=np.int64)
    k3_ids = np.empty((sample_count, target_length), dtype=np.int64)
    for index, sequence in enumerate(sequences):
        tokens[index], k2_ids[index], k3_ids[index] = encode_one_sequence(
            sequence, target_length
        )
    return tokens, k2_ids, k3_ids


class DNADataset(Dataset):
    def __init__(
        self,
        tokens: np.ndarray,
        k2_ids: np.ndarray,
        k3_ids: np.ndarray,
        species_ids: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        self.tokens = torch.as_tensor(tokens, dtype=torch.long)
        self.k2_ids = torch.as_tensor(k2_ids, dtype=torch.long)
        self.k3_ids = torch.as_tensor(k3_ids, dtype=torch.long)
        self.species_ids = torch.as_tensor(species_ids, dtype=torch.long)
        self.labels = (
            None if labels is None else torch.as_tensor(labels, dtype=torch.float32)
        )

        sample_count = len(self.tokens)
        if not (
            len(self.k2_ids)
            == len(self.k3_ids)
            == len(self.species_ids)
            == sample_count
        ):
            raise ValueError("Encoded arrays have inconsistent sample counts.")
        if self.labels is not None and len(self.labels) != sample_count:
            raise ValueError("Labels and encoded sequences have inconsistent sample counts.")

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        item = {
            "tokens": self.tokens[index],
            "k2": self.k2_ids[index],
            "k3": self.k3_ids[index],
            "species_id": self.species_ids[index],
            "index": index,
        }
        if self.labels is not None:
            item["label"] = self.labels[index]
        return item


def load_records(
    data_dir: Path,
    labels_required: bool,
    require_known_species: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    data_dir = Path(data_dir)
    sequence_files = collect_sequence_files(data_dir)
    if not sequence_files:
        raise FileNotFoundError(f"No TXT or FASTA files were found in {data_dir}.")

    records: list[dict] = []
    report_rows: list[dict] = []
    labeled_file_count = 0
    unlabeled_file_count = 0

    for path in sequence_files:
        label = infer_binary_label(path, data_dir)
        if label is None:
            unlabeled_file_count += 1
            if labels_required:
                raise ValueError(
                    f"A training label could not be inferred from '{path}'. "
                    "Include 'pos' or 'neg' in the file name or parent directory."
                )
        else:
            labeled_file_count += 1

        species_name, species_id = infer_species(path, data_dir)
        if require_known_species and species_id == 0:
            raise ValueError(
                f"Species could not be inferred from '{path}'. "
                "Rename the file using a supported species name."
            )

        file_records, invalid_count, content_type = read_sequence_records(path)
        relative_path = str(path.relative_to(data_dir))
        for record in file_records:
            record.update(
                {
                    "label": label,
                    "species": species_name,
                    "species_id": int(species_id),
                    "source_file": relative_path,
                }
            )
        records.extend(file_records)
        report_rows.append(
            {
                "file": relative_path,
                "species": species_name,
                "species_id": species_id,
                "label": (
                    "positive" if label == 1 else "negative" if label == 0 else "unlabeled"
                ),
                "format": content_type,
                "valid_sequences": len(file_records),
                "invalid_sequences": invalid_count,
                "lengths": str(
                    Counter(len(record["sequence"]) for record in file_records).most_common(5)
                ),
            }
        )

    if labeled_file_count and unlabeled_file_count:
        raise ValueError(
            "Labeled and unlabeled sequence files cannot be mixed in one evaluation directory."
        )

    dataframe = pd.DataFrame(records)
    if dataframe.empty:
        raise RuntimeError("No valid sequences were loaded.")

    has_labels = unlabeled_file_count == 0
    if labels_required and set(dataframe["label"].dropna().astype(int).unique()) != {0, 1}:
        raise RuntimeError("Both positive and negative training samples are required.")

    return dataframe, pd.DataFrame(report_rows), has_labels


def _prepare_dataset(
    data_dir: Path,
    target_length: Optional[int],
    labels_required: bool,
    deduplicate: bool,
    seed: int,
    require_known_species: bool,
    shuffle: bool,
    remove_label_conflicts: bool,
) -> tuple[pd.DataFrame, DNADataset, int, pd.DataFrame, bool]:
    dataframe, file_report, has_labels = load_records(
        data_dir=data_dir,
        labels_required=labels_required,
        require_known_species=require_known_species,
    )

    if target_length is None:
        length_counts = Counter(dataframe["sequence"].str.len().tolist())
        target_length = int(length_counts.most_common(1)[0][0])
    if target_length < 9:
        raise ValueError(f"Target sequence length is unexpectedly small: {target_length}.")

    dataframe["original_length"] = dataframe["sequence"].str.len()
    dataframe["sequence"] = dataframe["sequence"].map(
        lambda sequence: center_pad_crop(sequence, target_length)
    )

    if has_labels:
        dataframe["label"] = dataframe["label"].astype(int)
        if remove_label_conflicts:
            conflicting = dataframe.groupby(["species_id", "sequence"])["label"].nunique()
            conflict_keys = set(conflicting[conflicting > 1].index.tolist())
            if conflict_keys:
                keep_mask = [
                    (int(species_id), sequence) not in conflict_keys
                    for species_id, sequence in zip(
                        dataframe["species_id"], dataframe["sequence"]
                    )
                ]
                dataframe = dataframe.loc[keep_mask].copy()

    if deduplicate:
        subset = ["species_id", "sequence"]
        if has_labels:
            subset.append("label")
        dataframe = dataframe.drop_duplicates(subset=subset, keep="first").copy()

    if shuffle:
        dataframe = dataframe.sample(frac=1.0, random_state=seed)
    dataframe = dataframe.reset_index(drop=True)
    species_ids = dataframe["species_id"].to_numpy(dtype=np.int64)
    labels = (
        dataframe["label"].to_numpy(dtype=np.int64) if has_labels else None
    )
    tokens, k2_ids, k3_ids = encode_sequence_list(
        dataframe["sequence"].tolist(), target_length
    )
    dataset = DNADataset(tokens, k2_ids, k3_ids, species_ids, labels)
    dataframe["sequence_hash"] = dataframe["sequence"].map(sequence_sha256)
    return dataframe, dataset, target_length, file_report, has_labels


def prepare_training_data(
    data_dir: Path,
    target_length: Optional[int] = None,
    deduplicate: bool = True,
    seed: int = 42,
    require_known_species: bool = True,
) -> tuple[pd.DataFrame, DNADataset, int, pd.DataFrame]:
    dataframe, dataset, sequence_length, file_report, _ = _prepare_dataset(
        data_dir=data_dir,
        target_length=target_length,
        labels_required=True,
        deduplicate=deduplicate,
        seed=seed,
        require_known_species=require_known_species,
        shuffle=True,
        remove_label_conflicts=True,
    )
    return dataframe, dataset, sequence_length, file_report


def prepare_test_data(
    data_dir: Path,
    target_length: int,
    deduplicate: bool = False,
    seed: int = 42,
    require_known_species: bool = True,
) -> tuple[pd.DataFrame, DNADataset, pd.DataFrame, bool]:
    dataframe, dataset, _, file_report, has_labels = _prepare_dataset(
        data_dir=data_dir,
        target_length=target_length,
        labels_required=False,
        deduplicate=deduplicate,
        seed=seed,
        require_known_species=require_known_species,
        shuffle=False,
        remove_label_conflicts=False,
    )
    return dataframe, dataset, file_report, has_labels
