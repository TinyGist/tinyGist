import os
import numpy as np
import librosa
import soundfile as sf
import datasets
import fiftyone as fo
import io
from scipy.fft import dct


ALL_CLASSES = [
    "Yes", "No", "Up", "Down", "Left", "Right", "On",
    "Off", "Stop", "Go", "Zero", "One", "Two", "Three", "Four",
    "Five", "Six", "Seven", "Eight", "Nine", "Bed", "Bird",
    "Cat", "Dog", "Happy", "House", "Marvin", "Sheila", "Tree", "Wow",  # v0.1
    "Backward", "Forward", "Follow", "Learn", "Visual"  # v0.2 addition
]

KEY_WORDS = [
    "Yes", "No", "Up", "Down", "Left", "Right", "On", "Off",
]

SPEECH_COMMANDS_PARQUET_REVISION = "88a61cb409d327babe1177480229de411f5d035e"


class KWSDatasetPreparer:
    def __init__(self, key_words:list[str] | str=KEY_WORDS, dataset_dir="./data"):
        self.all_classes = ALL_CLASSES

        if key_words == "all":
            self.key_words = ALL_CLASSES
        else:
            for keyword in key_words:
                if keyword not in self.all_classes:
                    raise ValueError(f"{keyword} is not in {self.all_classes}")
            self.key_words = key_words

        self.key_words = set([k.strip().lower() for k in self.key_words])
        self.dataset_dir = dataset_dir
        self.num_kws = len(self.key_words)

        self.total_raw_dataset = None
        self.full_raw_dataset = None
        self.fo_train_dataset = None
        self.fo_test_dataset = None
        self.id2label = None

    def load_raw_dataset(self):
        self.total_raw_dataset = datasets.load_dataset(
            "google/speech_commands",
            data_dir="v0.02",
            revision=SPEECH_COMMANDS_PARQUET_REVISION,
            cache_dir=self.dataset_dir
        )
        train_raw_dataset = self.total_raw_dataset["train"]
        val_raw_dataset = self.total_raw_dataset["validation"]
        test_raw_dataset = self.total_raw_dataset["test"]

        self.full_raw_dataset = datasets.concatenate_datasets(
            [train_raw_dataset, val_raw_dataset, test_raw_dataset]
        )
        self.id2label = self.full_raw_dataset.features["label"].names

    @staticmethod
    def _pcen_feature(wav: np.ndarray, sample_rate: int) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=sample_rate,
            n_fft=1024,
            hop_length=253,
            win_length=1024,
            n_mels=64,
            fmin=20,
            fmax=8000,
            power=2.0,
            htk=False,
            center=True,
            norm="slaney",
        )
        pcen = librosa.pcen(mel, sr=sample_rate)
        pcen_db = librosa.power_to_db(pcen, ref=1.0).astype(np.float32)
        return pcen_db

    @staticmethod
    def _dct_2d_trans(raw_data: np.ndarray, first_keep, second_keep) -> np.ndarray:
        mean_value = np.mean(raw_data)
        raw_data = raw_data-mean_value
        trans_data = dct(dct(raw_data, axis=0, norm='ortho', type=2), norm='ortho', axis=1, type=2)
        max_first, max_second = trans_data.shape
        first_keep = min(max_first, first_keep)
        second_keep = min(max_second, second_keep)
        return trans_data[:first_keep, :second_keep]


    def generate_silence_dataset(
        self,
        num_silence: int,
        start_idx: int,
        wave_length: int = 16000,
        sample_rate: int = 16000,
        dataset_dir: str = "./data/kws/processed",
    ):
        print(f"Generating synthetic silence..., {num_silence}")
        os.makedirs(dataset_dir, exist_ok=True)

        feature_path = []
        label_list = []
        idx_list = []

        for idx in range(start_idx, start_idx + num_silence):
            wav = np.random.normal(0.0, 7e-4, size=wave_length).astype(np.float32)
            pcen_db = KWSDatasetPreparer._pcen_feature(wav, sample_rate)
            if len(self.key_words) > 10:
                dct_data = self._dct_2d_trans(pcen_db, 32, 32)
            else:
                dct_data = self._dct_2d_trans(pcen_db, 32, 32)

            out_path = os.path.join(dataset_dir, f"silence_{idx}.npy")
            np.save(out_path, dct_data)

            feature_path.append(out_path)
            label_list.append("_silence_")
            idx_list.append(idx)

        silence_dataset = datasets.Dataset.from_dict({
            "id": idx_list,
            "label_str": label_list,
            "feature_path": feature_path,
        })
        print("Finish generating synthetic silence.")
        return silence_dataset # [40, 45] [num_mel, num_frame]

    def preprocess_dataset(self, sample_rate=16000, wave_length=16000):
        if len(self.key_words) > 10:
            out_dir = os.path.join(self.dataset_dir, "kws-all", "processed")
        else:
            out_dir = os.path.join(self.dataset_dir, "kws", "processed")
        os.makedirs(out_dir, exist_ok=True)

        self.full_raw_dataset = self.full_raw_dataset.cast_column(
            "audio", datasets.Audio(sampling_rate=sample_rate, decode=False)
        )

        kws_dataset = self.full_raw_dataset.filter(
            lambda item: self.id2label[item["label"]].strip().lower() in self.key_words
        )
        unknown_dataset = self.full_raw_dataset.filter(
            lambda item: (self.id2label[item["label"]].strip().lower() not in self.key_words) and
                         (self.id2label[item["label"]].strip().lower() != "_silence_")
        )
        official_silence_dataset = self.full_raw_dataset.filter(
            lambda item: self.id2label[item["label"]].strip().lower() == "_silence_"
        )

        def feature_extract(batch, idx):
            label_str = self.id2label[batch["label"]].strip().lower()
            if label_str not in self.key_words and label_str != "_silence_":
                label_str = "unknown"

            audio_bytes = batch["audio"]["bytes"]
            with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
                wav = f.read(dtype="float32")
                sr = f.samplerate

            if wav.ndim > 1:
                wav = wav.mean(axis=1)

            if sr != sample_rate:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)

            if len(wav) >= wave_length:
                wav = wav[:wave_length]
            else:
                wav = np.pad(wav, (0, wave_length - len(wav)))

            pcen_db = self._pcen_feature(wav, sample_rate) # [40, 45] [num_mel, num_frame]
            if len(self.key_words) > 10:
                dct_data = self._dct_2d_trans(pcen_db, 32, 32)
            else:
                dct_data = self._dct_2d_trans(pcen_db, 32, 32)

            out_path = os.path.join(out_dir, f"kws_unk_{idx}.npy")
            np.save(out_path, dct_data)

            return {"id": idx, "label_str": label_str, "feature_path": out_path}

        num_data_each_class = int(len(kws_dataset) / self.num_kws)

        existed_num_silence = len(official_silence_dataset)
        target_num_silence = int(num_data_each_class * 0.5)

        if existed_num_silence >= target_num_silence:
            num_silence_from_official = target_num_silence
            to_generate_silence = 0
        else:
            num_silence_from_official = existed_num_silence
            to_generate_silence = target_num_silence - existed_num_silence

        existed_num_unknown = len(unknown_dataset)
        target_num_unknown = int(num_data_each_class * 1.0)
        num_unknown_dataset = min(existed_num_unknown, target_num_unknown)

        kws_unknown_dataset = datasets.concatenate_datasets(
            [
                kws_dataset,
                unknown_dataset.shuffle(seed=42).select(range(num_unknown_dataset)),
                official_silence_dataset.shuffle(seed=42).select(range(num_silence_from_official)),
            ]
        )

        silence_start_idx = len(kws_unknown_dataset)
        synthetic_silence_dataset = self.generate_silence_dataset(
            to_generate_silence,
            silence_start_idx,
            wave_length=wave_length,
            sample_rate=sample_rate,
            dataset_dir=out_dir
        )

        print("Processing dataset (extracting features)...")
        kws_unknown_dataset = kws_unknown_dataset.map(feature_extract, with_indices=True)
        print("Finish processing dataset.")

        kws_unknown_dataset = kws_unknown_dataset.remove_columns(
            [c for c in kws_unknown_dataset.column_names if c not in ["id", "label_str", "feature_path"]]
        )
        aligned_features = datasets.Features({
            "id": datasets.Value("int64"),
            "label_str": datasets.Value("string"),
            "feature_path": datasets.Value("string"),
        })
        kws_unknown_dataset = kws_unknown_dataset.cast(aligned_features)
        synthetic_silence_dataset = synthetic_silence_dataset.cast(aligned_features)

        final_dataset = datasets.concatenate_datasets([synthetic_silence_dataset, kws_unknown_dataset])

        split_dataset = final_dataset.shuffle(seed=42).train_test_split(test_size=0.3)
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]

        self.fo_train_dataset = self.generate_fiftyone_dataset(train_dataset, train=True)
        self.fo_test_dataset = self.generate_fiftyone_dataset(test_dataset, train=False)

    def generate_fiftyone_dataset(self, hf_dataset, train=True):
        dataset_name = "kws_pcen_train" if train else "kws_pcen_test"

        try:
            fo.delete_dataset(dataset_name)
        except Exception:
            pass

        fo_dataset = fo.Dataset(dataset_name)

        samples = []
        for data in hf_dataset:
            sample = fo.Sample(filepath=data["feature_path"])
            sample["id"] = int(data["id"])
            sample["label"] = data["label_str"]
            samples.append(sample)

        fo_dataset.add_samples(samples)
        return fo_dataset

    def store_fo_dataset(self, export_dir="./data"):
        os.makedirs(export_dir, exist_ok=True)

        generated_structure = fo.types.FiftyOneDataset
        if self.num_kws >= len(ALL_CLASSES):
            store_path = os.path.join(export_dir, "kws-all")
        else:
            store_path = os.path.join(export_dir, "kws")

        train_store_path = os.path.join(store_path, "train")
        test_store_path = os.path.join(store_path, "test")
        os.makedirs(train_store_path, exist_ok=True)
        os.makedirs(test_store_path, exist_ok=True)

        self.fo_train_dataset.export(
            export_dir=train_store_path,
            dataset_type=generated_structure,
            label_field="label",
            export_media=True
        )
        self.fo_test_dataset.export(
            export_dir=test_store_path,
            dataset_type=generated_structure,
            label_field="label",
            export_media=True
        )

if __name__ == "__main__":
    preparer = KWSDatasetPreparer(dataset_dir="../../data")
    preparer.load_raw_dataset()
    preparer.preprocess_dataset(sample_rate=16000, wave_length=16000)
    preparer.store_fo_dataset(export_dir="../../data")

    preparer = KWSDatasetPreparer(dataset_dir="../../data", key_words="all")
    preparer.load_raw_dataset()
    preparer.preprocess_dataset(sample_rate=16000, wave_length=16000)
    preparer.store_fo_dataset(export_dir="../../data")

    # import matplotlib.pyplot as plt
    # def npy_to_png(npy_path, out_dir):
    #     os.makedirs(out_dir, exist_ok=True)
    #     arr = np.load(npy_path)
    #
    #     out_path = os.path.join(
    #         out_dir, os.path.basename(npy_path).replace(".npy", ".png")
    #     )
    #
    #     plt.imsave(out_path, arr, cmap="magma")
    #     return out_path
    # train_data = fo.Dataset.from_dir(
    #     dataset_dir="../../data/kws/train",
    #     dataset_type=fo.types.FiftyOneDataset,
    # )
    # for sample in train_data.iter_samples(progress=True):
    #     png_path = npy_to_png(sample.filepath, "../../data/kws/png")
    #     sample.filepath = png_path
    #     sample.save()
    # session = fo.launch_app(train_data)
    # session.wait()
    # session.close()
