import os
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField
from PIL import Image, UnidentifiedImageError
import numpy as np

class FOMODatasetPreparation:
    def __init__(
            self,
            dataset_dir='./data',
            task="human",
            max_samples: int | None=None,
    ):
        if task in ['human', 'vehicle']:
            self.task = task
            if task == 'human':
                self.target_objects = ["person"]
                self.target_str_to_idx = {
                    "background": 0,
                    "person": 1,
                }
            else:
                self.target_objects = ["truck", "bus", "car", "bicycle", "motorcycle"]
                self.target_str_to_idx = {
                    "background": 0,
                    "truck": 1,
                    "bus": 2,
                    "car": 3,
                    "bicycle": 4,
                    "motorcycle": 5,
                    "train": 6,
                }
        else:
            raise NotImplementedError

        if max_samples is not None:
            self.max_samples = max_samples
        else:
            self.max_samples = 99999999999 # an extremely large number, means take all data

        self.dataset_dir = dataset_dir
        fo.config.dataset_zoo_dir = self.dataset_dir
        fo.config.default_dataset_dir = self.dataset_dir
        os.makedirs(self.dataset_dir, exist_ok=True)

    def load_raw_dataset(self):
        self.coco_2017_train = foz.load_zoo_dataset(
            'coco-2017',
            split='train',
            max_samples=self.max_samples,
        )
        self.coco_2017_val = foz.load_zoo_dataset(
            'coco-2017',
            split='validation',
            max_samples=self.max_samples,
        )

    def launch_train_dataset_session(self):
        session = fo.launch_app(self.coco_2017_train)
        return session

    def launch_val_dataset_session(self):
        session = fo.launch_app(self.coco_2017_val)
        return session

    @staticmethod
    def process_dataset(input_view: fo.DatasetView, output_size: int, output_dir: str) -> fo.DatasetView:
        os.makedirs(output_dir, exist_ok=True)
        bad_ids = []
        for sample in input_view.iter_samples(progress=True):
            file_name = os.path.basename(sample.filepath)
            output_file_path = os.path.join(output_dir, file_name)
            try:
                with Image.open(sample.filepath) as image:
                    image = image.convert('RGB')
                    image = image.resize((output_size, output_size))
                    image.save(output_file_path)

                    sample.filepath = output_file_path
                    if sample.metadata is None:
                        sample.metadata = fo.ImageMetadata(width=output_size, height=output_size)
                    else:
                        sample.metadata.width = output_size
                        sample.metadata.height = output_size
                    sample.save()
            except (UnidentifiedImageError, OSError) as e:
                print(f"{e} occurred while resizing: {sample.filepath}")
                sample.tags.append("bad_image")
                sample.save()
                bad_ids.append(sample.id)

        clean_view = input_view.exclude(bad_ids)
        print("Deleted bad images from view")

        return clean_view

    def generate_labels(self, input_view: fo.DatasetView, output_dir, output_size):
        store_dir = os.path.join(output_dir, "labels")
        os.makedirs(store_dir, exist_ok=True)
        editable_view = input_view.clone()
        for sample in editable_view.iter_samples(progress=True):
            label_array = np.zeros((output_size, output_size), dtype=np.float32)
            detections = sample.ground_truth.detections
            label_strs = []
            for detection in detections:
                label_strs.append(detection.label)
                label_idx = self.target_str_to_idx[detection.label]
                (x, y, w, h) = detection.bounding_box
                # first label data based on the centroid
                c_x, c_y = x + w / 2, y + h / 2
                c_x, c_y = min(max(int(c_x * output_size), 0), output_size - 1), min(max(int(c_y * output_size), 0), output_size - 1)
                label_array[c_y, c_x] = label_idx

                # second label data based on the area
                # x1 = int(np.floor(x * output_size))
                # y1 = int(np.floor(y * output_size))
                # x2 = int(np.ceil((x + w) * output_size)) - 1
                # y2 = int(np.ceil((y + h) * output_size)) - 1
                # x1 = min(max(x1, 0), output_size - 1)
                # y1 = min(max(y1, 0), output_size - 1)
                # x2 = min(max(x2, 0), output_size - 1)
                # y2 = min(max(y2, 0), output_size - 1)
                # if x2 >= x1 and y2 >= y1:
                #     label_array[y1:y2 + 1, x1:x2 + 1] = label_idx

            label_path = os.path.join(store_dir, f"{sample.id}.npy")
            np.save(label_path, label_array)

            sample["fomo_label_path"] = os.path.join("labels", f"{sample.id}.npy")
            sample["labels_strs"] = label_strs
            sample.save()

        return editable_view

    def generate_new_dataset_annotations(self, output_size: int=96):
        # 96/16 = 6, one cell in output represent a 16x16 block
        # min_area = 0.15*0.15 # minimum area occupies a block at least
        # min_length = 0.15*0.8
        min_area = 0
        min_length = 0
        train_view = self.coco_2017_train.filter_labels(
            "ground_truth",
            (ViewField("bounding_box")[2]*ViewField("bounding_box")[3]>=min_area)
            & (ViewField("bounding_box")[2]>=min_length)
            & (ViewField("bounding_box")[3]>=min_length),
            only_matches=False
        )
        val_view = self.coco_2017_val.filter_labels(
            "ground_truth",
            (ViewField("bounding_box")[2] * ViewField("bounding_box")[3] >= min_area)
            & (ViewField("bounding_box")[2] >= min_length)
            & (ViewField("bounding_box")[3] >= min_length),
            only_matches=False
        )

        train_view = train_view.filter_labels(
            "ground_truth",
            ViewField("label").is_in(self.target_objects),
            only_matches=False
        )
        val_view = val_view.filter_labels(
            "ground_truth",
            ViewField("label").is_in(self.target_objects),
            only_matches=False
        )

        with_classes_train_view = train_view.match(
            ViewField("ground_truth.detections").length() > 0
        )
        with_classes_val_view = val_view.match(
            ViewField("ground_truth.detections").length() > 0
        )
        without_classes_train_view = train_view.match(
            ViewField("ground_truth.detections").length() == 0
        )
        without_classes_val_view = val_view.match(
            ViewField("ground_truth.detections").length() == 0
        )

        num_data_with_classes_train = len(with_classes_train_view)
        num_data_without_classes_train = len(without_classes_train_view)
        num_data_with_classes_val = len(with_classes_val_view)
        num_data_without_classes_val = len(without_classes_val_view)

        target_num_data_without_classes_train = 0
        target_num_data_without_classes_train = min(target_num_data_without_classes_train,
                                                num_data_without_classes_train)
        target_num_data_without_classes_val = 0
        target_num_data_without_classes_val = min(target_num_data_without_classes_val,
                                               num_data_without_classes_val)


        without_classes_train_view = without_classes_train_view.shuffle().limit(int(target_num_data_without_classes_train))
        without_classes_val_view = without_classes_val_view.shuffle().limit(int(target_num_data_without_classes_val))

        final_train_view = with_classes_train_view.concat(without_classes_train_view).shuffle()
        final_val_view = with_classes_val_view.concat(without_classes_val_view).shuffle()


        print("="*30)
        print("Training dataset:")
        print(f"Number of images including classes: {num_data_with_classes_train}\n",
              f"Number of images excluding classes: {target_num_data_without_classes_train}\n",
              f"Total number of images: {num_data_with_classes_train+target_num_data_without_classes_train}\n.")
        print("="*30)
        print("Validation dataset:")
        print(f"Number of images including classes: {num_data_with_classes_val}\n",
              f"Number of images excluding classes: {target_num_data_without_classes_val}\n",
              f"Total number of images: {num_data_with_classes_val + target_num_data_without_classes_val}.")
        print("=" * 30)

        if self.task == 'human':
            export_dir = os.path.join(self.dataset_dir, "coco-2017-fomo-human")
        else:
            export_dir = os.path.join(self.dataset_dir, "coco-2017-fomo-vehicle")
        train_export_dir = os.path.join(export_dir, 'train')
        val_export_dir = os.path.join(export_dir, 'validation')
        os.makedirs(train_export_dir, exist_ok=True)
        os.makedirs(val_export_dir, exist_ok=True)

        img_temp_export_dir = os.path.join(export_dir, 'img_temp')
        final_train_view = self.process_dataset(final_train_view, output_size, img_temp_export_dir)
        final_val_view = self.process_dataset(final_val_view, output_size, img_temp_export_dir)
        final_train_view = self.generate_labels(final_train_view, train_export_dir, int(output_size//16))
        final_val_view = self.generate_labels(final_val_view, val_export_dir, int(output_size//16))

        final_train_view.export(
            export_dir=train_export_dir,
            dataset_type=fo.types.FiftyOneDataset,
            label_field="fomo_label_path",
            export_media=True
        )
        final_val_view.export(
            export_dir=val_export_dir,
            dataset_type=fo.types.FiftyOneDataset,
            label_field="fomo_label_path",
            export_media=True
        )



if __name__ == '__main__':
    preparer = FOMODatasetPreparation(dataset_dir='../../data', task="human")
    preparer.load_raw_dataset()
    preparer.generate_new_dataset_annotations(output_size=96)

    preparer = FOMODatasetPreparation(dataset_dir='../../data', task="vehicle")
    preparer.load_raw_dataset()
    preparer.generate_new_dataset_annotations(output_size=96)

    # train_data = fo.Dataset.from_dir(
    #     dataset_dir='../../data/coco-2017-fomo-vehicle/train',
    #     dataset_type=fo.types.FiftyOneDataset,
    # )
    # session = fo.launch_app(train_data)
    # session.wait()
    # session.close()

