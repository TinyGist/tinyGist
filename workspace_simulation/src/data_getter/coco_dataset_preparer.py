import os
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField
from PIL import Image, UnidentifiedImageError


class COCODatasetPreparation:
    def __init__(
            self, task='human',
            dataset_dir='./data',
            max_samples: int | None=None,
    ):
        if task in ['human', 'vehicle']:
            self.task = task
        else:
            raise RuntimeError('Task must be either "human" or "vehicle"')
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
    def resize_dataset(input_view: fo.DatasetView, output_size: int, output_dir: str) -> fo.DatasetView:
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

    @staticmethod
    def select_target_num_boxes(input_view: fo.DatasetView, target_num_boxes: int) -> fo.DatasetView:
        editable_view = input_view.clone()
        for sample in editable_view.iter_samples(progress=True):
            detections = sample.ground_truth.detections
            detections_sorted = sorted(
                detections,
                key=lambda d: d.bounding_box[2] * d.bounding_box[3],
                reverse=True,
            )
            sample.ground_truth.detections = detections_sorted[:target_num_boxes]
            sample.save()
        return editable_view

    def generate_new_dataset_annotations(self, export_data: bool=False, resize: bool=False, output_size: int=128, maximum_boxes: int=1):
        if self.task == 'human':
            filtered_train_view = self.coco_2017_train.filter_labels(
                "ground_truth",
                ViewField("label") == "person",
                only_matches=False
            )
            filtered_val_view = self.coco_2017_val.filter_labels(
                "ground_truth",
                ViewField("label") == "person",
                only_matches=False
            )
        elif self.task == 'vehicle':
            filtered_train_view = self.coco_2017_train.filter_labels(
                "ground_truth",
                ViewField("label").is_in(["truck", "bus", "car", "bicycle", "motorcycle"]),
                only_matches=False
            )
            filtered_val_view = self.coco_2017_val.filter_labels(
                "ground_truth",
                ViewField("label").is_in(["truck", "bus", "car", "bicycle", "motorcycle"]),
                only_matches=False
            )
        else:
            raise RuntimeError('Task must be either "human" or "vehicle"')

        with_classes_train_view = filtered_train_view.match(
            ViewField("ground_truth.detections").length() > 0
        ).match(
            ViewField("ground_truth.detections").length() <= maximum_boxes
        )
        without_classes_train_view = filtered_train_view.match(
            ViewField("ground_truth.detections").length() == 0
        )
        with_classes_val_view = filtered_val_view.match(
            ViewField("ground_truth.detections").length() > 0
        ).match(
            ViewField("ground_truth.detections").length() <= maximum_boxes
        )
        without_classes_val_view = filtered_val_view.match(
            ViewField("ground_truth.detections").length() == 0
        )

        # with_classes_train_view = self.select_target_num_boxes(with_classes_train_view, maximum_boxes)
        # with_classes_val_view = self.select_target_num_boxes(with_classes_val_view, maximum_boxes)
        # without_classes_train_view = self.select_target_num_boxes(without_classes_train_view, 0)
        # without_classes_val_view = self.select_target_num_boxes(without_classes_val_view, 0)

        num_data_with_classes_train = len(with_classes_train_view)
        num_data_without_classes_train = len(without_classes_train_view)
        num_data_with_classes_val = len(with_classes_val_view)
        num_data_without_classes_val = len(without_classes_val_view)
        if self.task == 'human':
            target_num_data_without_classes_train = num_data_with_classes_train * 0.5
            target_num_data_without_classes_train = min(target_num_data_without_classes_train,
                                                    num_data_without_classes_train)
            target_num_data_without_classes_val = num_data_with_classes_val * 0.5
            target_num_data_without_classes_val = min(target_num_data_without_classes_val,
                                                     num_data_without_classes_val)
        elif self.task == 'vehicle':
            target_num_data_without_classes_train = num_data_with_classes_train * 0.2
            target_num_data_without_classes_train = min(target_num_data_without_classes_train,
                                                    num_data_without_classes_train)
            target_num_data_without_classes_val = num_data_with_classes_val * 0.5
            target_num_data_without_classes_val = min(target_num_data_without_classes_val,
                                                   num_data_without_classes_val)
        else:
            raise RuntimeError('Task must be either "human" or "vehicle"')

        without_classes_train_view = without_classes_train_view.shuffle().limit(int(target_num_data_without_classes_train))
        without_classes_val_view = without_classes_val_view.shuffle().limit(int(target_num_data_without_classes_val))

        final_with_classes_train_view = with_classes_train_view.concat(without_classes_train_view).shuffle()
        final_with_classes_val_view = with_classes_val_view.concat(without_classes_val_view).shuffle()

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

        export_dir = os.path.join(self.dataset_dir, "coco-2017-yolo-"+self.task)
        train_export_dir = os.path.join(export_dir, 'train')
        val_export_dir = os.path.join(export_dir, 'validation')
        os.makedirs(train_export_dir, exist_ok=True)
        os.makedirs(val_export_dir, exist_ok=True)

        if resize:
            temp_export_dir = os.path.join(export_dir, 'temp')
            output_size = 88 if output_size<88 else output_size
            final_with_classes_train_view = self.resize_dataset(final_with_classes_train_view, output_size, temp_export_dir)
            final_with_classes_val_view = self.resize_dataset(final_with_classes_val_view, output_size, temp_export_dir)

        if export_data:
            if not resize:
                raise RuntimeWarning("If want to export data, it is recommended to resize the dataset.")
            final_with_classes_train_view.export(
                export_dir=train_export_dir,
                dataset_type=fo.types.COCODetectionDataset,
                label_field="ground_truth",
                export_media=True
            )
            final_with_classes_val_view.export(
                export_dir=val_export_dir,
                dataset_type=fo.types.COCODetectionDataset,
                label_field="ground_truth",
                export_media=True
            )
        else:
            if resize:
                raise RuntimeWarning(f"Images are resized to {output_size}x{output_size}, but data is not exported")
            final_with_classes_train_view.export(
                export_dir=train_export_dir,
                dataset_type=fo.types.COCODetectionDataset,
                label_field="ground_truth",
                export_media=False
            )
            final_with_classes_val_view.export(
                export_dir=val_export_dir,
                dataset_type=fo.types.COCODetectionDataset,
                label_field="ground_truth",
                export_media=False
            )


if __name__ == '__main__':
    preparer = COCODatasetPreparation(dataset_dir='../../data', task="vehicle")
    preparer.load_raw_dataset()
    preparer.generate_new_dataset_annotations(export_data=True, resize=True, output_size=128, maximum_boxes=2)
    preparer.task = "human"
    preparer.generate_new_dataset_annotations(export_data=True, resize=True, output_size=128, maximum_boxes=2)



    # train_data = fo.Dataset.from_dir(
    #     dataset_dir='../../data/coco-2017-yolo-vehicle/train',
    #     dataset_type=fo.types.COCODetectionDataset,
    # )
    # session = fo.launch_app(train_data)
    # session.wait()
    # session.close()

