import numpy as np
from pathlib import Path
from os.path import join, exists
import glob
import pylas
import logging

from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import make_dir, DATASET

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class CycloMedia(BaseDataset):
    """
    This class is used to create a dataset based on the CycloMedia dataset,
    and used in visualizer, training, or testing. The dataset is best used for
    semantic segmentation of urban roadways.
    """

    def __init__(self,
                 dataset_path,
                 name='CycloMedia',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 num_points=65536,
                 ignored_label_inds=[],
                 test_result_folder='./test',
                 skip_empty=True,
                 **kwargs):
        """
        Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            name: The name of the dataset (Semantic3D in this case).
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.
            num_points: The maximum number of points to use when splitting the
                dataset.
            ignored_label_inds: A list of labels that should be ignored in the
                dataset.
            test_result_folder: The folder where the test results should be
                stored.
            skip_empty: SPecifies whether empty .laz files should be omitted.

        Returns:
            class: The corresponding class.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         num_points=num_points,
                         ignored_label_inds=ignored_label_inds,
                         test_result_folder=test_result_folder,
                         skip_empty=skip_empty,
                         **kwargs)

        cfg = self.cfg

        self.label_to_names = self.get_label_to_names()

        self.dataset_path = cfg.dataset_path
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort(
            [k for k, v in self.label_to_names.items()]
            )
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array(cfg.ignored_label_inds)

        train_path = cfg.dataset_path + "/train/"
        if exists(train_path):
            self.train_files = self._get_file_list(path=train_path)
        else:
            self.train_files = self._get_file_list(path=cfg.dataset_path)
        self.val_files = self._get_file_list(path=cfg.dataset_path + "/val/")
        self.test_files = self._get_file_list(path=cfg.dataset_path + "/test/")

    def _get_file_list(self, path):
        """Returns a list of .laz files in a given folder. If the folder does not
        exist, an empty list is returned. If skip_empty is set to True, files
        with 0 points will be omitted.

        Args:
            path: the folder to index
        Returns:
            A list of .laz files found in the specified folder.
        """

        if not exists(path):
            return []
        candidates = glob.glob(path + "/*.laz")
        if self.cfg.skip_empty:
            file_list = []
            for file in candidates:
                with pylas.open(file) as f:
                    if f.header.point_count > 0:
                        file_list.append(file)
        else:
            file_list = candidates
        return file_list

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictonary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {
            0: 'Unclassified',
        }
        return label_to_names

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one
            of 'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return CycloMediaSplit(self, split=split)

    def get_split_list(self, split):
        """Returns the list of data splits available.

        Args:
            split: A string identifying the dataset split that is usually one
            of 'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The
            split name should be one of 'training', 'test', 'validation', or
            'all'.
        """
        if split in ['test', 'testing']:
            files = self.test_files
        elif split in ['train', 'training']:
            files = self.train_files
        elif split in ['val', 'validation']:
            files = self.val_files
        elif split in ['all']:
            files = self.val_files + self.train_files + self.test_files
        else:
            raise ValueError("Invalid split {}".format(split))

        return files

    def is_tested(self, attr):
        """Checks if a datum in the dataset has been tested.

        Args:
            dataset: The current dataset to which the datum belongs to.
            attr: The attribute that needs to be checked.

        Returns:
            If the dataum attribute is tested, then resturn the path where the
            attribute is stored; else, returns false.
        """
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        store_path = join(path, self.name, name + '.npy')
        if exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False

    def save_test_result(self, results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the
                attribute passed.
            attr: The attributes that correspond to the outputs passed in
                results.
        """
        cfg = self.cfg
        name = attr['name'].split('.')[0]
        path = cfg.test_result_folder
        make_dir(path)

        pred = results['predict_labels']
        pred = np.array(pred)

        for ign in cfg.ignored_label_inds:
            pred[pred >= ign] += 1

        store_path = join(path, self.name, name + '.npy')
        make_dir(Path(store_path).parent)
        np.save(store_path, pred)
        log.info("Saved {} in {}.".format(name, store_path))


class CycloMediaSplit(BaseDatasetSplit):

    def __init__(self, dataset, split='training'):
        super().__init__(dataset, split=split)

        log.info("Found {} pointclouds for {}".format(len(self.path_list),
                                                      split))

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        log.debug("get_data called {}".format(pc_path))

        data = pylas.read(pc_path)
        points = np.vstack(
            (data.x, data.y, data.z)).astype(np.float32).T
        # TODO: should we subtract RD offsets?
        # points = np.float32(points)

        feat = np.zeros((points.shape[0], 4), dtype=np.float32)
        feat[:, 0] = data['red']
        feat[:, 1] = data['green']
        feat[:, 2] = data['blue']
        feat[:, 3] = data['intensity']
        feat /= 65536

        # TODO: do we store labels separately or merge into laz?
        labels = np.array(data['classification'], dtype=np.int32)

        data = {'point': points, 'feat': feat, 'label': labels}

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace('.txt', '')

        pc_path = str(pc_path)
        split = self.split
        attr = {'idx': idx, 'name': name, 'path': pc_path, 'split': split}

        return attr


DATASET._register_module(CycloMedia)
