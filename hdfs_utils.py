import os
from pathlib import Path
import tempfile
import shutil
from typing import List, Union
from pyarrow.filesystem import resolve_filesystem_and_path

def copy_from_hdfs_to_local(hdfs_path: str, local_dir: str) -> List[str]:
    hdfs, hdfs_path = resolve_filesystem_and_path(hdfs_path)
    os.makedirs(local_dir, exist_ok=True)

    root_hdfs_path = None
    file_paths = []
    if hdfs.isdir(hdfs_path):
        # copy all files in hdfs_path to local_dir
        for root, directories, files in hdfs.walk(hdfs_path):
            if root_hdfs_path is None:
                root_hdfs_path = root
                local_root = local_dir
            else:
                local_root = os.path.join(
                    local_dir, root.replace(root_hdfs_path, "").lstrip("/")
                )

            for directory in directories:
                local_directory = os.path.join(local_root, directory)
                if not os.path.exists(local_directory):
                    os.mkdir(local_directory)
            for filename in files:
                local_file_path = os.path.join(local_root, filename)
                with open(local_file_path, "wb") as local_file:
                    hdfs.download(f"{root}/{filename}", local_file)
                file_paths.append(local_file_path)
    else:
        # copy the file with path hdfs_path to local_dir
        for root, _, files in hdfs.walk(hdfs_path):
            # should only be one loop, and one file in files
            assert len(files) == 1
            filename = files[0]
            local_file_path = os.path.join(local_dir, filename)
            with open(local_file_path, "wb") as local_file:
                hdfs.download(root, local_file)
            file_paths.append(local_file_path)
    return file_paths


def copy_from_local_to_hdfs(local_dir: str, hdfs_dir: str) -> None:
    if not os.path.isdir(local_dir):
        raise ValueError(f"{local_dir} is not a directory")

    hdfs, hdfs_dir = resolve_filesystem_and_path(hdfs_dir)
    if not hdfs.exists(hdfs_dir):
        hdfs.mkdir(hdfs_dir)
    if not hdfs.isdir(hdfs_dir):
        raise ValueError(f"{hdfs_dir} is not a directory")

    for folder, dirs, files in os.walk(local_dir):
        subfolder = folder[len(local_dir) :]
        if len(subfolder) > 0 and subfolder[0] == "/":
            subfolder = subfolder[1:]
        folder_out = os.path.join(hdfs_dir, subfolder)
        hdfs.mkdir(folder_out)
        for file in files:
            file_in = os.path.join(folder, file)
            file_out = os.path.join(folder_out, file)
            with open(file_in, "rb") as local_file:
                hdfs.upload(file_out, local_file)


class ToLocalDir:
    def __init__(self, path: str):
        self._path = path
        self._is_hdfs_path = self._path.startswith("hdfs:") or self._path.startswith(
            "viewfs:"
        )
        if not self._is_hdfs_path and not Path(self._path).is_dir():
            raise ValueError(
                f"'{self._path}' is not a local directory nor an HDFS path"
            )

    def __enter__(self):
        if self._is_hdfs_path:
            self._local_dir = tempfile.mkdtemp()
            get_model_from_hdfs(hdfs_path=self._path, to=self._local_dir)
        else:
            self._local_dir = self._path
        return self._local_dir

    def __exit__(self, type, value, traceback):
        if self._is_hdfs_path:
            shutil.rmtree(self._local_dir, ignore_errors=True)


class ToLocalFile:
    def __init__(self, path: Union[str, Path]):
        self._path = str(path)
        self._is_hdfs_path = self._path.startswith("hdfs:") or self._path.startswith(
            "viewfs:"
        )

    def __enter__(self):
        if self._is_hdfs_path:
            self._local_dir = tempfile.mkdtemp()
            hdfs, self._path = resolve_filesystem_and_path(self._path)
            self._local_file = self._local_dir + "/" + self._path.split("/")[-1]
            with open(self._local_file, "wb") as lf:
                hdfs.download(self._path, lf)
        else:
            self._local_file = self._path
        return self._local_file

    def __exit__(self, type, value, traceback):
        if self._is_hdfs_path:
            shutil.rmtree(self._local_dir, ignore_errors=True)
