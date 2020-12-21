import json
import requests
import os
import pathlib
import hashlib


class DataProvider:
    CHUNK_SIZE = 1 << 20

    def cache_file(self, url, local_filename=None):
        if not local_filename:
            local_filename = url.split('/')[-1]

        pathlib.Path(self.cache_dir()).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(self.cache_dir(), local_filename)

        if not os.path.exists(file_path):
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=self.CHUNK_SIZE):
                        f.write(chunk)
        return file_path

    def cache_dir(self):
        return os.path.join(pathlib.Path.home(), '.sttn', 'data', self.__class__.__name__)

    @staticmethod
    def hash_args(**kwargs) -> str:
        """MD5 hash of a dictionary."""
        md5 = hashlib.md5()
        encoded = json.dumps(kwargs, sort_keys=True).encode()
        md5.update(encoded)
        return md5.hexdigest()
