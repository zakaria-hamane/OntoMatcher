import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Callable, Union
import logging
import bs4
import pystow
import pystow.utils
import requests
from pystow.utils import name_from_url
import os


class UMLS_Downloader:
    """
    A class that serves as a downloader for UMLS data.

    Attributes:
        api_key (str): Your API key for UMLS.
        umls_version (str): The UMLS version you want to download.
        output_directory (str): The directory where the downloaded files will be stored.

    Methods:
        download_tgt: Downloads Ticket Granting Ticket (TGT) from UMLS.
        download_tgt_versioned: Downloads versioned TGT from UMLS.
        _download_umls: Private method to download UMLS files.
        download_umls: Downloads UMLS files.
        download_umls_metathesaurus: Downloads UMLS Metathesaurus files.
        open_umls: Context manager to open and close UMLS files.
        download_umls_metathesaurus_full: Downloads the full UMLS Metathesaurus.
        download_and_extract_files: Downloads and extracts specific files from the UMLS Metathesaurus.
    """
    def __init__(self, api_key: str, umls_version: str) -> None:
        self.logger = logging.getLogger(__name__)
        self.MODULE = pystow.module("bio", "umls")
        self.TGT_URL = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
        self.UMLS_URL_FMT = "https://download.nlm.nih.gov/umls/kss/{version}/umls-{version}-mrconso.zip"
        self.UMLS_METATHESAURUS_URL_FMT = (
            "https://download.nlm.nih.gov/umls/kss/{version}/umls-{version}-metathesaurus-full.zip"
        )
        self.api_key = api_key
        self.umls_version = umls_version
        self.output_directory = os.getcwd()

    def download_tgt(self, url: str, path: Union[str, Path], *, api_key: Optional[str] = None, force: bool = False) -> None:
        """
        Download TGT from the given URL and store it at the given path.

        Args:
            url (str): The URL of the TGT.
            path (Union[str, Path]): The directory where the TGT will be stored.
            force (bool, optional): Force download even if the file already exists. Defaults to False.
        """
        path = Path(path).resolve()
        if path.is_file() and not force:
            return
        api_key = pystow.get_config("umls", "api_key", passthrough=api_key, raise_on_missing=True)
        auth_res = requests.post(self.TGT_URL, data={"apikey": api_key})
        soup = bs4.BeautifulSoup(auth_res.text, features="html.parser")
        action_url = soup.find("form").attrs["action"]
        self.logger.info("[umls] got TGT url: %s", action_url)
        key_res = requests.post(action_url, data={"service": url})
        service_ticket = key_res.text
        self.logger.info("[umls] got service ticket: %s", service_ticket)
        pystow.utils.download(
            url=url,
            path=path,
            backend="requests",
            params={"ticket": service_ticket},
        )

    def download_tgt_versioned(self, url_fmt: str, version: Optional[str] = None, *, module_key: str,
                               version_key: str, api_key: Optional[str] = None, force: bool = False,
                               version_transform: Optional[Callable[[str], str]] = None,) -> Path:
        """
        Download a specific version of TGT.

        Args:
            url_fmt (str): The URL format of the TGT.
            version (Optional[str], optional): The version of the TGT. Defaults to None.
            module_key (str): The module key for the TGT.
            version_key (str): The version key for the TGT.
            api_key (Optional[str], optional): The API key for the TGT. Defaults to None.
            force (bool, optional): Force download even if the file already exists. Defaults to False.
            version_transform (Optional[Callable[[str], str]], optional): A function to transform the version number. Defaults to None.

        Returns:
            Path: The path of the downloaded TGT.
        """
        if "{version}" not in url_fmt:
            raise ValueError("URL string can't format in a version")
        if version is None:
            import bioversions

            version = bioversions.get_version(version_key)
        if version is None:
            raise RuntimeError(f"Could not get version for {version_key}")
        if version_transform:
            version = version_transform(version)
        url = url_fmt.format(version=version)
        path = pystow.join("bio", module_key, version, name=name_from_url(url))
        self.download_tgt(url, path, api_key=api_key, force=force)
        return path

    def _download_umls(self, url_fmt: str, version: Optional[str] = None, api_key: Optional[str] = None,
                       force: bool = False) -> Path:
        """
        Private method to download UMLS files.

        Args:
            url_fmt (str): The URL format of the UMLS files.
            version (Optional[str], optional): The version of the UMLS files. Defaults to None.
            api_key (Optional[str], optional): The API key for the UMLS files. Defaults to None.
            force (bool, optional): Force download even if the file already exists. Defaults to False.

        Returns:
            Path: The path of the downloaded UMLS files.
        """
        return self.download_tgt_versioned(
            url_fmt=url_fmt,
            version=version,
            version_key="umls",
            module_key="umls",
            api_key=api_key,
            force=force,
        )

    def download_umls(self, version: Optional[str] = None, api_key: Optional[str] = None, force: bool = False) -> Path:
        """
        Download UMLS files.

        Args:
            version (Optional[str], optional): The version of the UMLS files. Defaults to None.
            api_key (Optional[str], optional): The API key for the UMLS files. Defaults to None.
            force (bool, optional): Force download even if the file already exists. Defaults to False.

        Returns:
            Path: The path of the downloaded UMLS files.
        """
        return self._download_umls(url_fmt=self.UMLS_URL_FMT, version=version, api_key=api_key, force=force)

    def download_umls_metathesaurus(self, version: Optional[str] = None, api_key: Optional[str] = None,
                                    force: bool = False) -> Path:
        """
        Download UMLS Metathesaurus files.

        Args:
            version (Optional[str], optional): The version of the UMLS Metathesaurus files. Defaults to None.
            api_key (Optional[str], optional): The API key for the UMLS Metathesaurus files. Defaults to None.
            force (bool, optional): Force download even if the file already exists. Defaults to False.

        Returns:
            Path: The path of the downloaded UMLS Metathesaurus files.
        """
        return self._download_umls(
            url_fmt=self.UMLS_METATHESAURUS_URL_FMT, version=version, api_key=api_key, force=force
        )

    @contextmanager
    def open_umls(self, version: Optional[str] = None, api_key: Optional[str] = None, force: bool = False):
        """
        Context manager to open and close UMLS files.

        Args:
            path (Union[str, Path]): The path of the UMLS file.
            mode (str, optional): The mode in which to open the file. Defaults to 'r'.

        Yields:
            Generator[IO[Any], None, None]: The opened UMLS file.
        """
        path = self.download_umls(version=version, api_key=api_key, force=force)
        with zipfile.ZipFile(path) as zip_file:
            with zip_file.open("MRCONSO.RRF", mode="r") as file:
                yield file

    def download_umls_metathesaurus_full(self, version: Optional[str] = None, api_key: Optional[str] = None,
                                         force: bool = False, output_dir: Optional[Union[str, Path]] = None,) -> Path:
        """
        Download the full UMLS Metathesaurus.
        """
        output_dir = pystow.join("bio", "umls", version)

        url = self.UMLS_METATHESAURUS_URL_FMT.format(version=version)
        file_name = name_from_url(url)

        # Use the absolute path to output_directory
        output_path = os.path.join(self.output_directory, file_name)

        self.download_tgt(url, output_path, api_key=api_key, force=force)
        return output_path

    def download_and_extract_files(self):
        """
        Download and extract specific files from the UMLS Metathesaurus.
        """
        umls_path = self.download_umls_metathesaurus_full(version=self.umls_version, api_key=self.api_key)

        print(f"UMLS Metathesaurus {self.umls_version} full release downloaded at {umls_path}")

        # Use the absolute path to output_directory
        with zipfile.ZipFile(umls_path, 'r') as zip_file:
            for file_name in ["MRCONSO.RRF", "MRDEF.RRF", "MRHIER.RRF", "MRREL.RRF", "MRDOC.RRF", "MRSTY.RRF"]:
                zip_file.extract(f"2022AB/META/{file_name}", self.output_directory)

downloader = UMLS_Downloader(api_key="7a7aa541-de60-42f1-a032-beb246789dd1", umls_version='2022AB')
downloader.download_and_extract_files()
