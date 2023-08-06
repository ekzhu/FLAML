import json
import pkgutil
from typing import Any, Callable, Dict, List, Optional
from langchain.tools.python.tool import PythonAstREPLTool

from flaml.autogen.agentchat2.llm_chat import (
    LLMChatAgent,
    LLMChatContext,
    LLMChatMessage,
    llm_chat_action,
    llm_chat_save_messages,
)
from flaml.autogen.agentchat2.stream import MessageStream

_modules_to_try = [
    "absl-py",
    "affine",
    "aiohttp",
    "aiosignal",
    "analytics-pythonpost1",
    "anyio",
    "anytree",
    "argcomplete",
    "argon2-cffi-bindings",
    "argon2-cffi",
    "arviz",
    "asttokens",
    "async-timeout",
    "attrs",
    "audioread",
    "babel",
    "backcall",
    "backoff",
    "backports.zoneinfo",
    "basemap-data",
    "basemap",
    "bcrypt",
    "beautifulsoup4",
    "bleach",
    "blinker",
    "blis",
    "bokeh",
    "branca",
    "brotli",
    "cachetools",
    "cairocffi",
    "cairosvg",
    "camelot-py",
    "catalogue",
    "certifi",
    "cffi",
    "chardet",
    "charset-normalizer",
    "click-plugins",
    "click",
    "cligj",
    "cloudpickle",
    "cmudict",
    "comm",
    "compressed-rtf",
    "countryinfo",
    "cryptography",
    "cssselect2",
    "cycler",
    "cymem",
    "dbus-python",
    "debugpy",
    "decorator",
    "defusedxml",
    "deprecat",
    "dill",
    "distro-infoubuntu1",
    "dlib",
    "dnspython",
    "docx2txt",
    "ebcdic",
    "ebooklib",
    "einops",
    "email-validatorpost2",
    "entrypoints",
    "et-xmlfile",
    "exceptiongroup",
    "exchange-calendars",
    "executing",
    "extract-msg",
    "faker",
    "fastapi",
    "fastjsonschema",
    "fastprogress",
    "ffmpeg-python",
    "ffmpy",
    "filelock",
    "fiona",
    "flask-cachebuster",
    "flask-cors",
    "flask-login",
    "flask",
    "folium",
    "fonttools",
    "fpdf",
    "frozenlist",
    "future",
    "fuzzywuzzy",
    "gensim",
    "geographiclib",
    "geopandas",
    "geopy",
    "gradio",
    "graphviz",
    "gtts",
    "h11",
    "h2",
    "h5netcdf",
    "h5py",
    "hpack",
    "html5lib",
    "httpcore",
    "httptools",
    "httpx",
    "hypercorn",
    "hyperframe",
    "idna",
    "imageio-ffmpeg",
    "imageio",
    "imapclient",
    "imgkit",
    "importlib-metadata",
    "importlib-resources",
    "iniconfig",
    "ipykernel",
    "ipython-genutils",
    "ipython",
    "isodate",
    "itsdangerous",
    "jax",
    "jedi",
    "jinja2",
    "joblib",
    "json5",
    "jsonpickle",
    "jsonschema-specifications",
    "jsonschema",
    "jupyter-client",
    "jupyter-core",
    "jupyter-server",
    "jupyterlab-pygments",
    "jupyterlab-server",
    "jupyterlab",
    "keras",
    "kerykeion",
    "kiwisolver",
    "korean-lunar-calendar",
    "librosa",
    "llvmlite",
    "loguru",
    "lxml",
    "markdown2",
    "markdownify",
    "markupsafe",
    "matplotlib-inline",
    "matplotlib-venn",
    "matplotlib",
    "mistune",
    "mizani",
    "mne",
    "monotonic",
    "moviepy",
    "mpmath",
    "mtcnn",
    "multidict",
    "munch",
    "murmurhash",
    "mutagen",
    "nashpy",
    "nbclassic",
    "nbclient",
    "nbconvert",
    "nbformat",
    "nest-asyncio",
    "networkx",
    "nltk",
    "notebook-shim",
    "notebook",
    "numba",
    "numexpr",
    "numpy-financial",
    "numpy",
    "odfpy",
    "olefile",
    "opencv-python",
    "openpyxl",
    "opt-einsum",
    "orjson",
    "packaging",
    "pandas",
    "pandocfilters",
    "paramiko",
    "parso",
    "pathy",
    "patsy",
    "pdf2image",
    "pdfkit",
    "pdfminer.six",
    "pdfplumber",
    "pdfrw",
    "pexpect",
    "pickleshare",
    "pillow",
    "pip",
    "pkgutil-resolve-name",
    "platformdirs",
    "plotly",
    "plotnine",
    "pluggy",
    "pooch",
    "preshed",
    "priority",
    "proglog",
    "prometheus-client",
    "prompt-toolkit",
    "pronouncing",
    "psutil",
    "ptyprocess",
    "pure-eval",
    "py",
    "pyaudio",
    "pycountry",
    "pycparser",
    "pycryptodome",
    "pydantic",
    "pydot",
    "pydub",
    "pydyf",
    "pygments",
    "pygobject",
    "pygraphviz",
    "pylog",
    "pyluach",
    "pymc3",
    "pymupdf",
    "pynacl",
    "pypandoc",
    "pyparsing",
    "pypdf2",
    "pyphen",
    "pyproj",
    "pyprover",
    "pyshp",
    "pyswisseph",
    "pytesseract",
    "pytest",
    "pyth3",
    "python-apt",
    "python-dateutil",
    "python-docx",
    "python-dotenv",
    "python-multipart",
    "python-pptx",
    "pyttsx3",
    "pytz",
    "pywavelets",
    "pyxlsb",
    "pyyaml",
    "pyzbar",
    "pyzmq",
    "qrcode",
    "rarfile",
    "rasterio",
    "rdflib",
    "referencing",
    "regex",
    "reportlab",
    "requests-unixsocket",
    "requests",
    "resampy",
    "rpds-py",
    "scikit-image",
    "scikit-learn",
    "scipy",
    "seaborn",
    "semver",
    "send2trash",
    "sentencepiece",
    "setuptools",
    "shap",
    "shapely",
    "six",
    "slicer",
    "smart-open",
    "sniffio",
    "snuggs",
    "sortedcontainers",
    "soundfile",
    "soupsieve",
    "spacy-legacy",
    "spacy",
    "speechrecognition",
    "srsly",
    "stack-data",
    "starlette",
    "statsmodels",
    "svglib",
    "svgwrite",
    "sympy",
    "tables",
    "tabula",
    "tabulate",
    "tenacity",
    "terminado",
    "text-unidecode",
    "textblob",
    "textract",
    "theano-pymc",
    "thinc",
    "threadpoolctl",
    "tifffile",
    "tinycss2",
    "toml",
    "tomli",
    "toolz",
    "torch",
    "torchaudio",
    "torchtext",
    "torchvision",
    "tornado",
    "tqdm",
    "traitlets",
    "trimesh",
    "typer",
    "typing-extensions",
    "tzlocal",
    "ujson",
    "unattended-upgrades",
    "urllib3",
    "uvicorn",
    "uvloop",
    "wand",
    "wasabi",
    "watchfiles",
    "wcwidth",
    "weasyprint",
    "webencodings",
    "websocket-client",
    "websockets",
    "werkzeug",
    "wheel",
    "wordcloud",
    "wrapt",
    "wsproto",
    "xarray-einstats",
    "xarray",
    "xgboost",
    "xlrd",
    "xlsxwriter",
    "xml-python",
    "yarl",
    "zipp",
    "zopfli",
]
default_modules = [p.name for p in pkgutil.iter_modules() if p.name in _modules_to_try]


class CodeInterpreterAgent(LLMChatAgent):
    def __init__(
        self,
        name: str,
        address: str,
        message_stream: MessageStream,
        llm_config: Dict[str, Any],
        working_dir: str,
        upload_dir: str,
        output_dir: str,
        modules: List[str] = default_modules,
        trigger: Optional[Callable[[List[LLMChatMessage], LLMChatContext], bool]] = lambda messages, context: True,
        action: Optional[Callable[[List[LLMChatMessage], LLMChatContext], LLMChatContext]] = llm_chat_action,
        default_action: Optional[
            Callable[[List[LLMChatMessage], LLMChatContext], LLMChatContext]
        ] = llm_chat_save_messages,
    ) -> None:
        # System message modified based on: https://github.com/iamgreggarcia/codesherpa/blob/086f07e58c4e1dcdaf70f46bddc66e964d9662a7/frontend/src/constants/openai.ts#L137
        system_message_text = """You are an AI code interpreter with access to Python REPL, capable of writing, executing, and debugging code. Follow these guidelines:

1. Verify actions and values with the user before execution.
2. Stick to provided functions.
3. Use {working_dir} as the working directory for saving intermediate files.
4. If you write Python code, offer to execute it upon completion using Python_REPL.
5. Respond using markdown syntax.

User file handling:
- Unless specified, files are read from the {upload_dir} directory. For example, to read '{upload_dir}/data.csv', use:
```python
with open("{upload_dir}/data.csv", "r") as file:
    file_contents = file.read()
```

Python execution:
- Perform general programming tasks, data analysis, visualizations, etc.
- Use built-in modules and pre-installed packages: {modules}.

Visualizations and file creations:
- Save plots and media files to '{output_dir}' directory.
- Embed files in responses using URL. For example, to embed '{output_dir}/plot.png', use: ![plot]({output_dir}/plot.png).""".format(
            working_dir=working_dir,
            upload_dir=upload_dir,
            output_dir=output_dir,
            modules=", ".join(modules),
        )
        llm_config = llm_config or {}
        llm_config.setdefault("functions", []).append(
            {
                "name": "Python_REPL",
                "description": "Python REPL for code execution. Returns stdout and stderr.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute",
                        },
                    },
                    "required": ["code"],
                },
            }
        )
        repl_tool = PythonAstREPLTool()

        def _repl_func(code: str):
            return repl_tool.run(code)

        functions = {"Python_REPL": _repl_func}
        super().__init__(
            name=name,
            address=address,
            message_stream=message_stream,
            system_message={"role": "system", "content": system_message_text},
            llm_config=llm_config,
            functions=functions,
            trigger=trigger,
            action=action,
            default_action=default_action,
        )
