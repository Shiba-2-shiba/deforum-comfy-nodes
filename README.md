# Fork Version

This repository is designed to test for fix following errors

1. Error associated with  「IS_CHANGED」
   
2. Add Color-Matching. This change need adding 「pip install Color-Matching」

3. Compatibility with Python 3.10, 3.12 and PyTorch 2.7.0

## Installation

### 1. Standard pip install

In paperspace in pytorch 2.7.0+cuda12.8, you must upgrade install blinker before install.

```bash
!pip install --ignore-installed blinker
%cd /Your directory/ComfyUI/custom_nodes
!git clone https://github.com/Shiba-2-shiba/deforum-comfy-nodes.git
!pip install --no-cache-dir git+https://github.com/Shiba-2-shiba/deforum-studio.git
```

> **Note:** If you are *not* using the Docker image `shibashiba2/paperspace-gradient-base-pytorch270:v1.2`, you must pre-install additional environment packages:
>
> ````bash
> pip install pims==0.7 pandas numexpr simpleeval pydub loguru clip-interrogator contexttimer librosa mutagen py3d pyqt6 pyqt6-qt6 pyqtgraph python-decouple qtpy streamlit moviepy==1.0.3
>
> pip install git+[https://github.com/Shiba-2-shiba/deforum-studio.git](https://github.com/Shiba-2-shiba/deforum-studio.git)
> ````


### 2. Installing within Stability Matrix (Windows)

```bash
%cd /Your directory/ComfyUI/custom_nodes
git clone https://github.com/Shiba-2-shiba/deforum-comfy-nodes.git
> ````

If you run ComfyUI under Stability Matrix on Windows, install directly into the embedded Python:

```bash
"<YourPath>\StabilityMatrix\Data\Packages\ComfyUI\venv\Scripts\python.exe" \
  -m pip install git+https://github.com/Shiba-2-shiba/deforum-studio.git
```

### 3. Fixing Windows „pims“ build errors

Some Windows setups use a zipped `distutils` and fail to build `pims>=0.6.1`. To avoid build errors, extract a live copy of `distutils` and pre-install dependencies:

1. Change to your venv folder:

   ```bat
   cd /d <YourPath>\StabilityMatrix\Data\Packages\ComfyUI\venv
   ```

2. Expand `python310.zip` to a local `_distutils` directory:

   ```bat
   powershell -Command "Expand-Archive -Path Scripts\python310.zip -DestinationPath Lib\_distutils -Force"
   ```

3. Upgrade build tools:

   ```bat
   Scripts\python.exe -m pip install --upgrade pip setuptools wheel
   ```

4. Pre-install binary-compatible `pims` and other dependencies:

   ```bat
   Scripts\python.exe -m pip install pims==0.7 pandas numexpr simpleeval pydub loguru clip-interrogator contexttimer librosa mutagen py3d pyqt6 pyqt6-qt6 pyqtgraph python-decouple qtpy streamlit moviepy==1.0.3

   ```

5. Finally, install Deforum Studio without re-building dependencies:

   ```bat
   Scripts\python.exe -m pip install --no-deps git+https://github.com/Shiba-2-shiba/deforum-studio.git
   ```


# Deforum for ComfyUI

Deforum integration for ComfyUI.

## Installation

To get started with Deforum Comfy Nodes, please make sure ComfyUI is installed and you are using Python v3.10 or these nodes will not work. We recommend using a virtual environment.

Follow the steps below depending on your method of preference.

### ComfyUI Manager

Look for `Deforum Nodes` by `XmYx`

### Manual Install 

To install Deforum for ComfyUI we will clone this repo into the `custom_nodes` folder
```bash
git clone https://github.com/XmYx/deforum-comfy-nodes.git
```

## Recommended Custom Nodes
Here is a list of extra custom nodes that greatly improves the experience of using Deforum.
```bash
https://github.com/rgthree/rgthree-comfy
https://github.com/a1lazydog/ComfyUI-AudioScheduler
https://github.com/cubiq/ComfyUI_IPAdapter_plus
https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet
https://github.com/WASasquatch/was-node-suite-comfyui
https://github.com/11cafe/comfyui-workspace-manager
https://github.com/cubiq/ComfyUI_essentials
https://github.com/FizzleDorf/ComfyUI_FizzNodes
https://github.com/ltdrdata/ComfyUI-Impact-Pack
https://github.com/Fannovel16/ComfyUI-Frame-Interpolation
https://github.com/Fannovel16/ComfyUI-Video-Matting
https://github.com/crystian/ComfyUI-Crystools
```

## Usage

1. Launch ComfyUI
2. Load any of the example workflows from the examples folder.
3. Queue prompt, this will generate your first frame, you can enable Auto queueing, or batch as many images as long you'd
like your animation to be.

## Contribution

We welcome contributions from the community! If you're interested in improving Deforum Comfy Nodes or have ideas for new features, please follow these steps:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or fix.
3. Commit your changes with clear, descriptive messages.
4. Push your changes to the branch and open a pull request.

## License

Deforum Comfy Nodes is licensed under the MIT License. For more details, see the LICENSE file in the repository.

## Community and Support

Join our Discord community to discuss Deforum Comfy Nodes, share your creations, and get help from the developers and other users: 

[Join Discord](https://discord.gg/deforum)

[Visit our website](https://deforum.art)

![Deforum Website Logo](docs/logo.png)
