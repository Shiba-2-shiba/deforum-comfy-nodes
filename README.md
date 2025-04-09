# Fork Version

This repository is designed to test for fix following errors

1. Error associated with  「IS_CHANGED」
   
2. Add Color-Matching. This change need adding 「pip install Color-Matching」

3. Compatibility with Python 3.10, 3.12 and PyTorch 2.6.0+cu12.6

4. In Stability matrix, you can manually install deforum-comfy-nodes with the following command


```bash
Your custom_nodes directory  git clone https://github.com/Shiba-2-shiba/deforum-comfy-nodes

"Your directory \StabilityMatrix\Data\Packages\ComfyUI\venv\Scripts\python.exe" -m pip install  color-matcher git+https://github.com/Shiba-2-shiba/deforum-studio.git
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
