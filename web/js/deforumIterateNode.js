import { app } from "../../../scripts/app.js";
import { api } from '../../../scripts/api.js'
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Many functions copied from:
// https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
// Thank you!


// Hijack LiteGraph to sort categories and nodes alphabetically (case-insensitive)
(function(LiteGraph) {
    var originalGetNodeTypesCategories = LiteGraph.getNodeTypesCategories;
    LiteGraph.getNodeTypesCategories = function(filter) {
        var categories = originalGetNodeTypesCategories.call(this, filter);
        // Sort categories case-insensitively
        return categories.sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()));
    };

    var originalGetNodeTypesInCategory = LiteGraph.getNodeTypesInCategory;
    LiteGraph.getNodeTypesInCategory = function(category, filter) {
        var nodeTypes = originalGetNodeTypesInCategory.call(this, category, filter);
        // Sort node types case-insensitively by title
        return nodeTypes.sort((a, b) => a.title.toLowerCase().localeCompare(b.title.toLowerCase()));
    };
})(LiteGraph || global.LiteGraph); // Ensure LiteGraph is defined; use global.LiteGraph if it's not directly accessible



document.getElementById("comfy-file-input").accept += ",video/webm,video/mp4";

function chainCallback(object, property, callback) {
    if (object == undefined) {
        //This should not happen.
        console.error("Tried to add callback to non-existant object")
        return;
    }
    if (property in object) {
        const callback_orig = object[property]
        object[property] = function () {
            const r = callback_orig.apply(this, arguments);
            callback.apply(this, arguments);
            return r
        };
    } else {
        object[property] = callback;
    }
}

async function uploadFile(file) {
    //TODO: Add uploaded file to cache with Cache.put()?
    try {
        // Wrap file in formdata so it includes filename
        const body = new FormData();
        const i = file.webkitRelativePath.lastIndexOf('/');
        const subfolder = file.webkitRelativePath.slice(0,i+1)
        const new_file = new File([file], file.name, {
            type: file.type,
            lastModified: file.lastModified,
        });
        body.append("image", new_file);
        if (i > 0) {
            body.append("subfolder", subfolder);
        }
        const resp = await api.fetchApi("/upload/image", {
            method: "POST",
            body,
        });

        if (resp.status === 200) {
            return resp.status
        } else {
            alert(resp.status + " - " + resp.statusText);
        }
    } catch (error) {
        alert(error);
    }
}
function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1] + 20]) // +20 はボタンやマージン用
    node?.graph?.setDirtyCanvas(true);
}

function addVideoPreview(nodeType) {
    chainCallback(nodeType.prototype, "onNodeCreated", function() {
        var element = document.createElement("div");
        const previewNode = this; // 'this' refers to the node instance
        previewNode.currentFps = 24; // Default FPS
        previewNode.playbackInterval = 1000 / previewNode.currentFps;
        previewNode.currentFrames = []; // Holds the URLs of frames to be displayed
        previewNode.currentFrameIndex = 0;
        previewNode.isPlaying = false;
        previewNode.imageSequenceInterval = null;


        var previewWidget = this.addDOMWidget("videopreview", "preview", element, {
            serialize: false,
            hideOnZoom: false,
            getValue() {
                return element.value;
            },
            setValue(v) {
                element.value = v;
            },
        });
        previewWidget.computeSize = function(width) {
            if (this.aspectRatio && !this.parentEl.hidden) {
                let height = (previewNode.size[0]-20)/ this.aspectRatio + 10; // Adjusted for padding/margins
                if (!(height > 0)) {
                    height = 0;
                }
                // Height of image + audio player + button
                const audioHeight = previewWidget.audioEl && !previewWidget.audioEl.hidden ? previewWidget.audioEl.offsetHeight : 0;
                const buttonHeight = previewWidget.playPauseBtn ? previewWidget.playPauseBtn.offsetHeight + 8 : 0; // 8 for margin
                this.computedHeight = height + audioHeight + buttonHeight + 10; // Extra padding
                return [width, this.computedHeight];
            }
            return [width, -4];//no loaded src, widget should not display
        }
        previewWidget.parentEl = document.createElement("div");
        previewWidget.parentEl.className = "deforumVideoSavePreview";
        previewWidget.parentEl.style['width'] = "100%";
        element.appendChild(previewWidget.parentEl);

        previewWidget.imgEl = document.createElement("img");
        previewWidget.imgEl.style['width'] = "100%";
        previewWidget.imgEl.hidden = true; // Hide initially until a frame is loaded
        previewWidget.parentEl.appendChild(previewWidget.imgEl);

        previewWidget.audioEl = document.createElement("audio");
        previewWidget.audioEl.controls = true;
        previewWidget.audioEl.loop = true;
        previewWidget.audioEl.style["width"] = "100%";
        previewWidget.audioEl.style.display = "none"; // Initially hidden
        element.appendChild(previewWidget.audioEl);

        // Play/Pause Button
        previewWidget.playPauseBtn = document.createElement("button");
        previewWidget.playPauseBtn.textContent = "▶️ Play";
        previewWidget.playPauseBtn.style.margin = "4px 0";
        previewWidget.playPauseBtn.style.width = "100%";
        previewWidget.playPauseBtn.onclick = () => {
            if (previewNode.isPlaying) {
                previewNode.stopPlayback();
            } else {
                previewNode.startPlayback();
            }
        };
        element.appendChild(previewWidget.playPauseBtn);

        // Initial fitHeight call, might need to be called again after elements are fully rendered
        fitHeight(previewNode); // Adjust node height to fit widgets

        previewWidget.imgEl.onload = () => {
            previewWidget.aspectRatio = previewWidget.imgEl.naturalWidth / previewWidget.imgEl.naturalHeight;
            fitHeight(previewNode);
        };

        // Method to update a single frame display
        previewNode.showCurrentFrame = function() {
            if (previewNode.currentFrames && previewNode.currentFrames.length > 0) {
                previewWidget.imgEl.hidden = false;
                previewWidget.imgEl.src = previewNode.currentFrames[previewNode.currentFrameIndex];
            } else {
                previewWidget.imgEl.hidden = true;
            }
        };

        previewNode.startPlayback = function() {
            if (previewNode.isPlaying || !previewNode.currentFrames || previewNode.currentFrames.length === 0) return;

            previewNode.isPlaying = true;
            previewWidget.playPauseBtn.textContent = "⏸️ Pause";

            if (previewWidget.audioEl.src && previewWidget.audioEl.readyState >= 2) { // If audio is loaded
                 // Try to sync audio with current frame (approximate)
                const audioTargetTime = (previewNode.currentFrameIndex / previewNode.currentFps) % previewWidget.audioEl.duration;
                if (isFinite(audioTargetTime)) {
                    previewWidget.audioEl.currentTime = audioTargetTime;
                }
                previewWidget.audioEl.play();
            }

            previewNode.playbackInterval = 1000 / previewNode.currentFps;
            previewNode.imageSequenceInterval = setInterval(() => {
                if (!previewNode.currentFrames || previewNode.currentFrames.length === 0) {
                    previewNode.stopPlayback(); // Stop if no frames
                    return;
                }
                previewNode.showCurrentFrame();
                previewNode.currentFrameIndex = (previewNode.currentFrameIndex + 1) % previewNode.currentFrames.length;
            }, previewNode.playbackInterval);
        };

        previewNode.stopPlayback = function() {
            if (!previewNode.isPlaying) return;

            previewNode.isPlaying = false;
            previewWidget.playPauseBtn.textContent = "▶️ Play";
            clearInterval(previewNode.imageSequenceInterval);
            previewNode.imageSequenceInterval = null;
            if (previewWidget.audioEl.src) {
                previewWidget.audioEl.pause();
            }
        };

        previewNode.setPlaybackFPS = function(newFps) {
            if (newFps && typeof newFps === 'number' && newFps > 0) {
                previewNode.currentFps = newFps;
                previewNode.playbackInterval = 1000 / previewNode.currentFps;
                if (previewNode.isPlaying) { // If playing, restart with new interval
                    previewNode.stopPlayback();
                    previewNode.startPlayback();
                }
            }
        };

        previewNode.updateAudio = function (audioUrlInput) {
            let audioUrl = null;
            if (Array.isArray(audioUrlInput) && audioUrlInput.length > 0) {
                audioUrl = audioUrlInput[0]; // Python sends a tuple (url,)
            } else if (typeof audioUrlInput === 'string' && audioUrlInput.trim() !== "") {
                audioUrl = audioUrlInput;
            }


            const wasPlaying = previewNode.isPlaying && !previewWidget.audioEl.paused;
            const currentTime = previewWidget.audioEl.currentTime;

            if (audioUrl) {
                previewWidget.audioEl.style.display = "block";
                if (previewWidget.audioEl.src !== audioUrl) {
                    previewWidget.audioEl.src = audioUrl;
                    previewWidget.audioEl.load();
                    if (wasPlaying) {
                        // Attempt to resume playback after new src is loaded
                        previewWidget.audioEl.oncanplaythrough = function() {
                            if (isFinite(currentTime)) {
                                previewWidget.audioEl.currentTime = currentTime;
                            }
                            previewWidget.audioEl.play();
                            previewWidget.audioEl.oncanplaythrough = null; // Remove listener
                        };
                    }
                } else if (wasPlaying) { // Same src, but was playing, ensure it continues
                    previewWidget.audioEl.play();
                }
            } else {
                previewWidget.audioEl.style.display = "none";
                if (previewWidget.audioEl.src) { // If there was an old src
                    previewWidget.audioEl.pause();
                    previewWidget.audioEl.src = "";
                }
            }
            fitHeight(previewNode); // Adjust height if audio player visibility changed
        };

        // Override onRemoved to clean up interval
        const originalOnRemoved = previewNode.onRemoved;
        previewNode.onRemoved = function() {
            previewNode.stopPlayback(); // Stop playback and clear interval
            if (originalOnRemoved) {
                originalOnRemoved.apply(this, arguments);
            }
        };
    });
}


function addUploadWidget(nodeType, nodeData, widgetName, type="video") {
    chainCallback(nodeType.prototype, "onNodeCreated", function() {
        const pathWidget = this.widgets.find((w) => w.name === widgetName);
        const fileInput = document.createElement("input");
        if (type == "folder") {
            Object.assign(fileInput, {
                type: "file",
                style: "display: none",
                webkitdirectory: true,
                onchange: async () => {
                    const directory = fileInput.files[0].webkitRelativePath;
                    const i = directory.lastIndexOf('/');
                    if (i <= 0) {
                        throw "No directory found";
                    }
                    const path = directory.slice(0,directory.lastIndexOf('/'))
                    if (pathWidget.options.values.includes(path)) {
                        alert("A folder of the same name already exists");
                        return;
                    }
                    let successes = 0;
                    for(const file of fileInput.files) {
                        if (await uploadFile(file) == 200) {
                            successes++;
                        } else {
                            //Upload failed, but some prior uploads may have succeeded
                            //Stop future uploads to prevent cascading failures
                            //and only add to list if an upload has succeeded
                            if (successes > 0) {
                                break
                            } else {
                                return;
                            }
                        }
                    }
                    pathWidget.options.values.push(path);
                    pathWidget.value = path;
                    if (pathWidget.callback) {
                        pathWidget.callback(path)
                    }
                },
            });
        } else if (type == "video") {
            Object.assign(fileInput, {
                type: "file",
                accept: "video/webm,video/mp4,video/mkv,image/gif",
                style: "display: none",
                onchange: async () => {
                    if (fileInput.files.length) {
                        if (await uploadFile(fileInput.files[0]) != 200) {
                            //upload failed and file can not be added to options
                            return;
                        }
                        const filename = fileInput.files[0].name;
                        pathWidget.options.values.push(filename);
                        pathWidget.value = filename;
                        if (pathWidget.callback) {
                            pathWidget.callback(filename)
                        }
                    }
                },
            });
        } else {
            throw "Unknown upload type"
        }
        document.body.append(fileInput);
        let uploadWidget = this.addWidget("button", "choose " + type + " to upload", "image", () => {
            //clear the active click event
            app.canvas.node_widget = null

            fileInput.click();
        });
        uploadWidget.options.serialize = false;
    });
}

function extendNodePrototypeWithFrameCaching(nodeType) {
    // This will be managed by the node instance itself (previewNode.currentFrames)
    // but the caching for 'restore' functionality might still use these.
    // If DeforumVideoSaveNode's logic directly uses these cache methods, they are fine.
    nodeType.prototype.frameCache = [];
    nodeType.prototype.audioCache = ''; // Audio is now handled by URL, so this might be less relevant for preview

    nodeType.prototype.cacheFrames = function(frames) {
        if (Array.isArray(frames)) {
            this.frameCache = [].concat(frames); // Replace cache with new frames for preview purposes
        }
    };

    nodeType.prototype.cacheAudio = function(audioUrl) { // Changed from audioBase64
        if (typeof audioUrl === 'string') {
            this.audioCache = audioUrl;
        }
    };

    nodeType.prototype.clearCache = function() {
        this.frameCache = [];
        this.audioCache = '';
        if (this.currentFrames) this.currentFrames = [];
        if (this.currentFrameIndex) this.currentFrameIndex = 0;
        if (this.updateAudio) this.updateAudio(null); // Clear audio player
        if (this.showCurrentFrame) this.showCurrentFrame(); // Update display to show no image
    };

    nodeType.prototype.getCachedFrames = function() {
        return this.frameCache;
    };

    nodeType.prototype.getCachedAudio = function() {
        return this.audioCache;
    };
}


app.registerExtension({
	name: "deforum.deforumIterator",
	init() {
		const STRING = ComfyWidgets.STRING;
		ComfyWidgets.STRING = function (node, inputName, inputData) {
			const r = STRING.apply(this, arguments);
			r.widget.dynamicPrompts = inputData?.[1].dynamicPrompts;
			return r;
		};
	},
	beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeType.comfyClass === "DeforumIteratorNode") {
            const onIteratorExecuted = nodeType.prototype.onExecuted
            nodeType.prototype.onExecuted = function (message) {
                const r = onIteratorExecuted ? onIteratorExecuted.apply(this, message) : undefined
                for (const w of this.widgets || []) {
                    if (w.name === "reset_counter") {
                        const counterWidget = w;
                        counterWidget.value = false;
                    } else if (w.name === "reset_latent") {
                        const resetWidget = w;
                        resetWidget.value = false;
                    }
                }
                const v = app.nodeOutputs?.[this.id + ""];
                if (!this.flags.collapsed && v) {

                    const counter = v["counter"]
                    const max_frames = v["max_frames"]
                    const enableAutorun = v["enable_autoqueue"][0]

                    if (counter[0] >= max_frames[0]) {
                        if (document.getElementById('autoQueueCheckbox').checked === true) {
                            document.getElementById('autoQueueCheckbox').click();
                        }
                    }

                    if (enableAutorun === true) {
                        if (document.getElementById('autoQueueCheckbox').checked === false) {
                            document.getElementById('autoQueueCheckbox').click();
                            document.getElementById('extraOptions').style.display = 'block';
                        }
                    }
                }
            return r
            }

            const onDrawForeground = nodeType.prototype.onDrawForeground;
			nodeType.prototype.onDrawForeground = function (ctx) {
				const r = onDrawForeground?.apply?.(this, arguments);
				const v = app.nodeOutputs?.[this.id + ""];
				if (!this.flags.collapsed && v) {

					const text = v["counter"] + "";
					ctx.save();
					ctx.font = "bold 48px sans-serif";
					ctx.fillStyle = "dodgerblue";
					const sz = ctx.measureText(text);
					ctx.fillText(text, (this.size[0]) / 2 - sz.width - 5, LiteGraph.NODE_SLOT_HEIGHT * 3);
					ctx.restore();
				}

				return r;
			};
		}  else if (nodeType.comfyClass === "DeforumBigBoneResetNode") {


            const onResetExecuted = nodeType.prototype.onExecuted
            nodeType.prototype.onExecuted = function (message) {
                const r = onResetExecuted ? onResetExecuted.apply(this, message) : undefined
                for (const w of this.widgets || []) {

                    if (w.name === "reset_deforum") {
                        const counterWidget = w;
                        counterWidget.value = false;
                    }
                }
            return r
            }


		} else if (nodeType.comfyClass === "DeforumLoadVideo") {
                addUploadWidget(nodeType, nodeData, "video");

		} else if (nodeType.comfyClass === "DeforumVideoSaveNode") {
            extendNodePrototypeWithFrameCaching(nodeType); // Manages general cache, preview uses its own frame list
            addVideoPreview(nodeType); // Adds enhanced preview logic
            const onVideoSaveExecuted = nodeType.prototype.onExecuted

            let restoreWidget
            nodeType.prototype.onExecuted = function (message) {
                const r = onVideoSaveExecuted ? onVideoSaveExecuted.apply(this, message) : undefined
                let swapSkipSave = false;

                // Handle widget states (dump_now, skip_save, clear_cache)
                for (const w of this.widgets || []) {
                    if (w.name === "dump_now") {
                        const dumpWidget = w;
                        if (dumpWidget.value === true) { swapSkipSave = true; }
                        dumpWidget.value = false;
                        // this.shouldResetAnimation = true; // This property seems unused in preview logic
                    } else if (w.name === "skip_save") {
                        const saveWidget = w;
                        if (swapSkipSave === true) { saveWidget.value = false; }
                    } else if (w.name === "clear_cache") {
                        const cacheClearWidget = w;
                        if (cacheClearWidget.value === true) {
                            this.stopPlayback?.(); // Use optional chaining
                            this.clearCache?.();   // Use optional chaining
                            cacheClearWidget.value = false;
                        }
                    } else if (w.name === "restore_frames_for_preview") { // Assuming this is the widget for restore
                        restoreWidget = w;
                    }
                }

                const output = app.nodeOutputs?.[this.id + ""];
                if (output) {
                    const newFrames = output["frames"]; // Array of URLs
                    const newFps = output["fps"] ? output["fps"][0] : this.currentFps; // fps is an array from python
                    const newAudioUrl = output["audio"]; // Can be null or [url]
                    const currentTotalCached = output["counter"] ? output["counter"][0] : 0;
                    const shouldDumpVideo = output["should_dump"] ? output["should_dump"][0] : false;


                    if (this.setPlaybackFPS) this.setPlaybackFPS(newFps); // Update FPS for playback

                    let framesToDisplay = [];
                    if (newFrames && Array.isArray(newFrames)) {
                         // The `restoreWidget` logic from the original script seems to manage whether to use
                         // all cached frames or only new ones. Here, we simplify:
                         // If `restoreWidget.value` is true OR if cache is empty, use all `newFrames`.
                         // Otherwise, if we want to append, `newFrames` should represent *only new* frames.
                         // For now, let's assume `newFrames` is the complete set to display for this update.
                        framesToDisplay = newFrames;
                    }

                    // If `shouldDumpVideo` is true, it implies a reset or end of sequence.
                    // Cache is typically cleared *after* dump in Python.
                    // JS should reflect this by stopping playback and clearing its display frames.
                    if (shouldDumpVideo) {
                        this.stopPlayback?.();
                        this.currentFrames = []; // Clear internal frame list for preview
                        this.currentFrameIndex = 0;
                        this.cacheFrames([]); // Clear the node's general cache if it's used for restore
                        // After dumping, the python node might send a new set of frames (e.g. if starting a new sequence)
                        // or no frames if it's just a final dump.
                        if (framesToDisplay.length > 0) {
                           this.currentFrames = framesToDisplay; // Load new frames if any
                        }
                        this.showCurrentFrame?.(); // Update display (might be empty)
                        if (restoreWidget) restoreWidget.value = false; // Reset restore toggle
                    } else {
                        // Regular update, not a dump event
                        this.currentFrames = framesToDisplay;
                        // If not already playing and there are frames, show the first one
                        if (!this.isPlaying && this.currentFrames.length > 0) {
                            this.currentFrameIndex = 0;
                            this.showCurrentFrame?.();
                        }
                    }

                    if (this.updateAudio) this.updateAudio(newAudioUrl);

                    // Original logic for `restoreWidget` based on `getCachedFrames` vs `output["counter"]`
                    // This seems tied to the specific caching and restore strategy of the node.
                    // For the preview, `this.currentFrames` now holds the displayable frames.
                    if (restoreWidget) {
                        // This logic might need to be adapted based on how `frameCache` is populated
                        // by `cacheFrames` and used by the "restore" feature.
                        // If `restoreWidget.value` means "show all available frames from Python",
                        // then `this.currentFrames = newFrames` is correct.
                        // The check below was from the original script.
                        // if (this.getCachedFrames().length < currentTotalCached && newFrames.length > 0 && !shouldDumpVideo) {
                        //    restoreWidget.value = true; // Indicate more frames are available in backend than shown
                        //    this.clearCache(); // This might be too aggressive if it clears Python's intended cache
                        //} else if (!shouldDumpVideo) {
                        //    restoreWidget.value = false;
                        //}
                        // For simplicity now, if `shouldDumpVideo` happened, `restoreWidget` is set to false above.
                        // Otherwise, its state is preserved from user input unless Python overrides it.
                    }
                }
                return r;
            }

            const onVideoSaveForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                const r = onVideoSaveForeground?.apply?.(this, arguments);
                const v = app.nodeOutputs?.[this.id + ""];
                if (!this.flags.collapsed && v && typeof v["counter"] !== 'undefined') { // Check if counter exists

                    const frameCount = this.currentFrames ? this.currentFrames.length : (v["counter"] ? v["counter"][0] : 0);
                    const text = frameCount + " frame(s) for preview"; // Updated text
                    ctx.save();
                    ctx.font = "bold 14px sans-serif";

                    const sz = ctx.measureText(text);
                    const textWidth = sz.width;
                    const textHeight = 14; 

                    const rectWidth = textWidth + 20;
                    const rectHeight = textHeight + 10;
                    const rectX = (this.size[0] - rectWidth) / 2;
                    // Position above the preview widget, adjusting for title height
                    const widgetBaseY = this.widgets && this.widgets.length > 0 ? this.widgets[0].last_y : LiteGraph.NODE_TITLE_HEIGHT + 20;
                    const rectY = widgetBaseY - rectHeight - 5; // 5px margin above the widget

                    ctx.fillStyle = "rgba(0, 0, 0, 0.8)";
                    ctx.fillRect(rectX, rectY, rectWidth, rectHeight);

                    ctx.fillStyle = "white";
                    const textX = (this.size[0] - textWidth) / 2;
                    const textY = rectY + textHeight - (textHeight - 14)/2 + 2; // Center text vertically in rect

                    ctx.fillText(text, textX, textY);
                    ctx.restore();
                }
                return r;
            };
		};
	},
});

// FloatingConsole class and its registration (そのまま)
class FloatingConsole {
    constructor() {
        this.element = document.createElement('div');
        this.element.id = 'floating-console';
        this.titleBar = this.createTitleBar();
        this.contentContainer = this.createContentContainer();

        this.element.appendChild(this.titleBar);
        this.element.appendChild(this.contentContainer);

        document.body.appendChild(this.element);

        this.dragging = false;
        this.prevX = 0;
        this.prevY = 0;

        this.setupStyles();
        this.addEventListeners();
        this.addMenuButton();
    }

    setupStyles() {
        Object.assign(this.element.style, {
            position: 'fixed',
            bottom: '10px',
            right: '10px',
            width: '300px',
            maxHeight: '600px',
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            color: 'white',
            borderRadius: '5px',
            zIndex: '1000',
            display: 'none', // Consider starting visible for debugging
            boxSizing: 'border-box',
            overflow: 'hidden',
            resize: 'both',
        });

        // Ensure the content container allows for scrolling overflow content
        Object.assign(this.contentContainer.style, {
            overflowY: 'auto',
            maxHeight: '565px', // Adjust based on titleBar height to prevent overflow
        });
        this.adjustContentContainerSize();
    }
    adjustContentContainerSize() {
        // Calculate available height for content container
        const titleBarHeight = this.titleBar.offsetHeight;
        const consoleHeight = this.element.offsetHeight;
        const availableHeight = consoleHeight - titleBarHeight;

        // Update content container's maxHeight to fill available space
        this.contentContainer.style.maxHeight = `${availableHeight}px`;
    }
    createTitleBar() {
        const bar = document.createElement('div');
        bar.textContent = 'Console';
        Object.assign(bar.style, {
            padding: '5px',
            cursor: 'move',
            backgroundColor: '#333',
            borderTopLeftRadius: '5px',
            borderTopRightRadius: '5px',
            userSelect: 'none',
        });
        return bar;
    }

    createContentContainer() {
        const container = document.createElement('div');
        return container;
    }

    addEventListeners() {
        this.titleBar.addEventListener('mousedown', (e) => {
            this.dragging = true;
            this.prevX = e.clientX;
            this.prevY = e.clientY;
            if (!this.element.style.left || !this.element.style.top) {
                const rect = this.element.getBoundingClientRect();
                this.element.style.right = ''; 
                this.element.style.bottom = '';
                this.element.style.left = `${rect.left}px`;
                this.element.style.top = `${rect.top}px`;
            }
        });

        document.addEventListener('mousemove', (e) => {
            if (!this.dragging) return;
            const dx = e.clientX - this.prevX;
            const dy = e.clientY - this.prevY;
            const { style } = this.element;
            style.left = `${parseInt(style.left || 0, 10) + dx}px`;
            style.top = `${parseInt(style.top || 0, 10) + dy}px`;
            this.prevX = e.clientX;
            this.prevY = e.clientY;
        });

        document.addEventListener('mouseup', () => {
            this.dragging = false;
            this.adjustContentContainerSize();
        });
    }

    addMenuButton() {
        const menu = document.querySelector(".comfy-menu");
        const consoleToggleButton = document.createElement("button");
        consoleToggleButton.textContent = "Toggle Console";
        consoleToggleButton.onclick = () => {
            if (floatingConsole.isVisible()) {
                floatingConsole.hide();
                consoleToggleButton.textContent = "Show Console";
            } else {
                floatingConsole.show();
                consoleToggleButton.textContent = "Hide Console";
            }
        }
        menu.append(consoleToggleButton);
    }

    show() { this.element.style.display = 'block'; }
    hide() { this.element.style.display = 'none'; }
    isVisible() { return this.element.style.display !== 'none'; }
    log(message) {
        const msgElement = document.createElement('div');
        msgElement.textContent = message;
        this.contentContainer.appendChild(msgElement);
        this.contentContainer.scrollTop = this.contentContainer.scrollHeight;
    }
    clear() { this.contentContainer.innerHTML = ''; }
}
const floatingConsole = new FloatingConsole();
app.registerExtension({
    name: "consoleOutput",
    init() {
        api.addEventListener('console_output', (event) => {
            floatingConsole.log(event.detail.message);
        });
    }
});
