import base64
import gc
import os
import shutil
from io import BytesIO
from aiohttp import web
import hashlib
# import pprint # ログ出力の整形用にインポート (必要に応じてコメント解除)

import server # ComfyUIのサーバーインスタンス
import cv2
import numpy as np
import torch
from PIL import Image

import folder_paths # ComfyUIのディレクトリ管理
from ..modules.deforum_comfyui_helpers import tensor2pil, pil2tensor, find_next_index, pil_image_to_base64, tensor_to_webp_base64

video_extensions = ['webm', 'mp4', 'mkv', 'gif']

# moviepy と scipy はrequirements.txtに追加するか、ユーザーにインストールを促す必要があります
from moviepy import ImageSequenceClip, AudioFileClip # moviepy.editorからインポート # moviepy.editorの代わりに直接クラスをインポート
from scipy.io.wavfile import write as wav_write
import tempfile


def save_to_file(data, filepath: str): # moviepy用オーディオデータ保存ヘルパー
    if data.num_channels > 1:
        audio_data_reshaped = data.audio_data.reshape((-1, data.num_channels))
    else:
        audio_data_reshaped = data.audio_data
    wav_write(filepath, data.sample_rate, audio_data_reshaped.astype(np.int16))
    return True

class DeforumLoadVideo:
    def __init__(self):
        self.video_path = None
        self.cap = None
        self.current_frame = -1

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1] in video_extensions):
                    files.append(f)
        return {"required": {
                    "video": (sorted(files),),
                    "reset": ("BOOLEAN", {"default": False},),
                    "iterative": ("BOOLEAN", {"default": True},),
                    "start_frame": ("INT", {"default": 0, "min": 0, "max": 1000000},),
                    "return_frames": ("INT", {"default": 1, "min": 1, "max": 1000000},),
        }}

    CATEGORY = "deforum/video"
    display_name = "Load Video (Deforum)"
    RETURN_TYPES = ("IMAGE","INT","INT")
    RETURN_NAMES = ("IMAGE","FRAME_IDX","MAX_FRAMES")
    FUNCTION = "load_video_frame"

    def load_video_frame(self, video, reset, iterative, start_frame, return_frames):
        video_path = folder_paths.get_annotated_filepath(video)
        max_vid_frames = 0
        frames = []

        if self.cap is None or self.video_path != video_path or reset:
            if self.cap: self.cap.release()
            self.cap = cv2.VideoCapture(video_path)
            self.current_frame = -1
            self.video_path = video_path
            if self.cap.isOpened() and start_frame > 0 and not iterative:
                 self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                 self.current_frame = start_frame -1


        elif self.cap.get(cv2.CAP_PROP_POS_FRAMES) >= self.cap.get(cv2.CAP_PROP_FRAME_COUNT) and iterative:
             self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
             self.current_frame = -1


        if not iterative and self.cap.get(cv2.CAP_PROP_POS_FRAMES) != start_frame :
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.current_frame = start_frame - 1


        max_vid_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for _ in range(return_frames):
            success, frame_cv = self.cap.read()

            if not success:
                if iterative:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame = -1
                    success, frame_cv = self.cap.read()
                    if not success: break
                else: break

            if success:
                self.current_frame += 1
                frame_cv = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_cv)
                frame_tensor = pil2tensor(frame_pil)
                frames.append(frame_tensor[0])
            else: break

        output_frame_tensor = None
        if frames:
            output_frame_tensor = torch.stack(frames)

        return (output_frame_tensor, self.current_frame, max_vid_frames)


    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, video, **kwargs):
        if not folder_paths.exists_annotated_filepath(video):
            return "Invalid video file: {}".format(video)
        return True

temp_dir_base = os.path.join(tempfile.gettempdir(), "deforum_comfyui_video_cache")
os.makedirs(temp_dir_base, exist_ok=True)
web_temp_dir = tempfile.mkdtemp(prefix="deforum_web_", dir=temp_dir_base)
_hex_dig_object = hashlib.md5(web_temp_dir.encode())
GLOBAL_HEX_DIG = _hex_dig_object.hexdigest()
GLOBAL_ENDPOINT_PREFIX = f"/tmp/deforum_web/{GLOBAL_HEX_DIG}"


if not hasattr(server.PromptServer.instance, '_deforum_video_routes_registered_v3'):
    print(f"Deforum Video endpoint registering at {GLOBAL_ENDPOINT_PREFIX}/{{filename:.+}} for files from {web_temp_dir}")
    @server.PromptServer.instance.routes.get(f"{GLOBAL_ENDPOINT_PREFIX}/{{filename:.+}}")
    async def serve_temp_file_deforum_video_v3(request):
        filename = request.match_info['filename']
        if '..' in filename:
            return web.Response(status=400, text="Invalid file path (contains '..').")
        file_path = os.path.join(web_temp_dir, filename)
        normalized_file_path = os.path.normpath(file_path)
        if not normalized_file_path.startswith(os.path.normpath(web_temp_dir)):
            return web.Response(status=403, text="Forbidden: Invalid file path (attempts to escape base directory).")
        if os.path.isfile(normalized_file_path):
            return web.FileResponse(normalized_file_path)
        else:
            return web.Response(status=404, text=f"File not found: {filename}")
    server.PromptServer.instance._deforum_video_routes_registered_v3 = True

class DeforumVideoSaveNode:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.images = [] # Stores actual full paths to frame image files in self.instance_temp_dir

        self.instance_temp_dir = tempfile.mkdtemp(prefix="deforum_frames_", dir=temp_dir_base)
        self.instance_web_subdir_name = os.path.basename(self.instance_temp_dir)
        self.instance_web_path = os.path.join(web_temp_dir, self.instance_web_subdir_name)
        os.makedirs(self.instance_web_path, exist_ok=True)

        self.web_endpoint_prefix = GLOBAL_ENDPOINT_PREFIX
        self.audio_preview_path = None # Basename of audio file in web_temp_dir
        self.filepath = ""
        self.is_showing_final_preview = False # NEW: Flag to track if preview is for a completed video

    def clear_cache_directory_full(self): # RENAMED from clear_cache_directory
        # Clears all files in instance_web_path
        if hasattr(self, 'instance_web_path') and os.path.exists(self.instance_web_path):
            for item_name in os.listdir(self.instance_web_path):
                item_path = os.path.join(self.instance_web_path, item_name)
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.remove(item_path)
                except Exception as e:
                    print(f'[DeforumVideoSave] Failed to delete preview item {item_path}. Reason: {e}')
        
        # Clears all files in instance_temp_dir (actual frames)
        if hasattr(self, 'instance_temp_dir') and os.path.exists(self.instance_temp_dir):
            for filename in os.listdir(self.instance_temp_dir):
                file_path = os.path.join(self.instance_temp_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path): os.unlink(file_path)
                    elif os.path.isdir(file_path): shutil.rmtree(file_path)
                except Exception as e: print(f'Failed to delete {file_path}. Reason: {e}')
        self.images = []

    # NEW METHOD: To prune directories, keeping specified files
    def prune_directories(self, keep_actual_paths=None, keep_web_basenames=None):
        if keep_actual_paths is None: keep_actual_paths = []
        if keep_web_basenames is None: keep_web_basenames = []

        actual_paths_to_keep_set = set(keep_actual_paths)
        if hasattr(self, 'instance_temp_dir') and os.path.exists(self.instance_temp_dir):
            for filename in os.listdir(self.instance_temp_dir):
                file_path = os.path.join(self.instance_temp_dir, filename)
                if file_path not in actual_paths_to_keep_set:
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path): os.unlink(file_path)
                        elif os.path.isdir(file_path): shutil.rmtree(file_path)
                    except Exception as e: print(f'Failed to delete {file_path} during prune. Reason: {e}')

        web_basenames_to_keep_set = set(keep_web_basenames)
        if hasattr(self, 'instance_web_path') and os.path.exists(self.instance_web_path):
            for item_name in os.listdir(self.instance_web_path):
                if item_name not in web_basenames_to_keep_set:
                    item_path = os.path.join(self.instance_web_path, item_name)
                    try:
                        if os.path.isfile(item_path) or os.path.islink(item_path): os.remove(item_path)
                    except Exception as e: print(f'[DeforumVideoSave] Failed to delete preview item {item_path} during prune. Reason: {e}')


    def __del__(self):
        if hasattr(self, 'instance_temp_dir') and os.path.exists(self.instance_temp_dir):
            try: shutil.rmtree(self.instance_temp_dir)
            except Exception as e: print(f"Error cleaning up DeforumVideoSaveNode temp directory {self.instance_temp_dir}: {e}")
        
        if hasattr(self, 'instance_web_path') and os.path.exists(self.instance_web_path):
            try: shutil.rmtree(self.instance_web_path)
            except Exception as e: print(f"Error cleaning up DeforumVideoSaveNode web preview directory {self.instance_web_path}: {e}")

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("IMAGE",),
                    "filename_prefix": ("STRING",{"default":"Deforum"}),
                    "fps": ("INT", {"default": 24, "min": 1, "max": 10000},),
                    "codec": (["libx265", "libx264", "libvpx-vp9", "libaom-av1", "mpeg4", "libvpx"],),
                    "pixel_format": (["yuv420p", "yuv422p", "yuv444p", "rgb24", "rgba", "nv12", "nv21"],),
                    "format": (["mp4", "mov", "gif", "webm", "avi"],),
                    "quality": ("INT", {"default": 10, "min": 1, "max": 10},),
                    "dump_by": (["max_frames", "per_N_frames"],),
                    "dump_every": ("INT", {"default": 0, "min": 0, "max": 4096},),
                    "dump_now": ("BOOLEAN", {"default": False},),
                    "skip_save": ("BOOLEAN", {"default": False},),
                    "skip_return": ("BOOLEAN", {"default": True},),
                    "enable_preview": ("BOOLEAN", {"default": True},),
                    "restore_frames_for_preview": ("BOOLEAN", {"default": False}, {"label_on": "Restore All Cached Frames", "label_off": "Show Only New Frames"}),
                    "clear_cache": ("BOOLEAN", {"default": False},),
                    },
                "optional": {
                    "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                    "audio": ("AUDIO",),
                    "waveform_image": ("IMAGE",),
                }, "hidden": {}
               }

    RETURN_TYPES = ("IMAGE","STRING",)
    RETURN_NAMES = ("IMAGES","VIDEOPATH",)
    OUTPUT_NODE = True
    FUNCTION = "fn"
    CATEGORY = "deforum/video"

    def add_image(self, image_tensor_single_frame):
        os.makedirs(self.instance_temp_dir, exist_ok=True)
        # Use a consistent naming scheme for frames based on current cache size to avoid collisions after pruning
        frame_basename = f"frame_{len(self.images):06d}.png"
        actual_frame_path = os.path.join(self.instance_temp_dir, frame_basename)

        if image_tensor_single_frame.ndim == 4 and image_tensor_single_frame.shape[0] == 1:
            pil_image = tensor2pil(image_tensor_single_frame.squeeze(0))
        elif image_tensor_single_frame.ndim == 3:
            pil_image = tensor2pil(image_tensor_single_frame)
        else:
            print(f"[DeforumVideoSave] Warning: Unexpected tensor shape in add_image: {image_tensor_single_frame.shape}")
            return None

        if pil_image.mode == 'RGBA': pil_image.save(actual_frame_path, "PNG")
        else: pil_image.convert('RGB').save(actual_frame_path, "PNG")
        # self.images.append(actual_frame_path) # Moved appending to self.images to the main fn logic

        os.makedirs(self.instance_web_path, exist_ok=True)
        symlink_path_in_web_subdir = os.path.join(self.instance_web_path, frame_basename)

        if os.path.exists(symlink_path_in_web_subdir) or os.path.islink(symlink_path_in_web_subdir):
            try: os.remove(symlink_path_in_web_subdir)
            except OSError as e: print(f"[DeforumVideoSave] Warning: Could not remove existing preview file/symlink: {e}")
        
        try:
            os.symlink(actual_frame_path, symlink_path_in_web_subdir)
        except OSError as e:
            try:
                shutil.copy2(actual_frame_path, symlink_path_in_web_subdir)
            except Exception as e_copy:
                print(f"[DeforumVideoSave] Error: Could not copy frame for preview: {e_copy}")
        
        return actual_frame_path, frame_basename # Return both actual path and basename

    def fn(self, image, filename_prefix, fps, codec, pixel_format,
           format, quality, dump_by, dump_every, dump_now, skip_save, skip_return, enable_preview,
           restore_frames_for_preview, clear_cache,
           deforum_frame_data=None, audio=None, waveform_image=None):

        current_video_format_ext = format
        if deforum_frame_data is None: deforum_frame_data = {}
        
        ui_frame_urls_for_output = []
        audio_path_for_output = None
        # fps_for_output = fps # fps is an input, can be used directly for UI
        counter_for_output = 0
        returned_frames_tensor = None
        self.filepath = ""

        num_input_frames = 0
        if image is not None and isinstance(image, torch.Tensor) and image.nelement() > 0:
            num_input_frames = image.shape[0]

        # 1. Handle clear_cache command
        if clear_cache:
            self.clear_cache_directory_full()
            # self.images is already cleared by clear_cache_directory_full
            self.is_showing_final_preview = False

        # 2. Handle reset event
        is_reset_event = deforum_frame_data.get("reset", False)
        if is_reset_event and not clear_cache:
            self.clear_cache_directory_full()
            self.is_showing_final_preview = False
        
        # 3. Process incoming image tensor
        newly_added_actual_frame_paths = [] # Only for this call, if not dumping
        
        if num_input_frames > 0:
            if self.is_showing_final_preview:
                # New frames are arriving, clear the preview of the previously dumped video
                self.clear_cache_directory_full() # This also clears self.images
                self.is_showing_final_preview = False

            for i in range(num_input_frames):
                # add_image creates frame_XXXXXX.png based on current len(self.images)
                # so self.images must be appended to *before* calling add_image if names are to be sequential *within this batch*
                # However, add_image now returns basename too.
                # Let's ensure self.images is the true list of *stored* actual file paths.
                # The frame_basename calculation in add_image uses len(self.images) *before* the new image is conceptually added to the list for naming.
                
                # Create a temporary name based on potential new length for add_image internal logic
                # This is a bit tricky. Let add_image determine the name internally based on its own logic of what's there.
                # The self.images list will be the source of truth for what's *currently cached*.
                
                # Corrected add_image call and self.images update:
                # Frame name in add_image is based on the number of images *currently* in self.images.
                # So, we pass the tensor, add_image saves it and returns path. Then we add path to self.images.
                temp_actual_path, _ = self.add_image(image[i]) # image[i] is [H,W,C]
                if temp_actual_path:
                    self.images.append(temp_actual_path) # Add to our list of active frames
                    newly_added_actual_frame_paths.append(temp_actual_path)
            
            if not skip_return:
                 returned_frames_tensor = image[-1].unsqueeze(0)


        # 4. Determine dump_video_flag
        dump_video_flag = False
        anim_args = deforum_frame_data.get("anim_args")
        max_total_frames_for_dump_config = 0
        if anim_args and hasattr(anim_args, 'max_frames') and anim_args.max_frames > 0:
            max_total_frames_for_dump_config = anim_args.max_frames
        
        if dump_now:
            dump_video_flag = True
        elif self.images:
            if dump_by == "max_frames" and max_total_frames_for_dump_config > 0:
                if len(self.images) >= max_total_frames_for_dump_config: dump_video_flag = True
            elif dump_by == "per_N_frames" and dump_every > 0:
                if len(self.images) % dump_every == 0: dump_video_flag = True
        
        if is_reset_event and self.images: dump_video_flag = True


        # 5. If dumping video
        if dump_video_flag and self.images:
            video_frames_being_dumped_actual_paths = list(self.images) # These are full paths
            video_frame_basenames_for_web_preview = [os.path.basename(p) for p in video_frames_being_dumped_actual_paths]

            if not skip_save:
                first_img_pil = Image.open(video_frames_being_dumped_actual_paths[0])
                temp_h, temp_w = first_img_pil.height, first_img_pil.width
                first_img_pil.close()
                _folder, _base, _, _, _ = folder_paths.get_save_image_path(filename_prefix, self.output_dir, temp_h, temp_w)
                video_counter = find_next_index(_folder, _base, current_video_format_ext)
                # self.save_video internally uses self.images; ensure it uses video_frames_being_dumped_actual_paths
                # For simplicity, we'll pass the paths to save_video or ensure self.images is correct at time of call.
                # Current save_video uses self.images. So this is fine.
                self.filepath = self.save_video(_folder, _base, video_counter, fps, audio, codec, current_video_format_ext, quality, pixel_format)

            if enable_preview:
                for basename in video_frame_basenames_for_web_preview:
                    url_filename_part = f"{self.instance_web_subdir_name}/{basename}"
                    ui_frame_urls_for_output.append(f"{self.web_endpoint_prefix}/{url_filename_part}")
                counter_for_output = len(video_frame_basenames_for_web_preview)

                if audio:
                    if hasattr(self, 'audio_preview_path') and self.audio_preview_path and os.path.exists(os.path.join(web_temp_dir, self.audio_preview_path)):
                        try: os.remove(os.path.join(web_temp_dir, self.audio_preview_path))
                        except OSError: pass # Ignore error if file is already gone or in use
                    self.audio_preview_path = self.encode_audio_for_preview(audio, counter_for_output, fps)
                    if self.audio_preview_path:
                        audio_path_for_output = (f"{self.web_endpoint_prefix}/{self.audio_preview_path}",)

            # Prune directories, keeping only the files for the dumped video
            self.prune_directories(keep_actual_paths=video_frames_being_dumped_actual_paths,
                                   keep_web_basenames=video_frame_basenames_for_web_preview)
            # Update self.images to only contain paths of the dumped video frames (which are now the only ones physically present)
            self.images = video_frames_being_dumped_actual_paths
            self.is_showing_final_preview = True
            newly_added_actual_frame_paths = [] # Consumed by dump

        # 6. Else (not dumping or no images to dump)
        else:
            self.is_showing_final_preview = False
            paths_for_ui_urls_actual = [] # List of actual file paths for UI preview

            if enable_preview:
                if restore_frames_for_preview:
                    paths_for_ui_urls_actual = self.images
                elif newly_added_actual_frame_paths: # Only new frames if not restoring all
                    paths_for_ui_urls_actual = newly_added_actual_frame_paths
                elif self.images : # Fallback if no specific instruction but images exist
                    paths_for_ui_urls_actual = self.images
                
                if paths_for_ui_urls_actual:
                    for actual_fp in paths_for_ui_urls_actual:
                        frame_basename = os.path.basename(actual_fp)
                        url_filename_part = f"{self.instance_web_subdir_name}/{frame_basename}"
                        ui_frame_urls_for_output.append(f"{self.web_endpoint_prefix}/{url_filename_part}")
                    counter_for_output = len(paths_for_ui_urls_actual)

                    if audio:
                        if hasattr(self, 'audio_preview_path') and self.audio_preview_path and os.path.exists(os.path.join(web_temp_dir, self.audio_preview_path)):
                            try: os.remove(os.path.join(web_temp_dir, self.audio_preview_path))
                            except OSError: pass
                        self.audio_preview_path = self.encode_audio_for_preview(audio, counter_for_output, fps)
                        if self.audio_preview_path:
                            audio_path_for_output = (f"{self.web_endpoint_prefix}/{self.audio_preview_path}",)
            else: # Preview disabled
                ui_frame_urls_for_output = []
                counter_for_output = len(self.images) # Still report number of internally cached images


        # Prepare UI preview data structure for output
        ui_preview_data = {
            "counter": (counter_for_output,),
            "should_dump": (dump_video_flag,),
            "frames": ui_frame_urls_for_output if enable_preview else [],
            "fps": (fps,) # Send current FPS
        }
        if enable_preview and audio_path_for_output:
            ui_preview_data["audio"] = audio_path_for_output
        if enable_preview and waveform_image is not None and waveform_image.nelement() > 0:
            ui_preview_data["waveform"] = (tensor_to_webp_base64(waveform_image),)
        
        # Handle returned_frames_tensor (last frame for passthrough or output)
        if skip_return:
            returned_frames_tensor = None
        elif returned_frames_tensor is None and self.images: # If not set by new inputs, but cache exists (e.g. after dump)
            # self.images now correctly reflects the frames that are being previewed
            last_cached_pil = Image.open(self.images[-1]).convert("RGB")
            returned_frames_tensor = pil2tensor(last_cached_pil) # pil2tensor should return [1,H,W,C]

        # Cleanup (optional)
        if image is not None and isinstance(image, torch.Tensor): del image
        if waveform_image is not None and isinstance(waveform_image, torch.Tensor): del waveform_image
        # gc.collect()
        # if torch.cuda.is_available(): torch.cuda.empty_cache()

        return {"ui": ui_preview_data, "result": (returned_frames_tensor, self.filepath,)}

    def encode_audio_for_preview(self, audio_data_obj, frame_count, fps_val):
        default_sample_rate = 44100; audio_data_to_write = None; sample_rate_to_use = default_sample_rate
        
        if audio_data_obj is None or not hasattr(audio_data_obj, 'audio_data') or \
           not hasattr(audio_data_obj, 'sample_rate') or audio_data_obj.audio_data is None:
            duration_in_seconds = max(0.1, frame_count / float(fps_val if fps_val > 0 else 1.0))
            num_samples = int(duration_in_seconds * default_sample_rate)
            audio_data_to_write = np.zeros(num_samples, dtype=np.int16)
        else:
            source_audio_np = audio_data_obj.audio_data.cpu().numpy()
            sample_rate_to_use = audio_data_obj.sample_rate
            num_channels = getattr(audio_data_obj, 'num_channels', 1)

            target_duration_seconds = max(0.1, frame_count / float(fps_val if fps_val > 0 else 1.0))
            required_samples_for_segment = int(target_duration_seconds * sample_rate_to_use)
            
            if num_channels > 1:
                if source_audio_np.shape[0] == num_channels and source_audio_np.shape[1] > num_channels:
                    source_audio_np = source_audio_np.T
                if source_audio_np.ndim > 1 and source_audio_np.shape[1] == num_channels and num_channels > 1:
                     source_audio_np = np.mean(source_audio_np, axis=1)

            current_total_samples = source_audio_np.shape[0]

            if current_total_samples == 0:
                audio_data_to_write = np.zeros(required_samples_for_segment, dtype=np.int16)
            elif current_total_samples >= required_samples_for_segment:
                audio_data_to_write = source_audio_np[:required_samples_for_segment]
            else:
                num_repeats = (required_samples_for_segment + current_total_samples - 1) // current_total_samples
                audio_data_to_write = np.tile(source_audio_np, num_repeats)[:required_samples_for_segment]
            
            if audio_data_to_write.dtype == np.float32 or audio_data_to_write.dtype == np.float64:
                audio_data_to_write = np.clip(audio_data_to_write * 32767, -32768, 32767).astype(np.int16)

        if audio_data_to_write is None or audio_data_to_write.size == 0:
             audio_data_to_write = np.zeros(int(0.1 * sample_rate_to_use), dtype=np.int16)

        temp_audio_basename = None; temp_audio_full_path = ""
        try:
            os.makedirs(web_temp_dir, exist_ok=True)
            with tempfile.NamedTemporaryFile(delete=False, dir=web_temp_dir, suffix='.wav', mode='wb') as tmp_file:
                temp_audio_full_path = tmp_file.name
            
            wav_write(temp_audio_full_path, sample_rate_to_use, audio_data_to_write)
            temp_audio_basename = os.path.basename(temp_audio_full_path)
        except Exception as e:
            print(f"Error encoding audio for preview: {e}")
            if temp_audio_full_path and os.path.exists(temp_audio_full_path):
                 try: os.remove(temp_audio_full_path)
                 except Exception as e_del: print(f"Error deleting temp audio on fail: {e_del}")
            return None
        return temp_audio_basename

    def save_video(self, full_output_folder, filename_base, counter, fps, audio_input_obj,
                   codec_name, container_ext, quality_param, pixel_format_param):
        # This method now uses the passed 'video_frame_paths' instead of self.images directly
        # However, the current call self.save_video(...) still relies on self.images being correct.
        # For this iteration, we assume self.images is correctly set to video_frames_being_dumped_actual_paths before calling.
        output_path = os.path.join(full_output_folder, f"{filename_base}_{counter:05d}.{container_ext}")
        
        loaded_pil_frames = []
        if not self.images: # self.images should be the list of frames to save at this point
            print("[DeforumVideoSave] No frames in cache to save for video.")
            return ""
            
        for frame_path in self.images: # Use self.images which should be set correctly
            try:
                img = Image.open(frame_path)
                if container_ext.lower() == 'gif':
                    if img.mode != 'RGBA' and img.mode != 'P': img = img.convert('RGBA')
                elif img.mode != 'RGB': img = img.convert('RGB')
                loaded_pil_frames.append(img)
            except Exception as e:
                print(f"[DeforumVideoSave] Error loading frame {frame_path} for video: {e}")
                continue

        if not loaded_pil_frames:
            print("[DeforumVideoSave] No valid frames loaded to save video.")
            return ""

        numpy_frames = [np.array(frame) for frame in loaded_pil_frames]
        
        video_clip_obj = None; audio_tmp_file_path = None; final_audio_clip_for_moviepy = None; raw_audio_clip_local = None
        try:
            video_clip_obj = ImageSequenceClip(numpy_frames, fps=fps)
            
            if audio_input_obj and hasattr(audio_input_obj, 'audio_data') and hasattr(audio_input_obj, 'sample_rate') and audio_input_obj.audio_data is not None:
                fd, audio_tmp_file_path = tempfile.mkstemp(suffix='.wav'); os.close(fd)
                save_to_file(audio_input_obj, audio_tmp_file_path)
                
                raw_audio_clip_local = AudioFileClip(audio_tmp_file_path)
                video_duration = video_clip_obj.duration
                
                if raw_audio_clip_local.duration > video_duration:
                    final_audio_clip_for_moviepy = raw_audio_clip_local.subclip(0, video_duration)
                else:
                    final_audio_clip_for_moviepy = raw_audio_clip_local
                
                if final_audio_clip_for_moviepy:
                    video_clip_obj = video_clip_obj.set_audio(final_audio_clip_for_moviepy)

            ffmpeg_params_list = []
            if pixel_format_param: ffmpeg_params_list.extend(['-pix_fmt', pixel_format_param])

            if codec_name in ['libx264', 'libx265']:
                crf_val = round(18 + ( (10 - quality_param) / 9.0 ) * (30 - 18) )
                ffmpeg_params_list.extend(['-crf', str(crf_val)])
            elif codec_name == 'libvpx-vp9':
                crf_val = round(20 + ( (10 - quality_param) / 9.0 ) * (40 - 20) )
                ffmpeg_params_list.extend(['-crf', str(crf_val), '-b:v', '0'])
            elif codec_name == 'libaom-av1':
                crf_val = round(23 + ( (10 - quality_param) / 9.0 ) * (45 - 23) )
                ffmpeg_params_list.extend(['-crf', str(crf_val), '-b:v', '0', '-cpu-used', '4', '-row-mt', '1'])
            elif codec_name == 'mpeg4':
                q_val = round(2 + ( (10 - quality_param) / 9.0 ) * (10 - 2) )
                ffmpeg_params_list.extend(['-qscale:v', str(q_val)])

            num_threads = os.cpu_count() if os.cpu_count() else 1
            
            if container_ext.lower() == 'gif':
                video_clip_obj.write_gif(output_path, fps=fps, logger=None)
            else:
                video_clip_obj.write_videofile(output_path,
                                              codec=codec_name,
                                              audio_codec='aac',
                                              ffmpeg_params=ffmpeg_params_list,
                                              threads=num_threads,
                                              logger=None)
            print(f"[DeforumVideoSave] Video saved successfully: {output_path}")
        except Exception as e:
            print(f"Error during video saving with MoviePy: {e}")
        finally:
            if video_clip_obj: video_clip_obj.close()
            if final_audio_clip_for_moviepy: final_audio_clip_for_moviepy.close()
            if raw_audio_clip_local: raw_audio_clip_local.close()
            if audio_tmp_file_path and os.path.exists(audio_tmp_file_path):
                try: os.remove(audio_tmp_file_path)
                except Exception as e_del: print(f"Warning: Failed to delete temporary audio file {audio_tmp_file_path}: {e_del}")
        return output_path

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

NODE_CLASS_MAPPINGS = {
    "DeforumVideoSaveNode": DeforumVideoSaveNode,
    "DeforumLoadVideo": DeforumLoadVideo
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DeforumVideoSaveNode": "Save Video (Deforum Fixed Preview)",
    "DeforumLoadVideo": "Load Video (Deforum)"
}
