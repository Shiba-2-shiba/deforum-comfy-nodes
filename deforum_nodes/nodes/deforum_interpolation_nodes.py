import cv2
import torch
import numpy as np
from PIL import Image # PIL.Image をインポート

from deforum import FilmModel
from deforum.models import DepthModel, RAFT

from comfy import model_management # model_management をインポート
import comfy.utils # comfy.utils をインポート (ProgressBar用)
from ..modules.standalone_cadence import CadenceInterpolator
from ..modules.deforum_comfyui_helpers import tensor2pil, pil2tensor

from ..mapping import gs


class DeforumFILMInterpolationNode:
    def __init__(self):
        self.FILM_temp = []
        self.model = None
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "inter_amount": ("INT", {"default": 2, "min": 1, "max": 10000},),
                     "skip_first": ("BOOLEAN", {"default":True}),
                     "skip_last": ("BOOLEAN", {"default":False}),

                     }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fn"
    display_name = "FILM Interpolation"
    CATEGORY = "deforum/interpolation"
    @classmethod
    def IS_CHANGED(self, *args, **kwargs):
        return float("NaN")

    def interpolate(self, image_tensor, inter_frames, skip_first, skip_last):
        if self.model is None:
            self.model = FilmModel()

        return_frames_tensors = []
        pil_image = tensor2pil(image_tensor.clone().detach())
        np_image = np.array(pil_image.convert("RGB"))
        self.FILM_temp.append(np_image)

        output_tensor = image_tensor.unsqueeze(0) if image_tensor.ndim == 3 else image_tensor.clone()

        if len(self.FILM_temp) == 2:
            with torch.no_grad():
                generated_frames_np_or_pil = self.model.inference(self.FILM_temp[0], self.FILM_temp[1], inter_frames=inter_frames)

            if skip_first and generated_frames_np_or_pil:
                generated_frames_np_or_pil.pop(0)
            if skip_last and generated_frames_np_or_pil:
                generated_frames_np_or_pil.pop(-1)

            for frame_data in generated_frames_np_or_pil:
                if isinstance(frame_data, np.ndarray):
                    frame_pil = Image.fromarray(frame_data)
                elif isinstance(frame_data, Image.Image):
                    frame_pil = frame_data
                else:
                    print(f"[DeforumFILMInterpolationNode] Warning: Unknown frame data type: {type(frame_data)}")
                    continue
                tensor = pil2tensor(frame_pil)[0]
                return_frames_tensors.append(tensor.detach().cpu())

            self.FILM_temp = [self.FILM_temp[1]]

        if return_frames_tensors:
            output_tensor = torch.stack(return_frames_tensors, dim=0)

        return output_tensor


    def fn(self, image, inter_amount, skip_first, skip_last):
        if image.ndim == 3:
            image = image.unsqueeze(0)

        all_interpolated_frames = []

        if image.shape[0] > 1:
            for i in range(image.shape[0]):
                single_image_tensor = image[i]
                interpolated_batch = self.interpolate(single_image_tensor, inter_amount, skip_first, skip_last)
                if interpolated_batch is not None and interpolated_batch.nelement() > 0 :
                    all_interpolated_frames.append(interpolated_batch)

            if all_interpolated_frames:
                ret = torch.cat(all_interpolated_frames, dim=0)
            else:
                ret = image
        else:
            ret = self.interpolate(image[0], inter_amount, skip_first, skip_last)

        return (ret,)

class DeforumSimpleInterpolationNode:
    def __init__(self):
        self.interp_temp = []

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "method": (["DIS Medium", "DIS Fast", "DIS UltraFast", "Farneback Fine", "Normal"],),
                     "inter_amount": ("INT", {"default": 2, "min": 1, "max": 10000},),
                     "skip_first": ("BOOLEAN", {"default":False}),
                     "skip_last": ("BOOLEAN", {"default":False}),
                     },
                "optional":{
                    "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                }
                }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("IMAGES", "LAST_IMAGE")
    FUNCTION = "fn"
    display_name = "Simple Interpolation"
    CATEGORY = "deforum/interpolation"

    @classmethod
    def IS_CHANGED(self, *args, **kwargs):
        return float("NaN")

    def interpolate(self, current_image_tensor, method, inter_frames, skip_first, skip_last):
        return_frames_tensors = []
        pil_image = tensor2pil(current_image_tensor.clone().detach())
        np_image_rgb = np.array(pil_image.convert("RGB"))

        self.interp_temp.append(np_image_rgb)

        output_images_tensor = None
        last_image_tensor = current_image_tensor.unsqueeze(0) if current_image_tensor.ndim == 3 else current_image_tensor.clone()


        if len(self.interp_temp) == 2:
            prev_np_image, current_np_image = self.interp_temp

            if inter_frames >= 1:
                from ..modules.interp import optical_flow_cadence
                generated_frames_pil = optical_flow_cadence(prev_np_image, current_np_image, inter_frames + 1, method)

                if skip_first and generated_frames_pil:
                    generated_frames_pil.pop(0)
                if skip_last and generated_frames_pil:
                    generated_frames_pil.pop(-1)

                for frame_pil_data in generated_frames_pil:
                    tensor = pil2tensor(frame_pil_data)[0]
                    return_frames_tensors.append(tensor)

            self.interp_temp = [current_np_image]

        if return_frames_tensors:
            output_images_tensor = torch.stack(return_frames_tensors, dim=0)
            if output_images_tensor.nelement() > 0:
                last_image_tensor = output_images_tensor[-1].unsqueeze(0)

        return output_images_tensor, last_image_tensor


    def fn(self, image, method, inter_amount, skip_first, skip_last, deforum_frame_data={}):
        if image.ndim == 3:
            image = image.unsqueeze(0)

        if deforum_frame_data.get("reset", False):
            print("[SimpleInterpolation] Resetting interpolation buffer.")
            self.interp_temp = []

        all_output_batches = []
        processed_last_image = None

        for i in range(image.shape[0]):
            single_image_tensor = image[i]

            interpolated_batch, last_of_this_iter = self.interpolate(single_image_tensor, method, inter_amount, skip_first, skip_last)

            if interpolated_batch is not None and interpolated_batch.nelement() > 0:
                all_output_batches.append(interpolated_batch)

            if last_of_this_iter is not None and last_of_this_iter.nelement() > 0:
                processed_last_image = last_of_this_iter
            elif processed_last_image is None and single_image_tensor.nelement() > 0 :
                processed_last_image = single_image_tensor.unsqueeze(0)


        if not all_output_batches:
            final_output_tensor = None
            if image.nelement() > 0 and processed_last_image is None:
                processed_last_image = image[-1].unsqueeze(0)
        else:
            final_output_tensor = torch.cat(all_output_batches, dim=0)
            if final_output_tensor.nelement() > 0 and processed_last_image is None:
                 processed_last_image = final_output_tensor[-1].unsqueeze(0)


        return (final_output_tensor, processed_last_image)


class DeforumCadenceNode:
    def __init__(self):
        self.interpolator = None
        self.logger = None
        self.vram_state = "high"
        self.skip_return = False

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "first_image": ("IMAGE",),
                     "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                     "depth_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                     "preview": ("BOOLEAN", {"default":False}),
                     },
                "optional":
                    {"hybrid_images": ("IMAGE",),}
                }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    FUNCTION = "fn"
    display_name = "Cadence Interpolation"
    CATEGORY = "deforum/interpolation"

    @classmethod
    def IS_CHANGED(self, *args, **kwargs):
        return float("NaN")

    def interpolate(self, current_keyframe_tensor, first_image_tensor_opt, deforum_frame_data, depth_strength, preview=False, dry_run=False, hybrid_images_tensor_opt=None):
        self.skip_return = False

        args = deforum_frame_data.get("args")
        anim_args = deforum_frame_data.get("anim_args")

        if args is None or anim_args is None:
            print("[DeforumCadenceNode.interpolate] Error: 'args' or 'anim_args' not found in deforum_frame_data.")
            return []

        current_keyframe_np = None
        if not dry_run:
            if current_keyframe_tensor is None or current_keyframe_tensor.nelement() == 0:
                print("[DeforumCadenceNode.interpolate] Error: current_keyframe_tensor is empty or None for interpolation.")
                return []

            img_to_convert = current_keyframe_tensor
            if img_to_convert.ndim == 4 and img_to_convert.shape[0] == 1:
                img_to_convert = img_to_convert.squeeze(0)

            pil_image = tensor2pil(img_to_convert.clone().detach())
            current_keyframe_np = np.array(pil_image.convert("RGB")).astype(np.uint8)
            current_keyframe_np = cv2.normalize(current_keyframe_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

        predict_depths = (anim_args.animation_mode == '3D' and anim_args.use_depth_warping) or anim_args.save_depth_maps
        predict_depths = predict_depths or (hasattr(anim_args, 'hybrid_composite') and anim_args.hybrid_composite and hasattr(anim_args, 'hybrid_comp_mask_type') and anim_args.hybrid_comp_mask_type in ['Depth', 'Video Depth'])


        if "depth_model" not in gs.deforum_models or gs.deforum_depth_algo != anim_args.depth_algorithm:
            if "depth_model" in gs.deforum_models and gs.deforum_models["depth_model"] is not None:
                try: gs.deforum_models["depth_model"].to("cpu"); del gs.deforum_models["depth_model"]; torch.cuda.empty_cache()
                except Exception as e: print(f"Error releasing depth model: {e}")
            gs.deforum_models["depth_model"] = None
            if predict_depths:
                device = model_management.get_torch_device()
                midas_path = getattr(anim_args, 'midas_path', 'models/other')
                gs.deforum_models["depth_model"] = DepthModel(midas_path, device,
                                              keep_in_vram=(self.vram_state == 'high'),
                                              depth_algorithm=anim_args.depth_algorithm, Width=args.width,
                                              Height=args.height, midas_weight=anim_args.midas_weight)
                gs.deforum_depth_algo = anim_args.depth_algorithm
            if hasattr(anim_args, 'hybrid_composite') and anim_args.hybrid_composite != 'None' and hasattr(anim_args, 'hybrid_comp_mask_type') and anim_args.hybrid_comp_mask_type == 'Depth':
                anim_args.save_depth_maps = True

        if gs.deforum_models.get("depth_model") is not None and not predict_depths:
            try: gs.deforum_models["depth_model"].to("cpu"); del gs.deforum_models["depth_model"]; torch.cuda.empty_cache()
            except Exception as e: print(f"Error releasing depth model (no predict): {e}")
            gs.deforum_models["depth_model"] = None

        if gs.deforum_models.get("depth_model") is not None:
            gs.deforum_models["depth_model"].to(model_management.get_torch_device())

        if "raft_model" not in gs.deforum_models or gs.deforum_models.get("raft_model") is None :
            raft_path = getattr(anim_args, 'raft_path', 'models/raft')
            print(f"[DeforumCadenceNode] INFO: Attempting to initialize RAFT model.")
            print(f"[DeforumCadenceNode] INFO: Configured raft_path: '{raft_path}'")

            try:
                print(f"[DeforumCadenceNode] INFO: Initializing RAFT model with RAFT().")
                new_raft_model = RAFT()
                gs.deforum_models["raft_model"] = new_raft_model
                print(f"[DeforumCadenceNode] INFO: RAFT() class instance created.")
                print(f"[DeforumCadenceNode] INFO: If '{raft_path}' is required for loading specific model weights,")
                print(f"[DeforumCadenceNode] INFO: the RAFT class must handle this internally (e.g., use a default path that can be overridden by '{raft_path}')")
                print(f"[DeforumCadenceNode] INFO: or provide a separate method (e.g., model.load_weights('{raft_path}')).")

            except Exception as e_raft:
                print(f"[DeforumCadenceNode] ERROR: Failed to initialize RAFT model using RAFT(). Error: {e_raft}")
                print(f"[DeforumCadenceNode] ERROR: Please check the 'deforum.models.RAFT' class definition for the correct initialization procedure, especially if a custom model path ('{raft_path}') is needed.")
                raise

            if gs.deforum_models.get("raft_model"):
                raft_instance = gs.deforum_models["raft_model"]
                device_to_use = model_management.get_torch_device()
                moved_to_device = False
                try:
                    if hasattr(raft_instance, "to") and callable(getattr(raft_instance, "to")):
                        print(f"[DeforumCadenceNode] INFO: Attempting to move RAFT object directly to device: {device_to_use}")
                        raft_instance.to(device_to_use)
                        moved_to_device = True
                        print(f"[DeforumCadenceNode] INFO: RAFT object successfully moved using its own .to() method.")
                    elif hasattr(raft_instance, "model") and hasattr(raft_instance.model, "to") and callable(getattr(raft_instance.model, "to")):
                        print(f"[DeforumCadenceNode] INFO: Attempting to move inner '.model' attribute of RAFT object to device: {device_to_use}")
                        raft_instance.model.to(device_to_use)
                        moved_to_device = True
                        print(f"[DeforumCadenceNode] INFO: RAFT's inner '.model' attribute successfully moved to device.")
                    elif hasattr(raft_instance, "net") and hasattr(raft_instance.net, "to") and callable(getattr(raft_instance.net, "to")):
                        print(f"[DeforumCadenceNode] INFO: Attempting to move inner '.net' attribute of RAFT object to device: {device_to_use}")
                        raft_instance.net.to(device_to_use)
                        moved_to_device = True
                        print(f"[DeforumCadenceNode] INFO: RAFT's inner '.net' attribute successfully moved to device.")
                    
                    if not moved_to_device:
                        print(f"[DeforumCadenceNode] WARNING: RAFT object (and its common inner attributes like '.model' or '.net') does not have a callable '.to()' method. Device placement might be handled internally by the RAFT class or may require a different approach for this specific implementation.")
                    
                except Exception as e_device_move:
                    print(f"[DeforumCadenceNode] ERROR: Exception while trying to move RAFT model to device {device_to_use}. Error: {e_device_move}")
            else:
                print(f"[DeforumCadenceNode] WARNING: RAFT model was not successfully assigned to gs.deforum_models after initialization attempt, skipping device placement.")


        if deforum_frame_data.get("reset") or not self.interpolator:
            self.interpolator = CadenceInterpolator()
            if deforum_frame_data.get("reset") and self.interpolator is not None:
                 self.interpolator.turbo_prev_image = None
                 self.interpolator.turbo_next_image = None
                 self.interpolator.turbo_prev_frame_idx = -1
                 self.interpolator.turbo_next_frame_idx = -1


        self.interpolator.turbo_prev_image, self.interpolator.turbo_prev_frame_idx = \
            self.interpolator.turbo_next_image, self.interpolator.turbo_next_frame_idx
        self.interpolator.turbo_next_image, self.interpolator.turbo_next_frame_idx = \
            current_keyframe_np, deforum_frame_data.get("frame_idx", 0)


        if self.interpolator.turbo_prev_image is None and \
           first_image_tensor_opt is not None and first_image_tensor_opt.nelement() > 0:

            img_to_convert_first = first_image_tensor_opt
            if img_to_convert_first.ndim == 4 and img_to_convert_first.shape[0] == 1:
                img_to_convert_first = img_to_convert_first.squeeze(0)

            pil_first_image = tensor2pil(img_to_convert_first.clone().detach())
            np_first_image = np.array(pil_first_image.convert("RGB")).astype(np.uint8)
            np_first_image = cv2.normalize(np_first_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

            self.interpolator.turbo_prev_image = np_first_image
            self.interpolator.turbo_prev_frame_idx = 0

            if self.interpolator.turbo_next_frame_idx == 0 and anim_args.diffusion_cadence > 0:
                 self.interpolator.turbo_next_frame_idx = anim_args.diffusion_cadence
                 if "frame_idx" in deforum_frame_data:
                    deforum_frame_data["frame_idx"] = anim_args.diffusion_cadence
                 self.skip_return = True

        hybrid_provider_np_list = None
        if hybrid_images_tensor_opt is not None and hybrid_images_tensor_opt.nelement() > 0:
            hybrid_provider_np_list = []
            current_hybrid_tensor = hybrid_images_tensor_opt
            if current_hybrid_tensor.ndim == 3:
                current_hybrid_tensor = current_hybrid_tensor.unsqueeze(0)

            for i in range(current_hybrid_tensor.shape[0]):
                img_to_convert_hybrid = current_hybrid_tensor[i]
                pil_hybrid = tensor2pil(img_to_convert_hybrid.clone().detach())
                np_hybrid = np.array(pil_hybrid.convert("RGB")).astype(np.uint8)
                np_hybrid = cv2.normalize(np_hybrid, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
                hybrid_provider_np_list.append(np_hybrid)

            if hybrid_provider_np_list and (len(hybrid_provider_np_list) -1 > anim_args.diffusion_cadence):
                print(f"[DeforumCadenceNode] Warning: Hybrid frames ({len(hybrid_provider_np_list)}) might not align with diffusion cadence ({anim_args.diffusion_cadence}).")

        interpolated_frames_np = []
        if not dry_run and self.interpolator.turbo_prev_image is not None and self.interpolator.turbo_next_image is not None:
            if not self.logger and preview:
                self.logger = comfy.utils.ProgressBar(anim_args.diffusion_cadence)

            root_obj = deforum_frame_data.get("root", None)
            keys_obj = deforum_frame_data.get("keys", None)
            if root_obj is None or keys_obj is None:
                print("[DeforumCadenceNode.interpolate] Error: 'root' or 'keys' not found in deforum_frame_data.")
                return []

            with torch.no_grad():
                interpolated_frames_np = self.interpolator.new_standalone_cadence(
                    args, anim_args,
                    root_obj, keys_obj,
                    self.interpolator.turbo_next_frame_idx,
                    gs.deforum_models.get("depth_model"),
                    gs.deforum_models.get("raft_model"),
                    depth_strength,
                    self.logger if preview else None,
                    hybrid_provider=hybrid_provider_np_list
                )
        elif dry_run:
            return []
        else:
            return []
        return interpolated_frames_np


    def fn(self, image, first_image, deforum_frame_data, depth_strength, preview, hybrid_images=None):
        if image is None or image.nelement() == 0:
            if deforum_frame_data.get("reset"):
                 _ = self.interpolate(None, None, deforum_frame_data, depth_strength, dry_run=True, hybrid_images_tensor_opt=None)
            return (None, None,)

        if image.ndim == 3:
            image = image.unsqueeze(0)

        first_image_to_pass = None
        if first_image is not None and first_image.nelement() > 0:
            if first_image.ndim == 3:
                first_image_to_pass = first_image
            elif first_image.ndim == 4 and first_image.shape[0] > 0 :
                first_image_to_pass = first_image[0]

        hybrid_images_to_pass = None
        if hybrid_images is not None and hybrid_images.nelement() > 0:
            hybrid_images_to_pass = hybrid_images

        device = model_management.get_torch_device()
        free_memory = model_management.get_free_memory(device)
        if free_memory < 4 * (1024**3): # 4GB
            print("[DeforumCadenceNode.fn] Low current VRAM, offloading all models.")
            model_management.unload_all_models()
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        all_interpolated_np_frames = []
        for i in range(image.shape[0]):
            current_keyframe_tensor = image[i]

            np_frames_list = self.interpolate(current_keyframe_tensor,
                                              first_image_to_pass if i == 0 else None,
                                              deforum_frame_data,
                                              depth_strength,
                                              preview=preview,
                                              dry_run=False,
                                              hybrid_images_tensor_opt=hybrid_images_to_pass)

            if np_frames_list:
                all_interpolated_np_frames.extend(np_frames_list)

        output_images_tensor = None
        last_frame_tensor = None

        if all_interpolated_np_frames:
            try:
                tensor_list = [torch.from_numpy(frame.astype(np.float32) / 255.0) for frame in all_interpolated_np_frames if frame is not None and frame.size > 0]
                if tensor_list:
                    output_images_tensor = torch.stack(tensor_list, dim=0)
                    if output_images_tensor.nelement() > 0:
                        last_frame_tensor = output_images_tensor[-1].unsqueeze(0)
                    else:
                        print("[DeforumCadenceNode.fn] Warning: Stacked tensor from interpolated frames is empty.")
                else:
                     print("[DeforumCadenceNode.fn] No valid NumPy frames to convert to tensor.")

            except Exception as e:
                print(f"[DeforumCadenceNode.fn] Error converting/stacking NumPy frames to PyTorch tensor: {e}")
                if image.nelement() > 0: last_frame_tensor = image[-1].unsqueeze(0)
                return (None, last_frame_tensor)
        else:
            if image.nelement() > 0:
                last_frame_tensor = image[-1].unsqueeze(0)


        if hasattr(self, 'skip_return') and self.skip_return:
            return (None, last_frame_tensor)
        else:
            return (output_images_tensor, last_frame_tensor)
