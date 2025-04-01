import cv2
from PIL import Image
import numpy as np
import torch
import pkg_resources # For skimage version check

# Import color-matcher components
try:
    from color_matcher import ColorMatcher
    from color_matcher.normalizer import Normalizer
    color_matcher_available = True
except ImportError:
    print("[Deforum] Warning: color-matcher library not found. Some color coherence modes will be unavailable.")
    ColorMatcher = None
    Normalizer = None
    color_matcher_available = False

# Import skimage for histogram matching (fallback and specific modes)
try:
    from skimage.exposure import match_histograms
    skimage_available = True
except ImportError:
    print("[Deforum] Warning: scikit-image library not found. Fallback color coherence modes (RGB, HSV, LAB) will be unavailable.")
    match_histograms = None
    skimage_available = False

from deforum.generators.deforum_noise_generator import add_noise
# Import the new blend function and other utils from image_utils
from deforum.utils.image_utils import unsharp_mask, compose_mask_with_check, optimized_pixel_diffusion_blend
from ..modules.deforum_comfyui_helpers import tensor2pil, tensor2np, pil2tensor

class DeforumColorMatchNode:

    def __init__(self):
        self.depth_model = None # Keep if needed elsewhere, otherwise remove
        self.algo = ""          # Keep if needed elsewhere, otherwise remove
        self.color_match_sample = None # This will store the BGR NumPy array
        # Initialize ColorMatcher if available
        self.matcher = ColorMatcher() if color_matcher_available else None
        # Check skimage version for match_histograms kwargs
        self.match_histograms_kwargs = {}
        if skimage_available:
            try:
                skimage_version = pkg_resources.get_distribution('scikit-image').version
                is_skimage_v20_or_higher = pkg_resources.parse_version(skimage_version) >= pkg_resources.parse_version('0.20.0')
                self.match_histograms_kwargs = {'channel_axis': -1} if is_skimage_v20_or_higher else {'multichannel': True}
                print(f"[Deforum] scikit-image version {skimage_version} detected. Using kwargs: {self.match_histograms_kwargs}")
            except pkg_resources.DistributionNotFound:
                print("[Deforum] Warning: Could not determine scikit-image version. Using default match_histograms args.")
                self.match_histograms_kwargs = {'multichannel': True} # Safer default
            except Exception as e:
                print(f"[Deforum] Warning: Error checking scikit-image version: {e}. Using default match_histograms args.")
                self.match_histograms_kwargs = {'multichannel': True} # Safer default


    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                        "image": ("IMAGE",),
                        "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                        "color_coherence_alpha": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}), # Alpha for pixel diffusion blend
                        "force_use_sample": ("BOOLEAN", {"default":False}),
                    },
                "optional":
                    {
                        "force_sample_image":("IMAGE",)
                    }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fn"
    display_name = "Color Match (Advanced)" # Renamed slightly
    CATEGORY = "deforum/image"

    def fn(self, image, deforum_frame_data, color_coherence_alpha, force_use_sample, force_sample_image=None):
        if image is None:
            return (None,)

        anim_args = deforum_frame_data.get("anim_args")
        frame_idx = deforum_frame_data.get("frame_idx", 0)
        color_mode = anim_args.color_coherence # e.g., 'LAB', 'RGB', 'HM', 'None'

        # Convert current image tensor (RGB) to NumPy array (BGR)
        image_np_rgb = tensor2np(image)
        if image_np_rgb is None or image_np_rgb.size == 0:
            print("[Deforum] Warning: Input image tensor is empty.")
            return (image,) # Return original if conversion fails
        image_np_bgr = cv2.cvtColor(image_np_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)

        perform_color_match = True
        current_sample_image_np_bgr = None

        # --- Sample Image Handling ---
        if force_use_sample and force_sample_image is not None:
            print(f"[Deforum] ColorMatch: Forcing sample image for frame {frame_idx}.")
            force_sample_np_rgb = tensor2np(force_sample_image)
            if force_sample_np_rgb is not None and force_sample_np_rgb.size > 0:
                self.color_match_sample = cv2.cvtColor(force_sample_np_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
                current_sample_image_np_bgr = self.color_match_sample
            else:
                print("[Deforum] Warning: Forced sample image tensor is empty.")
                perform_color_match = False # Don't match if forced sample is bad
        elif frame_idx == 0 and not force_use_sample:
             print(f"[Deforum] ColorMatch: Frame 0, initializing color sample.")
             self.color_match_sample = image_np_bgr.copy() # Use current frame as first sample
             perform_color_match = False # Don't perform matching on the very first frame unless forced
        elif self.color_match_sample is None:
            print(f"[Deforum] ColorMatch: No color sample available for frame {frame_idx}. Using current frame as sample.")
            self.color_match_sample = image_np_bgr.copy()
            perform_color_match = False # No prior sample to match against
        else:
            # Use the stored sample from the previous frame
            current_sample_image_np_bgr = self.color_match_sample


        matched_image_np_bgr = image_np_bgr # Default to current image if no matching happens

        # --- Perform Color Matching ---
        if perform_color_match and color_mode != 'None' and current_sample_image_np_bgr is not None:
            print(f"[deforum] ColorMatch: Applying {color_mode} coherence for frame {frame_idx}. Alpha: {color_coherence_alpha:.2f}")

            # Ensure sample and image have same dimensions (resize sample if needed) - Added basic check
            if image_np_bgr.shape != current_sample_image_np_bgr.shape:
                 print(f"[Deforum] Warning: Color sample shape {current_sample_image_np_bgr.shape} differs from image shape {image_np_bgr.shape}. Resizing sample.")
                 current_sample_image_np_bgr = cv2.resize(current_sample_image_np_bgr, (image_np_bgr.shape[1], image_np_bgr.shape[0]), interpolation=cv2.INTER_AREA)


            # 1. Optimized Pixel Diffusion Blend (Prepare the reference image)
            # Alpha=1 means 100% sample pixels, Alpha=0 means 0% sample pixels (all current image pixels)
            # The function blends image1 (sample) onto image2 (current) based on alpha
            blended_reference_np_bgr = optimized_pixel_diffusion_blend(
                current_sample_image_np_bgr.astype(np.uint8), # Sample (image1)
                image_np_bgr.astype(np.uint8),               # Current Frame (image2)
                color_coherence_alpha                        # Alpha (proportion of sample pixels)
            )

            # 2. Apply Color Matching method
            try:
                if color_mode in ['HM', 'Reinhard', 'MVGD', 'MKL', 'HM-MVGD-HM', 'HM-MKL-HM']:
                    if self.matcher:
                        # Use color-matcher library
                        matched_image_np_bgr = self.matcher.transfer(src=image_np_bgr.astype(np.uint8),
                                                                      ref=blended_reference_np_bgr.astype(np.uint8),
                                                                      method=color_mode.lower())
                        matched_image_np_bgr = Normalizer(matched_image_np_bgr).uint8_norm()
                    else:
                        print(f"[Deforum] Warning: color-matcher not available, cannot use mode '{color_mode}'. Skipping color match.")
                        matched_image_np_bgr = image_np_bgr # Skip if library missing

                elif color_mode in ['RGB', 'HSV', 'LAB']:
                    if match_histograms and self.match_histograms_kwargs:
                        # Use skimage match_histograms
                        src_img_conv = None
                        ref_img_conv = None
                        target_colorspace = None
                        source_colorspace = cv2.COLOR_BGR2RGB # Default just in case

                        if color_mode == 'RGB':
                            target_colorspace = cv2.COLOR_BGR2RGB
                            source_colorspace = cv2.COLOR_RGB2BGR
                        elif color_mode == 'HSV':
                            target_colorspace = cv2.COLOR_BGR2HSV
                            source_colorspace = cv2.COLOR_HSV2BGR
                        else: # LAB
                            target_colorspace = cv2.COLOR_BGR2LAB
                            source_colorspace = cv2.COLOR_LAB2BGR

                        src_img_conv = cv2.cvtColor(image_np_bgr.astype(np.uint8), target_colorspace)
                        ref_img_conv = cv2.cvtColor(blended_reference_np_bgr.astype(np.uint8), target_colorspace)

                        matched_conv = match_histograms(src_img_conv, ref_img_conv, **self.match_histograms_kwargs)
                        matched_image_np_bgr = cv2.cvtColor(matched_conv.astype(np.uint8), source_colorspace)
                    else:
                        print(f"[Deforum] Warning: scikit-image not available or config error, cannot use mode '{color_mode}'. Skipping color match.")
                        matched_image_np_bgr = image_np_bgr # Skip if library missing
                else:
                    print(f"[Deforum] Warning: Unknown color coherence mode '{color_mode}'. Skipping color match.")
                    matched_image_np_bgr = image_np_bgr # Skip for unknown modes

            except Exception as e:
                 print(f"[Deforum] Error during color matching (mode: {color_mode}): {e}")
                 matched_image_np_bgr = image_np_bgr # Fallback to original on error

        # --- Update sample for next frame ---
        # Store the result *before* grayscale conversion as the sample for the next iteration
        # Make sure to copy!
        if matched_image_np_bgr is not None and matched_image_np_bgr.size > 0:
            self.color_match_sample = matched_image_np_bgr.copy()
        else:
             # If matching failed badly, fall back to using the input as sample to avoid None sample
             self.color_match_sample = image_np_bgr.copy()
             if perform_color_match: # Only reset matched_image if matching was attempted
                 matched_image_np_bgr = image_np_bgr

        # --- Apply Grayscale ---
        if anim_args.color_force_grayscale:
            print(f"[Deforum] ColorMatch: Applying grayscale.")
            if matched_image_np_bgr is not None and matched_image_np_bgr.ndim == 3:
                 gray_image = cv2.cvtColor(matched_image_np_bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                 matched_image_np_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR) # Convert back to 3 channels

        # --- Convert final result back to RGB Tensor ---
        if matched_image_np_bgr is None or matched_image_np_bgr.size == 0:
             print("[Deforum] Warning: Final image before tensor conversion is empty. Returning original input.")
             return (image,) # Return original input if result is bad

        final_image_np_rgb = cv2.cvtColor(matched_image_np_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)
        final_image_pil = Image.fromarray(final_image_np_rgb)
        final_image_tensor = pil2tensor(final_image_pil) # Assumes pil2tensor handles PIL -> Tensor conversion

        return (final_image_tensor,)


# Keep DeforumAddNoiseNode as it was, no changes needed here for color match logic
class DeforumAddNoiseNode:
    @classmethod
    def INPUT_TYPES(cls):
        # ... (original INPUT_TYPES)
        return {"required":
                    {
                        "image": ("IMAGE",),
                        "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                    }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fn"
    display_name = "Add Noise"
    CATEGORY = "deforum/noise"

    def fn(self, image, deforum_frame_data):
        # ... (original fn logic)
        if image is not None:
            keys = deforum_frame_data.get("keys")
            args = deforum_frame_data.get("args")
            anim_args = deforum_frame_data.get("anim_args")
            root = deforum_frame_data.get("root")
            frame_idx = deforum_frame_data.get("frame_idx")
            noise = keys.noise_schedule_series[frame_idx]
            kernel = int(keys.kernel_schedule_series[frame_idx])
            sigma = keys.sigma_schedule_series[frame_idx]
            amount = keys.amount_schedule_series[frame_idx]
            threshold = keys.threshold_schedule_series[frame_idx]
            contrast = keys.contrast_schedule_series[frame_idx]
            if anim_args.use_noise_mask and keys.noise_mask_schedule_series[frame_idx] is not None:
                noise_mask_seq = keys.noise_mask_schedule_series[frame_idx]
            else:
                noise_mask_seq = None
            mask_vals = {}
            noise_mask_vals = {}

            # Check if args width/height exist, provide default if not
            width = getattr(args, 'width', 512)
            height = getattr(args, 'height', 512)
            seed = getattr(args, 'seed', 123)
            use_mask = getattr(args, 'use_mask', False)
            invert_mask = getattr(args, 'invert_mask', False)


            mask_vals['everywhere'] = Image.new('1', (width, height), 1)
            noise_mask_vals['everywhere'] = Image.new('1', (width, height), 1)

            prev_img = tensor2np(image) # Needs BGR
            if prev_img is None: return (image,) # Handle potential None from tensor2np

            # Convert tensor2np result (RGB) to BGR if needed by subsequent functions
            if prev_img.ndim == 3 and prev_img.shape[2] == 3:
                 prev_img = cv2.cvtColor(prev_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
            else: # Handle grayscale or unexpected shapes
                 print("[Deforum] AddNoiseNode: Unexpected image format from tensor2np.")
                 # Attempt grayscale conversion or return original
                 if prev_img.ndim == 2: # Grayscale
                     prev_img = cv2.cvtColor(prev_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                 else:
                      return (image,)


            mask_image = None
            # apply scaling
            contrast_image = (prev_img * contrast).round().astype(np.uint8)
            # anti-blur
            if amount > 0:
                contrast_image = unsharp_mask(contrast_image, (kernel, kernel), sigma, amount, threshold,
                                              mask_image if use_mask else None)
            # apply frame noising
            noise_mask = None # Initialize noise_mask
            if use_mask or anim_args.use_noise_mask:
                # Ensure root object exists and has required attributes safely
                if root and hasattr(root, 'noise_mask'):
                    try:
                        root.noise_mask = compose_mask_with_check(root, args, noise_mask_seq,
                                                                   noise_mask_vals,
                                                                   Image.fromarray(cv2.cvtColor(contrast_image.astype(np.uint8), cv2.COLOR_BGR2RGB)))
                        noise_mask = root.noise_mask
                    except Exception as e:
                        print(f"[Deforum] Error composing noise mask: {e}")
                        noise_mask = None # Fallback to no mask on error
                else:
                     print("[Deforum] AddNoise: 'root' object or 'root.noise_mask' not available for mask composition.")
                     noise_mask = None

            noised_image = add_noise(contrast_image, noise, int(seed), anim_args.noise_type,
                                     (anim_args.perlin_w, anim_args.perlin_h,
                                      anim_args.perlin_octaves,
                                      anim_args.perlin_persistence),
                                     noise_mask, invert_mask) # Pass potentially None noise_mask

            print(f"[deforum] Adding Noise {noise} {anim_args.noise_type}")

            # Convert final BGR numpy result back to RGB PIL/Tensor
            noised_image_rgb = cv2.cvtColor(noised_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
            image = pil2tensor(Image.fromarray(noised_image_rgb))#.detach().cpu() # pil2tensor should handle tensor creation

        return (image,)
