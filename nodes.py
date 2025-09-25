import os
from pathlib import Path
import json
import folder_paths
from comfy import model_management
import comfy.utils
from .aura_sr import AuraSR
from .utils import *

import torch
import gc
from weakref import WeakValueDictionary

from aiohttp import web
from server import PromptServer

# Ensure Aura-SR folder is registered (keeps original behaviour)
if "aura-sr" not in folder_paths.folder_names_and_paths:
    aurasr_folders = [p for p in os.listdir(folder_paths.models_dir) if os.path.isdir(os.path.join(folder_paths.models_dir, p)) and (p.lower() == "aura-sr" or p.lower() == "aurasr" or p.lower() == "aura_sr")]
    aurasr_fullpath = os.path.join(folder_paths.models_dir, aurasr_folders[0]) if len(aurasr_folders) > 0 else os.path.join(folder_paths.models_dir, "Aura-SR")
    if not os.path.isdir(aurasr_fullpath):
        os.mkdir(aurasr_fullpath)

    folder_paths.folder_names_and_paths["aura-sr"] = ([aurasr_fullpath], folder_paths.supported_pt_extensions)
else:
    aurasr_fullpath = folder_paths.folder_names_and_paths["aura-sr"][0][0]
    folder_paths.folder_names_and_paths.pop('aura-sr', None)
    folder_paths.folder_names_and_paths["aura-sr"] = ([aurasr_fullpath], folder_paths.supported_pt_extensions)

# A weak-value cache keyed by model_name. Using weakrefs ensures that if no hard
# references remain to an AuraSRUpscaler instance, it can be garbage-collected.
MODEL_CACHE = WeakValueDictionary()


def get_config(model_path):
    configs = [f for f in Path(aurasr_fullpath).rglob('*') if f.is_file() and f.name.lower().endswith(".json")]
    # picking rules by priority (exit immediately when picked):
    # 0 - if a config file is at the same location of the model AND has the same name (without ext)
    # 1 - if a config file is at the same location of the model AND is named 'config' (without ext)
    # 2 - if a config file is at aurasr_fullpath and is named 'config' (without ext)
    #   -- notify user for potential invalid config.json when case #2
    rule = 0
    while (rule < 3):
        for c in configs:
            if rule == 0 and c.parent == Path(model_path).parent and c.stem.lower() == Path(model_path).stem.lower():
                return json.loads(c.read_text())
            if rule == 1 and c.parent == Path(model_path).parent and c.stem.lower() == "config":
                return json.loads(c.read_text())
            if rule == 2 and str(c.parent) == aurasr_fullpath and c.stem.lower() == "config":
                print(f"\n[AuraSR-ComfyUI] WARNING:\n\tCould not find a config named 'config.json'/modelname.json for model: '\\{c.parent.name}\\{Path(model_path).parent.name}\\{Path(model_path).name}'")
                print(f"\tUsing '\\{c.parent.name}\\{c.name}' instead.")
                print("\tIf this configuration is not intended for this model then it can cause errors or quality loss in the output!!\n")
                return json.loads(c.read_text())
        rule += 1
    return None


def get_model_from_cache(model_name):
    """Return a cached AuraSRUpscaler (if available and loaded) or None.
    Removes stale/unloaded entries from cache.
    """
    cl = MODEL_CACHE.get(model_name)
    if cl is None:
        return None
    if not getattr(cl, 'loaded', False):
        # stale entry - ensure it's unloaded and removed
        try:
            cl.unload()
        except Exception:
            pass
        MODEL_CACHE.pop(model_name, None)
        return None
    return cl


class AuraSRUpscaler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model_name": (folder_paths.get_filename_list("aura-sr"),),
                             "image": ("IMAGE",),
                             "mode": (["4x", "4x_overlapped_checkboard", "4x_overlapped_constant"],),
                             "reapply_transparency": ("BOOLEAN", {"default": True}),
                             "tile_batch_size": ("INT", {"default": 8, "min": 1, "max": 32}),
                             "device": (["default", "cpu"],),
                             "offload_to_cpu": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "transparency_mask": ("MASK",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"

    CATEGORY = "AuraSR"
    
    def __init__(self):
        self.loaded = False
        self.model_name = ""
        self.aura_sr = None
        self.upscaling_factor = 4
        self.device_warned = False
        self.config = None
        self.device = "cpu"
    
    def _clear_module_weights(self, module):
        # helper to free parameters/buffers on a module
        try:
            for p in list(module.parameters()):
                del p
        except Exception:
            pass
        try:
            for b in list(module.buffers()):
                del b
        except Exception:
            pass

    def unload(self):
        """Attempt to free GPU memory and remove references to the model so it can be GC'd.
        """
        try:
            if self.aura_sr is not None:
                # if upsampler exists, try to move to cpu before deleting
                upsampler = getattr(self.aura_sr, 'upsampler', None)
                if upsampler is not None:
                    try:
                        upsampler.to('cpu')
                    except Exception:
                        pass
                    # try to clear parameters/buffers
                    try:
                        self._clear_module_weights(upsampler)
                    except Exception:
                        pass
                # drop aura_sr reference
                self.aura_sr = None
                del self.aura_sr
        except Exception:
            pass

        # reset metadata
        self.loaded = False
        self.model_name = ""
        self.upscaling_factor = 4
        self.config = None
        self.device = "cpu"

        # remove from cache if present
        try:
            MODEL_CACHE.pop(self.model_name, None)
        except Exception:
            pass
            
        # Explicit garbage collection
        # encourage python GC
        try:
            gc.collect()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        # Clear PyTorch CPU caching allocator (added in PyTorch 2.0+)
        if hasattr(torch, 'memory') and hasattr(torch.memory, 'empty_cache'):
            try:
                torch.memory.empty_cache()
            except:
                pass
        
        
    
    def load(self, model_name, device):
        model_path = folder_paths.get_full_path("aura-sr", model_name)
        self.config = get_config(model_path)
        if self.config is None:
            return
        
        try:
            self.upscaling_factor = int(self.config.get("image_size", 256) / self.config.get("input_image_size", 64))
        except Exception:
            print(f"[AuraSR-ComfyUI] Failed to calculate {model_name}'s upscaling factor. Defaulting to 4.")
            self.upscaling_factor = 4
        
        checkpoint = comfy.utils.load_torch_file(model_path, safe_load=True)
        
        # create AuraSR and load weights
        self.aura_sr = AuraSR(config=self.config, device=device)
        # ensure no gradients are tracked
        try:
            self.aura_sr.upsampler.load_state_dict(checkpoint, strict=True)
            #self.aura_sr.upsampler.eval()
            #for p in self.aura_sr.upsampler.parameters():
            #    p.requires_grad = False
        except Exception as e:
            print(f"[AuraSR-ComfyUI] Failed to load checkpoint for {model_name}: {e}")
            # cleanup
            self.aura_sr = None
            return
        
        self.loaded = True
        self.model_name = model_name
        self.device = device

        # store a hard reference in cache so other nodes can reuse the loaded model
        try:
            MODEL_CACHE[model_name] = self
        except Exception:
            pass
    
    def load_from_memory(self, cl, device):
        # reuse the AuraSR instance from another cached class
        self.loaded = True
        self.model_name = cl.model_name
        self.aura_sr = cl.aura_sr
        self.upscaling_factor = cl.upscaling_factor
        self.device_warned = cl.device_warned
        self.config = cl.config
        # move model if devices differ
        if device != cl.device and self.aura_sr is not None:
            try:
                self.aura_sr.upsampler.to(device)
            except Exception:
                pass
            cl.device = device
        self.device = device
    
    def main(self, model_name, image, mode, reapply_transparency, tile_batch_size, device, offload_to_cpu, transparency_mask=None):
        # set device
        torch_device = model_management.get_torch_device()
        if model_management.directml_enabled:
            if device == "default" and not self.device_warned:
                print("[AuraSR-ComfyUI] Cannot run AuraSR on DirectML device. Using CPU instead (this will be VERY SLOW!)")
                self.device_warned = True
            device = "cpu"
        else:
            device = torch_device if device == "default" else "cpu"
            device = device if str(device).lower() != "cpu" else "cpu"

        # load/unload model
        class_in_memory = get_model_from_cache(model_name)
        if (not self.loaded) or (self.model_name != model_name):
            if class_in_memory is None:
                # ensure current instance is clean
                try:
                    self.unload()
                except Exception:
                    pass
                self.load(model_name, device)
            else:
                # reuse existing loaded model
                self.load_from_memory(class_in_memory, device)
            if self.config is None:
                print("[AuraSR-ComfyUI] Could not find a config/ModelName .json file! Please download it from the model's HF page and place it according to the instructions (https://github.com/GreenLandisaLie/AuraSR-ComfyUI?tab=readme-ov-file#instructions).\nReturning original image.")
                return (image, )
        else:
            # if device changed, move model
            if self.device != device and self.aura_sr is not None:
                try:
                    self.aura_sr.upsampler.to(device)
                    self.device = device
                    if class_in_memory is not None:
                        class_in_memory.device = device
                except Exception:
                    pass
        
        # iterate through images input
        upscaled_images = []
        
        reapply_transparency = reapply_transparency if len(image) == 1 else False

        any_inference_failed = False
        for tensor_image in image:
            
            # prepare input image and resized_alpha
            input_image, resized_alpha = prepare_input(tensor_image, transparency_mask, reapply_transparency, self.upscaling_factor)
            
            # upscale
            inference_failed = False
            try:
                if mode == "4x":
                    upscaled_image = self.aura_sr.upscale_4x(image=input_image, max_batch_size=tile_batch_size)
                elif mode == "4x_overlapped_checkboard":
                    upscaled_image = self.aura_sr.upscale_4x_overlapped(image=input_image, max_batch_size=tile_batch_size, weight_type='checkboard')
                else:
                    upscaled_image = self.aura_sr.upscale_4x_overlapped(image=input_image, max_batch_size=tile_batch_size, weight_type='constant')
            except Exception as e:
                inference_failed = True
                any_inference_failed = True
                print(f"[AuraSR-ComfyUI] Failed to upscale with AuraSR ({e}). Returning original image.")
                upscaled_image = input_image
            
            # apply resized_alpha
            if reapply_transparency and resized_alpha is not None:
                try:
                    upscaled_image = paste_alpha(upscaled_image, resized_alpha)
                except Exception:
                    print("[AuraSR-ComfyUI] Failed to apply alpha layer.")
            
            # back to tensor and add to list
            upscaled_images.append(pil2tensor(upscaled_image))
        
        # create output tensor from list of tensors
        output = torch.cat(upscaled_images, dim=0)
        
        # offload to cpu if requested
        if offload_to_cpu and self.aura_sr is not None:
            try:
                self.aura_sr.upsampler.to('cpu')
                self.device = 'cpu'
                cached = MODEL_CACHE.get(self.model_name)
                if cached is not None:
                    cached.device = 'cpu'
            except Exception:
                pass

        # force unload when inference fails (any of the images failed)
        if any_inference_failed:
            try:
                self.unload()
            except Exception:
                pass

        return (output, )



@PromptServer.instance.routes.post("/aura_sr/purge_cache")
async def api_purge_aura_sr_cache(request):
    try:
        purged = 0
        for key in list(MODEL_CACHE.keys()):
            cl = MODEL_CACHE.pop(key, None)
            if cl is not None:
                try:
                    cl.unload()
                    purged += 1
                except Exception as e:
                    print(f"[AuraSR-ComfyUI] Error unloading {key}: {e}")

        return web.json_response({"success": True, "purged": purged})
    except Exception as e:
        print("[AuraSR-ComfyUI] purge_cache error:", e)
        return web.json_response({"success": False, "error": str(e)}, status=500)



NODE_CLASS_MAPPINGS = {
    "AuraSR.AuraSRUpscaler": AuraSRUpscaler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AuraSR.AuraSRUpscaler": "AuraSR Upscaler"
}
