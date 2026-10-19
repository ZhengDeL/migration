"""Reuse one frame's original image metadata across its DriveLM questions."""
import os


def install_metadata_cache():
    from llamafactory.data.mm_plugin import Qwen3VLPlugin

    original = Qwen3VLPlugin._get_mm_token_metadata
    if getattr(original, '_drivelm_one_frame_cache', False):
        return

    def cached(self, images, videos, audios, processor):
        # Keep the original path for other modalities and in-memory image objects.
        if len(images) != 6 or videos or audios or not all(isinstance(item, str) for item in images):
            return original(self, images, videos, audios, processor)
        files = []
        for name in images:
            status = os.stat(name)
            files.append((name, status.st_size, status.st_mtime_ns))
        key = (tuple(files), getattr(processor, 'image_max_pixels', None),
               getattr(processor, 'image_min_pixels', None))
        previous = getattr(self, '_drivelm_metadata_frame', None)
        if previous is not None and previous[0] is processor and previous[1] == key:
            return previous[2]
        value = original(self, images, videos, audios, processor)
        # Store the unmodified native return value. The training collator still
        # calls the original _get_mm_inputs for its actual batch pixels.
        self._drivelm_metadata_frame = (processor, key, value)
        return value

    cached._drivelm_one_frame_cache = True
    Qwen3VLPlugin._get_mm_token_metadata = cached

