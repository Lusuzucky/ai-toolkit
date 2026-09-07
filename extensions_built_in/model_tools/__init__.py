from toolkit.extension import Extension


class QuantizeModelExtension(Extension):
    uid = "quantize_model"
    name = "Quantize Model"

    @classmethod
    def get_process(cls):
        from jobs.process.QuantizeModelProcess import QuantizeModelProcess
        return QuantizeModelProcess


class CacheVAEExtension(Extension):
    uid = "cache_vae"
    name = "Cache VAE"

    @classmethod
    def get_process(cls):
        from jobs.process.CacheVAEProcess import CacheVAEProcess
        return CacheVAEProcess


class CacheTEExtension(Extension):
    uid = "cache_te"
    name = "Cache Text Encoder"

    @classmethod
    def get_process(cls):
        from jobs.process.CacheTEProcess import CacheTEProcess
        return CacheTEProcess


AI_TOOLKIT_EXTENSIONS = [
    QuantizeModelExtension,
    CacheVAEExtension,
    CacheTEExtension,
]
