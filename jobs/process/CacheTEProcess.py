from collections import OrderedDict
from jobs.process.BaseProcess import BaseProcess
from toolkit.tools.cache_te import run_cache_te
from toolkit.print import print_acc


class CacheTEProcess(BaseProcess):
    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)
        self.device = self.get_conf("device", None)

    def run(self):
        super().run()
        config_path = self.get_conf("config_path", None)
        if config_path:
            run_cache_te(config_path, device=self.device)
        else:
            cfg = {
                "job": "extension",
                "config": {
                    "process": [self.config]
                }
            }
            run_cache_te(cfg, device=self.device)
