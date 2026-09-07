from collections import OrderedDict
from jobs.process.BaseProcess import BaseProcess
from toolkit.tools.quantize_model import quantize_state_dict_to_float8, DEFAULT_EXCLUDE_PATTERNS
from toolkit.print import print_acc


class QuantizeModelProcess(BaseProcess):
    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)
        self.input_path = self.get_conf("input_path", self.get_conf("model.name_or_path", None))
        if self.input_path is None:
            raise ValueError("QuantizeModelProcess requires 'input_path' or 'model.name_or_path'")
        self.output_path = self.get_conf("output_path", required=True)
        self.qtype = self.get_conf("qtype", "float8")
        self.exclude_patterns = self.get_conf("exclude_patterns", DEFAULT_EXCLUDE_PATTERNS)
        self.device = self.get_conf("device", "cpu")

    def run(self):
        super().run()
        print_acc(f"Running QuantizeModelProcess: {self.input_path} -> {self.output_path} ({self.qtype})")
        quantize_state_dict_to_float8(
            input_path=self.input_path,
            output_path=self.output_path,
            exclude_patterns=self.exclude_patterns,
            device=self.device,
        )
