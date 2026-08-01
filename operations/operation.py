import torch
import random


def parameter_to_list(param):
    tensor = param.data
    if tensor.numel() > 0:
        return [round(value.item(), 3) for value in tensor]
    else:
        return []


class Operation(torch.nn.Module):
    def __init__(self, name):
        super(Operation, self).__init__()
        self.name = name
        self.params = torch.nn.ParameterDict()
        self.param_ranges = {}
        self.non_diff_params = {}
        self.non_diff_param_ranges = {}

    def randomize_params(self):
        with torch.no_grad():
            for key, value in self.params.items():
                new_value = []
                for i in range(0, len(value.data)):
                    new_value.append(random.uniform(self.param_ranges[key][0], self.param_ranges[key][1]))
                self.set_params({key: new_value})

    def set_params(self, params):
        for key, value in params.items():
            if key in self.params:
                import torch
                if isinstance(value, (int, float)):
                    value = [value]
                elif isinstance(value, torch.Tensor):
                    if value.dim() == 0:
                        value = [value.item()]
                    else:
                        value = value.tolist()
                elif not isinstance(value, (list, tuple)):
                    try:
                        value = list(value)
                    except TypeError:
                        value = [value]
                param_len = self.params[key].numel()
                if len(value) == param_len:
                    if self.params[key].dim() == 0:
                        self.params[key].data.fill_(value[0])
                    else:
                        for i in range(0, len(value)):
                            self.params[key].data[i] = value[i]
                else:
                    raise Exception(f"Expected parameter {key} to have length {len(self.params[key].data)} but got length {len(value)} instead")

    def clamp_params(self):
        for key, _ in self.params.items():
            self.params[key].data.clamp_(self.param_ranges[key][0], self.param_ranges[key][1])

    def apply_non_diff_param_mutation(self):
        return False

    def __str__(self):
        params_string = "{"
        for key, parameter in self.params.items():
            values = parameter_to_list(parameter)
            values_str = ", ".join([str(value) for value in values])
            params_string += f"'{key}': [{values_str}], "
        params_string = params_string.rstrip(", ")
        params_string += "}"
        return "'{}', {}".format(self.name, params_string)