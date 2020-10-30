from torch import nn

class ProgressiveBaseModel(nn.Module):

    def __init__(self, start_img_size, transition_channels):
        
        super().__init__()
        self.transition_channels = transition_channels
        self.current_img_size = start_img_size
        self.transition_value = 1.0
        self.transition_step = 0

    def extend(self):
        self.transition_value = 0.0
        self.transition_step += 1
        self.current_img_size *= 2

    def increase_transition_value(self, increase_number):
        self.transition_value += increase_number
        self.transition_value = min(self.transition_value, 1.0)

    def state_dict(self):
        return {
            "transition_step" : self.transition_step,
            "transition_value" : self.transition_value,
            "parameters" : super().state_dict()
        }

    def load_state_dict(self, ckpt, **kwargs):
        for _ in range(ckpt["transition_step"]):
            self.extend()
        self.transition_value = ckpt["transition_value"]

        super().load_state_dict(ckpt["parameters"], **kwargs)