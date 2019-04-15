def adjust_fd(self, type_function):
    if type_function == "mu":
        self.FD_function_use = "mu"
        self.m = 0
    else:
        self.FD_function_use = "FD"
