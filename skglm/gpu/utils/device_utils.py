from numba import cuda
from numba.cuda.cudadrv import enums


# modified version of code in
# https://stackoverflow.com/questions/62457151/access-gpu-hardware-specifications-in-python # noqa
def get_device_properties():
    device = cuda.get_current_device()

    device_props_name = [name.replace("CU_DEVICE_ATTRIBUTE_", "")
                         for name in dir(enums)
                         if name.startswith("CU_DEVICE_ATTRIBUTE_")]

    device_props_value = [getattr(device, prop) for prop in device_props_name]

    return dict(zip(device_props_name, device_props_value))
