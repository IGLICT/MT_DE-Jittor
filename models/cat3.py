# import torch
import jittor
def cat3(input, dim=1):
    stack = jittor.cat([input, input, input], dim=dim)
    return stack
