import numpy as np
import tensorflow as tf


def compute_rounded_memory(memory_size: int, memory_int: int, memory_base=3) -> tf.Tensor:
    memory = np.zeros((memory_size,))
    # increase the memory by 1 given the previous memory. Every memory cell can be {-1, 0, 1}
    for i in range(memory_size):
        memory[i] = ((memory_int + 1) % memory_base) - 1
        memory_int = memory_int // memory_base
    return tf.convert_to_tensor(memory, dtype=tf.float32)


def decompute_rounded_memory(memory_size: int, memory_vector: tf.Tensor, memory_base=3) -> int:
    memory_int = 0
    for i in range(memory_size):
        memory_int += (memory_vector[0][i] % 3) * (memory_base ** i)
    return memory_int


def one_hot_encode_memory(memory_size: int, memory_int: int, memory_base=3) -> tf.Tensor:
    memory = np.zeros((memory_size,))
    memory[memory_int] = 1
    return tf.convert_to_tensor(memory, dtype=tf.float32)


def one_hot_decode_memory(memory_size=0, memory_vector: tf.Tensor = None, memory_base=3) -> int:
    return tf.argmax(memory_vector, axis=-1).numpy()[0]


def get_encoding_functions(is_one_hot: bool = True) -> tuple[callable, callable]:
    if is_one_hot:
        compute_memory = one_hot_encode_memory
        decompute_memory = one_hot_decode_memory
    else:
        compute_memory = compute_rounded_memory
        decompute_memory = decompute_rounded_memory
    return compute_memory, decompute_memory
