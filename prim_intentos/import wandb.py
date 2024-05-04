import tensorflow as tf

def get_tensorflow_version():
    try:
        return tf.__version__
    except Exception as e:
        return f"Error: {e}"

version = get_tensorflow_version()
print(version)
