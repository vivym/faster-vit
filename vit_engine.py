import ctypes
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import tensorrt as trt
from pydantic import BaseModel
from safetensors import safe_open

TRT_PLUGIN_PATH = Path(__file__).parent / "3rdparty" / "FasterTransformer" / "build" / "lib" / "libvit_plugin.so"

weight_name_suffix_mapping = {
    ".layer_norm1.bias": "/LayerNorm_0/bias",
    ".layer_norm1.weight": "/LayerNorm_0/scale",
    ".layer_norm2.bias": "/LayerNorm_2/bias",
    ".layer_norm2.weight": "/LayerNorm_2/scale",
    ".mlp.fc1.bias": "/MlpBlock_3/Dense_0/bias",
    ".mlp.fc1.weight": "/MlpBlock_3/Dense_0/kernel",
    ".mlp.fc2.bias": "/MlpBlock_3/Dense_1/bias",
    ".mlp.fc2.weight": "/MlpBlock_3/Dense_1/kernel",
    ".self_attn.projection.bias": "/MultiHeadDotProductAttention_1/out/bias",
    ".self_attn.projection.weight": "/MultiHeadDotProductAttention_1/out/kernel",
}


class ViTConfig(BaseModel):
    patch_size: int = 14
    hidden_size: int = 1408
    intermediate_size: int = 6144
    num_attention_heads: int = 16
    num_hidden_layers: int = 39
    use_fp16: bool = True
    image_size: int = 224
    num_channels: int = 3
    max_batch_size: int = 32
    with_cls_token: bool = True

    architectures: Optional[List[str]] = None
    attention_dropout: float = 0.0
    dropout: float = 0.0
    hidden_act: str = "gelu"
    initializer_factor: float = 1.0
    initializer_range: float = 1e-10
    layer_norm_eps: float = 1e-6
    model_type: str = "vit"
    projection_dim: int = 512
    qkv_bias: bool = True
    torch_dtype: str = "float16"
    transformers_version: Optional[str] = None


def build_plugin_field_collection(config: ViTConfig, model_path: Path):
    field_type = trt.PluginFieldType.FLOAT16 if config.use_fp16 else trt.PluginFieldType.FLOAT32
    arr_type = np.float16 if config.use_fp16 else np.float32

    value_holder = {
        "max_batch": np.array([config.max_batch_size]).astype(np.int32),
        "img_size": np.array([config.image_size]).astype(np.int32),
        "patch_size": np.array([config.patch_size]).astype(np.int32),
        "in_chans": np.array([config.num_channels]).astype(np.int32),
        "embed_dim": np.array([config.hidden_size]).astype(np.int32),
        "num_heads": np.array([config.num_attention_heads]).astype(np.int32),
        "inter_size": np.array([config.intermediate_size]).astype(np.int32),
        "layer_num": np.array([config.num_hidden_layers]).astype(np.int32),
        "with_cls_token": np.array([config.with_cls_token]).astype(np.int32),
    }

    with safe_open(model_path, framework="np") as f:
        for key in f.keys():
            if key == "embeddings.class_embedding":
                new_key = "cls"
                assert config.with_cls_token
            elif key.startswith("embeddings.patch_embedding."):
                suffix = key.split(".")[-1]
                if suffix == "weight":
                    suffix = "kernel"
                new_key = "embedding/" + suffix
            elif key == "embeddings.position_embedding":
                new_key = "Transformer/posembed_input/pos_embedding"
            elif key == "post_layernorm.bias":
                new_key = "Transformer/encoder_norm/bias"
            elif key == "post_layernorm.weight":
                new_key = "Transformer/encoder_norm/scale"
            elif key.startswith("encoder.layers."):
                layer_id, *suffix = key[len("encoder.layers."):].split(".")
                suffix = "." + ".".join(suffix)
                if suffix in weight_name_suffix_mapping:
                    new_key = f"Transformer/encoderblock_{layer_id}{weight_name_suffix_mapping[suffix]}"
                else:
                    assert "self_attn.qkv" in key
                    qkv = f.get_tensor(key).astype(arr_type)
                    q, k, v = np.split(qkv, 3, axis=0)
                    suffix = key.split(".")[-1]
                    if suffix == "weight":
                        suffix = "kernel"
                    value_holder[f"Transformer/encoderblock_{layer_id}/MultiHeadDotProductAttention_1/query/{suffix}"] = q
                    value_holder[f"Transformer/encoderblock_{layer_id}/MultiHeadDotProductAttention_1/key/{suffix}"] = k
                    value_holder[f"Transformer/encoderblock_{layer_id}/MultiHeadDotProductAttention_1/value/{suffix}"] = v
                    continue
            else:
                raise RuntimeError(f"Unknown key: {key}")

            print(key, "->", new_key)
            value_holder[new_key] = f.get_tensor(key).astype(arr_type)

    collection = []
    for key, value in value_holder.items():
        if value.dtype == np.int32:
            collection.append(trt.PluginField(key, value, trt.PluginFieldType.INT32))
        else:
            collection.append(trt.PluginField(key, value, field_type))

    return trt.PluginFieldCollection(collection), value_holder


class ViTEngine:
    def __init__(
        self,
        config: ViTConfig,
        engine,
        value_holder: Dict[str, np.ndarray],
    ):
        self.config = config
        self.engine = engine
        self.value_holder = value_holder

    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        # TODO: Load model by name
        model_dir = Path(model_name_or_path)
        config_path = model_dir / "config.json"
        model_path = model_dir / "model.safetensors"

        if not model_dir.exists() or not config_path.exists() or not model_path.exists():
            raise RuntimeError(f"Invalid model: {model_dir}")

        config = ViTConfig.parse_file(config_path)
        print("config", config)

        if not TRT_PLUGIN_PATH.exists():
            raise FileNotFoundError(f"Could not find {TRT_PLUGIN_PATH}")

        handle = ctypes.CDLL(str(TRT_PLUGIN_PATH.resolve()), mode=ctypes.RTLD_GLOBAL)
        if not handle:
            raise RuntimeError(f"Could not load {TRT_PLUGIN_PATH}")

        logger = trt.Logger(trt.Logger.VERBOSE)
        trt.init_libnvinfer_plugins(logger, "")

        plg_registry = trt.get_plugin_registry()
        plg_creator = plg_registry.get_plugin_creator("CustomVisionTransformerPlugin", "1", "")

        trt_dtype = trt.float16 if config.use_fp16 else trt.float32
        explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        with trt.Builder(logger) as builder, builder.create_network(explicit_batch_flag) as network, builder.create_builder_config() as builder_config:
            builder_config.max_workspace_size = 8 << 30
            if config.use_fp16:
                builder_config.set_flag(trt.BuilderFlag.FP16)
                builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)

            # Create the network
            input_tensor = network.add_input(
                name="input_img",
                dtype=trt_dtype,
                shape=(-1, config.num_channels, config.image_size, config.image_size),
            )

            # Specify profiles
            profile = builder.create_optimization_profile()
            min_shape = (1, config.num_channels, config.image_size, config.image_size)
            max_shape = (config.max_batch_size, config.num_channels, config.image_size, config.image_size)
            profile.set_shape("input_img", min=min_shape, opt=min_shape, max=max_shape)
            builder_config.add_optimization_profile(profile)

            pfc, value_holder = build_plugin_field_collection(config, model_path=model_path)

            fn = plg_creator.create_plugin("vision_transformer", pfc)
            inputs = [input_tensor]
            vit = network.add_plugin_v2(inputs, fn)

            output_tensor = vit.get_output(0)
            output_tensor.name = "visiont_transformer_output"

            if config.use_fp16:
                vit.precision = trt.float16
                vit.set_output_type(0, trt.float16)
            network.mark_output(output_tensor)

            engine = builder.build_engine(network, builder_config)

        return cls(config, engine, value_holder)

    def inference(self, images: torch.Tensor):
        images = images.to("cuda", non_blocking=True)

        with self.engine.create_execution_context() as context:
            context.active_optimization_profile = 0

            stream = torch.cuda.Stream()

            input_shape = (
                self.config.max_batch_size,
                self.config.num_channels,
                self.config.image_size,
                self.config.image_size,
            )
            context.set_binding_shape(0, input_shape)
            output_shape = tuple(context.get_binding_shape(1))
            print(output_shape)

            # Copy input h2d
            d_inputs = [images]
            d_output = torch.empty(output_shape, dtype=torch.float32, device=images.device)

            # warm up
            for _ in range(10):
                context.execute_async_v2(
                    [d_inp.data_ptr() for d_inp in d_inputs] + [d_output.data_ptr()],
                    stream.cuda_stream,
                )

            #ignore the last fc layer
            torch.cuda.synchronize()
            import time
            op_end = time.time()
            for _ in range(100):
                context.execute_async_v2([
                    d_inp.data_ptr() for d_inp in d_inputs] + [d_output.data_ptr()],
                    stream.cuda_stream,
                )
            stream.synchronize()

            torch.cuda.synchronize()
            print("plugin time : ", (time.time() - op_end) / 100 * 1000.0, "ms")

        return d_output.cpu().numpy()


def main():
    engine = ViTEngine.from_pretrained("./weights/blip2-vit-hf")
    images = torch.randn(32, 3, 224, 224, device="cuda", dtype=torch.float16)
    ft_output = engine.inference(images)

    from transformers import Blip2VisionModel
    model = Blip2VisionModel.from_pretrained(
        "./weights/blip2-vit-hf", torch_dtype=torch.float16
    ).to("cuda")
    model.eval()

    for _ in range(10):
        with torch.no_grad():
            hf_output = model(pixel_values=images).last_hidden_state

    torch.cuda.synchronize()
    import time
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            hf_output = model(pixel_values=images).last_hidden_state
    torch.cuda.synchronize()
    print("hf time : ", (time.time() - start_time) / 100 * 1000.0, "ms")

    hf_output = hf_output.cpu().numpy()
    diff = np.abs(ft_output - hf_output)
    print(diff.max(), diff.mean())


if __name__ == "__main__":
    main()
