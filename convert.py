import argparse
import os

import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import torch

from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers import BertForSequenceClassification
from transformers import convert_pytorch_checkpoint_to_tf2
from pytorch_transformers import BertForSequenceClassification
# from modeling_unilm import UnilmForLM


def convert_pytorch_checkpoint_to_tf(model: BartForConditionalGeneration, ckpt_dir: str, model_name: str):
    tensors_to_transpose = ("lm_head.weight",
                            "fc1.weight",
                            "fc2.weight",
                            "out_proj.weight",
                            "self_attn.k_proj",
                            "self_attn.v_proj",
                            "self_attn.q_proj",
                            "encoder_attn.k_proj",
                            "encoder_attn.v_proj",
                            "encoder_attn.q_proj",)
    var_map = (
        ("layers.", "layers_"),
        ("shared.weight", "shared"),
        ("embed_tokens.weight", "embed_tokens"),
        ("embed_positions.weight", "embed_positions"),
        (".", "/"),
        ("self_attn_layer_norm/weight", "self_attn_layer_norm/gamma"),
        ("self_attn_layer_norm/bias", "self_attn_layer_norm/beta"),
        ("final_layer_norm/weight", "final_layer_norm/gamma"),
        ("final_layer_norm/bias", "final_layer_norm/beta"),
        ("layernorm_embedding/weight", "layernorm_embedding/gamma"),
        ("layernorm_embedding/bias", "layernorm_embedding/beta"),
        ("encoder_attn_layer_norm/weight", "encoder_attn_layer_norm/gamma"),
        ("encoder_attn_layer_norm/bias", "encoder_attn_layer_norm/beta"),
        ("weight", "kernel"),
    )

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    state_dict = model.state_dict()
    for param_tensor in model.state_dict():
        # 打印 key value字典
        print(param_tensor, '\t', model.state_dict()[param_tensor].size())
    assert 1==2
    def to_tf_var_name(name: str):
        for patt, repl in iter(var_map):
            name = name.replace(patt, repl)
        return "bart/{}".format(name)

    def create_tf_var(tensor: np.ndarray, name: str, session: tf.Session):
        tf_dtype = tf.dtypes.as_dtype(tensor.dtype)
        tf_var = tf.get_variable(dtype=tf_dtype, shape=tensor.shape, name=name, initializer=tf.zeros_initializer())
        session.run(tf.variables_initializer([tf_var]))
        session.run(tf_var)
        return tf_var

    tf.reset_default_graph()
    with tf.Session() as session:
        for var_name in state_dict:
            tf_name = to_tf_var_name(var_name)
            torch_tensor = state_dict[var_name].numpy()
            if any([x in var_name for x in tensors_to_transpose]):
                torch_tensor = torch_tensor.T
            tf_var = create_tf_var(tensor=torch_tensor, name=tf_name, session=session)
            tf.keras.backend.set_value(tf_var, torch_tensor)
            tf_weight = session.run(tf_var)
            print("Successfully created {}: {}".format(tf_name, np.allclose(tf_weight, torch_tensor)))

        saver = tf.train.Saver(tf.trainable_variables())
        # saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables())
        saver.save(session, os.path.join(ckpt_dir, model_name.replace("-", "_") + ".ckpt"))


def main(raw_args=None):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name", type=str, required=True, help="model name e.g. bert-base-uncased")
    # parser.add_argument(
    #     "--cache_dir", type=str, default=None, required=False, help="Directory containing pytorch model"
    # )
    # parser.add_argument("--pytorch_model_path", type=str, required=True, help="/path/to/<pytorch-model-name>.bin")
    # parser.add_argument("--tf_cache_dir", type=str, required=True, help="Directory in which to save tensorflow model")
    # args = parser.parse_args(raw_args)

    # model = UnilmForLM.from_pretrained(
    #     pretrained_model_name_or_path=args.model_name,
    #     state_dict=torch.load(args.pytorch_model_path),
    #     cache_dir=args.cache_dir,
    # )
    model_path = r"C:\Users\刘平生\Desktop\公司业务\query结果\所用模型\bart-12L-zh-char_pt"
    model = BartForConditionalGeneration.from_pretrained(model_path)
    # model = BertForSequenceClassification.from_pretrained("bert-base-chinese")
    tf_cache_dir = "tf_saved"
    model_name = "bart-base-chinese"
    convert_pytorch_checkpoint_to_tf(model=model, ckpt_dir=tf_cache_dir, model_name=model_name)


if __name__ == "__main__":
    main()