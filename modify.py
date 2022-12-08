from transformers.models.bart.modeling_tf_bart import TFBartMainLayer, shift_tokens_right, TFBartEncoderLayer, \
    TFBartDecoderLayer, TFBartLearnedPositionalEmbedding, _expand_mask, _make_causal_mask

from transformers import TFBartPretrainedModel, TFBartModel, BartConfig
from transformers.modeling_tf_outputs import TFBaseModelOutput, TFSeq2SeqLMOutput, TFSeq2SeqModelOutput
from transformers.modeling_tf_utils import TFCausalLanguageModelingLoss, input_processing, \
    TFSharedEmbeddings, TFWrappedEmbeddings, shape_list
import random
import math
from typing import Optional, Dict
import tensorflow as tf


class TFBartEncoder(tf.keras.layers.Layer):
    config_class = BartConfig
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`TFBartEncoderLayer`.

    Args:
        config: BartConfig
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[TFSharedEmbeddings] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.layerdrop = config.encoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0

        self.embed_tokens = embed_tokens
        self.embed_positions = TFBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        self.layers = [TFBartEncoderLayer(config, name=f"layers.{i}") for i in range(config.encoder_layers)]
        self.layernorm_embedding = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")

    def get_embed_tokens(self):
        return self.embed_tokens

    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    def call(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs,
    ):
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["input_ids"] is not None and inputs["inputs_embeds"] is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif inputs["input_ids"] is not None:
            input_shape = shape_list(inputs["input_ids"])
        elif inputs["inputs_embeds"] is not None:
            input_shape = shape_list(inputs["inputs_embeds"])[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs["inputs_embeds"] is None:
            inputs["inputs_embeds"] = self.embed_tokens(inputs["input_ids"]) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)
        hidden_states = inputs["inputs_embeds"] + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states, training=inputs["training"])

        # check attention mask and invert
        if inputs["attention_mask"] is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(inputs["attention_mask"])
        else:
            attention_mask = None

        encoder_states = () if inputs["output_hidden_states"] else None
        all_attentions = () if inputs["output_attentions"] else None

        # check if head_mask has a correct number of layers specified if desired
        # The tf.debugging asserts are not compliant with XLA then they
        # have to be disabled in other modes than eager.
        if inputs["head_mask"] is not None and tf.executing_eagerly():
            tf.debugging.assert_equal(
                shape_list(inputs["head_mask"])[0],
                len(self.layers),
                message=f"The head_mask should be specified for {len(self.layers)} layers, but it is for {shape_list(inputs['head_mask'])[0]}.",
            )

        # encoder layers
        for idx, encoder_layer in enumerate(self.layers):

            if inputs["output_hidden_states"]:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if inputs["training"] and (dropout_probability < self.layerdrop):  # skip the layer
                continue

            hidden_states, attn = encoder_layer(
                hidden_states,
                attention_mask,
                inputs["head_mask"][idx] if inputs["head_mask"] is not None else None,
            )

            if inputs["output_attentions"]:
                all_attentions += (attn,)

        if inputs["output_hidden_states"]:
            encoder_states = encoder_states + (hidden_states,)

        return hidden_states
        # return TFBaseModelOutput(
        #     last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        # )

class TFBartDecoder(tf.keras.layers.Layer):
    config_class = BartConfig
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`TFBartDecoderLayer`

    Args:
        config: BartConfig
        embed_tokens: output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[TFSharedEmbeddings] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.embed_tokens = embed_tokens
        self.layerdrop = config.decoder_layerdrop
        self.embed_positions = TFBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0
        self.layers = [TFBartDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]
        self.layernorm_embedding = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")

        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def get_embed_tokens(self):
        return self.embed_tokens

    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    def call(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        head_mask=None,
        encoder_head_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs,
    ):
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            encoder_head_mask=encoder_head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["input_ids"] is not None and inputs["inputs_embeds"] is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif inputs["input_ids"] is not None:
            input_shape = shape_list(inputs["input_ids"])
        elif inputs["inputs_embeds"] is not None:
            input_shape = shape_list(inputs["inputs_embeds"])[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = (
            shape_list(inputs["past_key_values"][0][0])[2] if inputs["past_key_values"] is not None else 0
        )

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        if inputs["inputs_embeds"] is None:
            inputs["inputs_embeds"] = self.embed_tokens(inputs["input_ids"]) * self.embed_scale

        hidden_states = inputs["inputs_embeds"]

        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(input_shape, past_key_values_length=past_key_values_length)
        else:
            combined_attention_mask = _expand_mask(
                tf.ones((input_shape[0], input_shape[1] + past_key_values_length)), tgt_len=input_shape[-1]
            )

        if inputs["attention_mask"] is not None:
            combined_attention_mask = combined_attention_mask + _expand_mask(
                inputs["attention_mask"], tgt_len=input_shape[-1]
            )

        if inputs["encoder_hidden_states"] is not None and inputs["encoder_attention_mask"] is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            inputs["encoder_attention_mask"] = _expand_mask(inputs["encoder_attention_mask"], tgt_len=input_shape[-1])

        hidden_states = self.layernorm_embedding(hidden_states + positions)
        hidden_states = self.dropout(hidden_states, training=inputs["training"])

        # decoder layers
        all_hidden_states = () if inputs["output_hidden_states"] else None
        all_self_attns = () if inputs["output_attentions"] else None
        present_key_values = () if inputs["use_cache"] else None

        # check if head_mask has a correct number of layers specified if desired
        # The tf.debugging asserts are not compliant with XLA then they
        # have to be disabled in other modes than eager.
        if inputs["head_mask"] is not None and tf.executing_eagerly():
            tf.debugging.assert_equal(
                shape_list(inputs["head_mask"])[0],
                len(self.layers),
                message=f"The head_mask should be specified for {len(self.layers)} layers, but it is for {shape_list(inputs['head_mask'])[0]}.",
            )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if inputs["output_hidden_states"]:
                all_hidden_states += (hidden_states,)

            dropout_probability = random.uniform(0, 1)

            if inputs["training"] and (dropout_probability < self.layerdrop):
                continue

            past_key_value = inputs["past_key_values"][idx] if inputs["past_key_values"] is not None else None

            hidden_states, layer_self_attn, present_key_value = decoder_layer(
                hidden_states,
                attention_mask=combined_attention_mask,
                encoder_hidden_states=inputs["encoder_hidden_states"],
                encoder_attention_mask=inputs["encoder_attention_mask"],
                layer_head_mask=inputs["head_mask"][idx] if inputs["head_mask"] is not None else None,
                encoder_layer_head_mask=inputs["encoder_head_mask"][idx]
                if inputs["encoder_head_mask"] is not None
                else None,
                past_key_value=past_key_value,
            )

            if inputs["use_cache"]:
                present_key_values += (present_key_value,)

            if inputs["output_attentions"]:
                all_self_attns += (layer_self_attn,)

        if inputs["output_hidden_states"]:
            all_hidden_states += (hidden_states,)

        if inputs["output_attentions"]:
            all_self_attns = list(all_self_attns)

        if inputs["use_cache"]:
            present_key_values = (inputs["encoder_hidden_states"], present_key_values)

        return hidden_states
        # if not inputs["return_dict"]:
        #     return hidden_states, present_key_values, all_hidden_states, all_self_attns
        # else:
        #     return TFBaseModelOutputWithPast(
        #         last_hidden_state=hidden_states,
        #         past_key_values=present_key_values,
        #         hidden_states=all_hidden_states,
        #         attentions=all_self_attns,
        #     )


class TFBartMainLayer(tf.keras.layers.Layer):
    config_class = BartConfig

    def __init__(self, config: BartConfig, load_weight_prefix=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.shared = TFSharedEmbeddings(config.vocab_size, config.d_model, config.pad_token_id, name="model.shared")

        # set tf scope correctly
        if load_weight_prefix is None:
            load_weight_prefix = "model.shared"

        with tf.compat.v1.variable_scope(load_weight_prefix) as shared_abs_scope_name:
            pass

        # Wraps layer to avoid problems with weight restoring and ensuring we're in the correct TF scope.
        embed_tokens = TFWrappedEmbeddings(self.shared, abs_scope_name=shared_abs_scope_name)
        embed_tokens.vocab_size = self.shared.vocab_size
        embed_tokens.hidden_size = self.shared.hidden_size

        self.encoder = TFBartEncoder(config, embed_tokens, name="encoder")
        self.decoder = TFBartDecoder(config, embed_tokens, name="decoder")

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared.weight = new_embeddings
        self.shared.vocab_size = self.shared.weight.shape[0]
        # retrieve correct absolute scope for embed token wrapper
        with tf.compat.v1.variable_scope("model.shared") as shared_abs_scope_name:
            pass
        # Wraps layer to avoid problems with weight restoring and ensuring we're in the correct TF scope.
        embed_tokens = TFWrappedEmbeddings(self.shared, abs_scope_name=shared_abs_scope_name)
        self.encoder.set_embed_tokens(embed_tokens)
        self.decoder.set_embed_tokens(embed_tokens)

    def call(
        self,
        decoder_input_ids=None,
        past_key_values=None,
        encoder_outputs=None,
        decoder_attention_mask=None,
        attention_mask=None,
        input_ids=None,
        head_mask=None,
        decoder_head_mask=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs
    ):
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["decoder_input_ids"] is None and inputs["decoder_inputs_embeds"] is None:
            inputs["use_cache"] = False

        inputs["output_hidden_states"] = (
            inputs["output_hidden_states"]
            if inputs["output_hidden_states"] is not None
            else self.config.output_hidden_states
        )

        if inputs["decoder_input_ids"] is None and inputs["input_ids"] is not None:
            inputs["decoder_input_ids"] = shift_tokens_right(
                inputs["input_ids"], self.config.pad_token_id, self.config.decoder_start_token_id
            )

        if inputs["encoder_outputs"] is None:
            inputs["encoder_outputs"] = self.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                head_mask=inputs["head_mask"],
                inputs_embeds=inputs["inputs_embeds"],
                output_attentions=inputs["output_attentions"],
                output_hidden_states=inputs["output_hidden_states"],
                return_dict=inputs["return_dict"],
                training=inputs["training"],
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a TFBaseModelOutput when return_dict=True
        elif inputs["return_dict"] and not isinstance(inputs["encoder_outputs"], TFBaseModelOutput):
            inputs["encoder_outputs"] = TFBaseModelOutput(
                last_hidden_state=inputs["encoder_outputs"][0],
                hidden_states=inputs["encoder_outputs"][1] if len(inputs["encoder_outputs"]) > 1 else None,
                attentions=inputs["encoder_outputs"][2] if len(inputs["encoder_outputs"]) > 2 else None,
            )
        # If the user passed a TFBaseModelOutput for encoder_outputs, we wrap it in a tuple when return_dict=False
        elif not inputs["return_dict"] and not isinstance(inputs["encoder_outputs"], tuple):
            inputs["encoder_outputs"] = inputs["encoder_outputs"].to_tuple()

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,  # decoder_input_ids
            attention_mask=decoder_attention_mask,  # decoder_attention_mask
            encoder_hidden_states=encoder_outputs,  # encoder_last_hidden_states
            encoder_attention_mask=attention_mask,  # src_attention_mask
            past_key_values=past_key_values,  # decoder_hidden_state
        )
        return decoder_outputs
        # return TFSeq2SeqModelOutput(
        #     last_hidden_state=decoder_outputs.last_hidden_state,
        #     past_key_values=decoder_outputs.past_key_values,
        #     decoder_hidden_states=decoder_outputs.hidden_states,
        #     decoder_attentions=decoder_outputs.attentions,
        #     encoder_last_hidden_state=inputs["encoder_outputs"].last_hidden_state,
        #     encoder_hidden_states=inputs["encoder_outputs"].hidden_states,
        #     encoder_attentions=inputs["encoder_outputs"].attentions,
        # )


class TFBartForConditionalGeneration(TFBartPretrainedModel, TFCausalLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = [
        r"model.encoder.embed_tokens.weight",
        r"model.decoder.embed_tokens.weight",
    ]

    _requires_load_weight_prefix = True

    def __init__(self, config, load_weight_prefix=None, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model = TFBartMainLayer(config, load_weight_prefix=load_weight_prefix, name="model")
        self.use_cache = config.use_cache
        # final_bias_logits is registered as a buffer in pytorch, so not trainable for the the sake of consistency.
        self.final_logits_bias = self.add_weight(
            name="final_logits_bias", shape=[1, config.vocab_size], initializer="zeros", trainable=False
        )

    def get_decoder(self):
        return self.model.decoder

    def get_encoder(self):
        return self.model.encoder

    def get_output_embeddings(self):
        return self.get_input_embeddings()

    def set_output_embeddings(self, value):
        self.set_input_embeddings(value)

    def get_bias(self):
        return {"final_logits_bias": self.final_logits_bias}

    def set_bias(self, value):
        self.final_logits_bias = value["final_logits_bias"]

    def call(
        self,
        decoder_input_ids=None,  # tgt_input_ids
        encoder_outputs=None,  # encoder last_hidden_state
        attention_mask=None,  # src_attention_mask
        past_key_values=None,  # decoder_hidden_states
        **kwargs,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        """
        input_ids = None
        decoder_attention_mask = None
        head_mask = None
        decoder_head_mask = None
        inputs_embeds = None
        decoder_inputs_embeds = None
        use_cache = None
        output_attentions = None
        output_hidden_states = None
        return_dict = None
        labels = None
        training = False
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )

        outputs = self.model(
            decoder_input_ids=decoder_input_ids,  # decoder_input_ids
            past_key_values=past_key_values,  # decoder_hidden_state
            encoder_outputs=encoder_outputs,  # encoder_last_hidden_state
            decoder_attention_mask=decoder_attention_mask,  # None
            attention_mask=attention_mask,  # src_attention_mask
        )
        lm_logits = self.model.shared(outputs[0], mode="linear")
        lm_logits = lm_logits + self.final_logits_bias
        masked_lm_loss = None if inputs["labels"] is None else self.compute_loss(inputs["labels"], lm_logits)

        # if not inputs["return_dict"]:
        #     output = (lm_logits,) + outputs[1:]
        #     return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        # return TFSeq2SeqLMOutput(
        #     loss=masked_lm_loss,
        #     logits=lm_logits,
        #     past_key_values=outputs.past_key_values,  # index 1 of d outputs
        #     decoder_hidden_states=outputs.decoder_hidden_states,  # index 2 of d outputs
        #     decoder_attentions=outputs.decoder_attentions,  # index 3 of d outputs
        #     encoder_last_hidden_state=outputs.encoder_last_hidden_state,  # index 0 of encoder outputs
        #     encoder_hidden_states=outputs.encoder_hidden_states,  # 1 of e out
        #     encoder_attentions=outputs.encoder_attentions,  # 2 of e out
        # )
        return lm_logits,

    def serving_output(self, output):
        # pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # return TFSeq2SeqLMOutput(
        #     logits=output.logits,
        #     past_key_values=pkv,
        #     decoder_hidden_states=dec_hs,
        #     decoder_attentions=dec_attns,
        #     encoder_last_hidden_state=output.encoder_last_hidden_state,
        #     encoder_hidden_states=enc_hs,
        #     encoder_attentions=enc_attns,
        # )
        return output

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past,
        attention_mask,
        head_mask=None,
        use_cache=None,
        **kwargs,
    ) -> Dict:
        assert past is not None and len(past) in {1, 2}, f"past has to be an iterable of length 1,2 got {past}"
        if len(past) == 1:
            assert isinstance(past[0], tf.Tensor), f"`past[0]` has to be of type `tf.Tensor`, but is {type(past[0])}"
            encoder_outputs = TFBaseModelOutput(last_hidden_state=past[0])
            past_key_values = None
        else:
            assert (
                len(past) == 2
            ), "`past` has to be of length 2 with the encoder_outputs at the first position and past_key_values at the second position."
            encoder_outputs, past_key_values = past
            if isinstance(encoder_outputs, tuple):
                assert isinstance(
                    encoder_outputs[0], tf.Tensor
                ), f"`encoder_outputs[0]` has to be of type `tf.Tensor`, but is {type(encoder_outputs[0])}"
                encoder_outputs = TFBaseModelOutput(last_hidden_state=encoder_outputs[0])
            elif isinstance(encoder_outputs, tf.Tensor):
                encoder_outputs = TFBaseModelOutput(last_hidden_state=encoder_outputs)
            assert (
                past_key_values
            ), f"decoder cached states must be truthy. got {past_key_values} from the 2nd element of past"
            decoder_input_ids = decoder_input_ids[:, -1:]

        assert isinstance(
            encoder_outputs, TFBaseModelOutput
        ), f"encoder_outputs should be a TFBaseModelOutput, Instead got {type(encoder_outputs)}."
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    @staticmethod
    def _reorder_cache(past, beam_idx):
        if len(past) == 1:
            return past

        past_key_values = past[1]

        reordered_past = ()
        for layer_past_key_values in past_key_values:
            reordered_past += (
                tuple(tf.gather(layer_past_key_value, beam_idx) for layer_past_key_value in layer_past_key_values[:2])
                + layer_past_key_values[2:],
            )
        return (past[0], reordered_past)

