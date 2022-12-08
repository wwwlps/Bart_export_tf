import torch
import tensorflow as tf
from transformers.models.bart.modeling_tf_bart import TFBartForConditionalGeneration
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers.models.bert.modeling_tf_bert import TFBertForSequenceClassification
from transformers.models.gpt2.modeling_tf_gpt2 import TFGPT2LMHeadModel
from transformers import BertTokenizer
import numpy as np


model_path = r"C:\Users\刘平生\Desktop\公司业务\query结果\所用模型\bart-predict10-with2All"
bart_model = TFBartForConditionalGeneration.from_pretrained(model_path, from_pt=True)
toker = BertTokenizer.from_pretrained(model_path, do_lower_case=True)


# bart_model = TFBartForConditionalGeneration.from_pretrained("")
# bart_model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
# bart_model = BartForConditionalGeneration.from_pretrained(model_path)
bart_model.generate()
# tf.saved_model.save(bart_model, "test_tf_save")


class Query(tf.Module):
    def __init__(self):
        super(Query, self).__init__()
        self.model = bart_model
        self.bos_token_id = 101
        self.pad_token_id = 0
        self.eos_token_id = 102
        self.max_length = 4
        self.min_length = 1
        self.do_sample = True
        self.num_beams = 1
        self.temperature = 0.7
        self.top_k = 0
        self.top_p = 0.9
        self.repetition_penalty = 1.05
        self.num_return_sequences = 1
        self.decoder_start_token_id = 101
        self.filter_value = -float("Inf")
        self.min_tokens_to_keep = 1

    def cal(self, x):
        f = x.numpy()
        size = tf.math.reduce_min(x)
        ans = tf.ones([4, size])
        # xy = [1 for u in x]
        xxx = np.zeros(ans.shape[0])
        for i in range(ans.shape[0]):
            print(ans[i])
        print(x[1])
        xxxx = tf.unique(ans[0])
        return ans

    # @tf.function(input_signature=[{
    #     'sentence1': tf.TensorSpec(shape=[None], dtype=tf.string),
    #     'sentence2': tf.TensorSpec(shape=[None], dtype=tf.string)}])
    @tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.string),))
    def __call__(self, inputs):
        # print(inputs)
        # xxx = tf.constant(0).numpy()

        while tf.constant(1) < tf.constant(2):
            if tf.math.reduce_max(tf.constant([0])) == tf.constant(0):
                break
        # ans = self.model(inputs)
        # aaa = np.ones(10)
        # ress = self.cal(tf.constant([1,2,3]))
        # yyz = [np.unique(input_id) for input_id in [1,2,3]]
        xxx = self.model(tf.constant([[1, 2, 3], [4, 5, 6]]))
        ans = self.model.generate(tf.constant([[101, 1599, 3614, 4638, 1599], [101, 1599, 3614, 4638, 1599]]), num_beams=1, min_length=1, max_length=20, pad_token_id=0, top_p=0.9, top_k=1,
                                  temperature=0.7,
                                  bos_token_id=101,
                                  eos_token_id=102,
                                  decoder_start_token_id=101,
                                  repetition_penalty=1.0,
                                  no_repeat_ngram_size=0)
        # ans = tf.zeros(2)
        # input_ids = inputs
        # attention_mask = tf.cast(tf.math.not_equal(ans, 0), dtype=tf.int32)
        # encoder_input_ids = input_ids  # src_input_ids
        #
        # encoder_last_hidden_state = self.bart_encoder(encoder_input_ids, attention_mask)
        #
        # input_ids = (
        #         tf.ones(
        #             (input_ids.shape[0], 1),
        #             dtype=tf.int32,
        #         )
        #         * self.decoder_start_token_id
        # )
        # cur_len = 1
        # unfinished_sents = tf.ones_like(input_ids[:, 0])
        # sent_lengths = tf.ones_like(input_ids[:, 0]) * self.max_length
        #
        # while cur_len < self.max_length:
        #     decoder_input_ids = input_ids
        #     decoder_input_ids = decoder_input_ids[:, -1:]
        #     outputs = self.model(decoder_input_ids, encoder_last_hidden_state, attention_mask)
        #     next_token_logits = outputs[0][:, -1, :]
        #
        #     #
        #     logits = next_token_logits
        #     repetition_penalty = self.repetition_penalty
        #     static = logits.shape.as_list()
        #     dynamic = tf.shape(logits)
        #     xxxx = [dynamic[i] if s is None else s for i, s in enumerate(static)]
        #     token_penalties = np.ones(xxxx)
        #     prev_input_ids = [np.unique(input_id) for input_id in input_ids.numpy()]
        #     for i, prev_input_id in enumerate(prev_input_ids):
        #         logit_penalized = logits[i].numpy()[prev_input_id]
        #         logit_penalties = np.zeros(logit_penalized.shape)
        #         # if previous logit score is < 0 then multiply repetition penalty else divide
        #         logit_penalties[logit_penalized < 0] = repetition_penalty
        #         logit_penalties[logit_penalized > 0] = 1 / repetition_penalty
        #         np.put(token_penalties[i], prev_input_id, logit_penalties)
        #     next_token_logits_penalties = tf.convert_to_tensor(token_penalties, dtype=tf.float32)
        #     next_token_logits = tf.math.multiply(next_token_logits, next_token_logits_penalties)
        #
        #     # do sample
        #     next_token_logits = next_token_logits / self.temperature
        #     next_token_logits = tf_top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        #     # Sample
        #     next_token = tf.squeeze(
        #         tf.random.categorical(next_token_logits, dtype=tf.int32, num_samples=1), axis=1
        #     )
        #
        #     tokens_to_add = next_token * unfinished_sents + (self.pad_token_id) * (1 - unfinished_sents)
        #     input_ids = tf.concat([input_ids, tf.expand_dims(tokens_to_add, -1)], 1)
        #     cur_len = cur_len + 1
        #
        #     eos_in_sents = tokens_to_add == self.eos_token_id
        #     # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
        #     is_sents_unfinished_and_token_to_add_is_eos = tf.math.multiply(
        #         unfinished_sents, tf.cast(eos_in_sents, tf.int32)
        #     )
        #     sent_lengths = (
        #             sent_lengths * (1 - is_sents_unfinished_and_token_to_add_is_eos)
        #             + cur_len * is_sents_unfinished_and_token_to_add_is_eos
        #     )
        #
        #     # unfinished_sents is set to zero if eos in sentence
        #     unfinished_sents -= is_sents_unfinished_and_token_to_add_is_eos
        #
        #     if tf.math.reduce_max(unfinished_sents) == 0:
        #         break
        #
        # min_sent_length = tf.math.reduce_min(sent_lengths)
        # max_sent_length = tf.math.reduce_max(sent_lengths)
        # if min_sent_length != max_sent_length:
        #     # finished sents are filled with pad_token
        #     padding = tf.ones([batch_size, max_sent_length.numpy()], dtype=tf.int32) * self.pad_token_id
        #
        #     # create length masks for tf.where operation
        #     broad_casted_sent_lengths = tf.broadcast_to(
        #         tf.expand_dims(sent_lengths, -1), [batch_size, max_sent_length]
        #     )
        #     broad_casted_range = tf.transpose(
        #         tf.broadcast_to(tf.expand_dims(tf.range(max_sent_length), -1), [max_sent_length, batch_size])
        #     )
        #
        #     decoded = tf.where(broad_casted_range < broad_casted_sent_lengths, input_ids, padding)
        # else:
        #     decoded = input_ids

        b = "you is "

        # return {"ans": tf.constant([[1,2,3], [4,5,6]])}
        return {"ans": ans}


if __name__ == '__main__':
    model = Query()
    # tf.saved_model.save(model, "test_tf_save",
    #                     signatures=model.call.get_concrete_function(tf.TensorSpec(shape=[None, None], dtype=tf.int32)))
    # tf.compat.v1.enable_eager_execution()
    tf.saved_model.save(model, "test_tf_save",
                        signatures={'serving_default': model.__call__})
    aaa = tf.saved_model.load('test_tf_save')
    # my_examples = {
    #     'sentence1': [
    #         'The rain in Spain falls mainly on the plain.',
    #         'Look I fine tuned BERT.'],
    # }
    # my_examples = {"inputs": ["你好", "好的"]}
    my_examples = ["你好", "好的"]
    res = aaa(my_examples)
    print(res["ans"])
    # tf.keras.models.save_model(model, "test_tf_save2")
