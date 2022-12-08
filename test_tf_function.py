import torch
import tensorflow as tf
from transformers.models.bart.modeling_tf_bart import TFBartForConditionalGeneration
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers.models.bert.modeling_tf_bert import TFBertForSequenceClassification
from transformers.models.gpt2.modeling_tf_gpt2 import TFGPT2LMHeadModel
from transformers import BertTokenizer
import numpy as np

a = tf.constant([1,2,3])

@tf.function
def cal(x):
    size = tf.math.reduce_min(x)
    ans = tf.ones([4, size])
    # xy = [1 for u in x]
    xxx = np.zeros(ans.shape[0])
    for i in range(ans.shape[0]):
        print(ans[i])
    print(x[1])
    xxxx = np.unique(ans[0])
    return ans

# res = cal(a)
# print(res)

# 101 2582  720 1091  102
x = tf.constant([0,0,0])


def ftx(ii):
    amd = tf.py_function(lambda x: tf.math.reduce_max(x) == 0, inp=[ii], Tout=tf.bool)
    return amd


# amd = tf.math.reduce_max(unfinished_sents) == 0
ans = ftx(tf.constant(0))
print(ans)
