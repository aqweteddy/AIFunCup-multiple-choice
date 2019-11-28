import tensorflow as tf

from transformers.configuration_albert import AlbertConfig
from transformers.file_utils import add_start_docstrings
from transformers.modeling_tf_albert import TFAlbertPreTrainedModel, TFAlbertModel, ALBERT_START_DOCSTRING, \
    ALBERT_INPUTS_DOCSTRING
from transformers.modeling_tf_utils import get_initializer


@add_start_docstrings("""Bert Model with a multiple choice classification head on top (a linear layer on top of
                      the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
                      ALBERT_START_DOCSTRING,
                      ALBERT_INPUTS_DOCSTRING)
class TFAlbertForMultipleChoice(TFAlbertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **classification_scores**: ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, num_choices)`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above).
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``Numpy array`` or ``tf.Tensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``Numpy array`` or ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        import tensorflow as tf
        from transformers import BertTokenizer, TFBertForMultipleChoice
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = TFBertForMultipleChoice.from_pretrained('bert-base-uncased')
        choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]
        input_ids = tf.constant([tokenizer.encode(s) for s in choices])[None, :]  # Batch size 1, 2 choices
        outputs = model(input_ids)
        classification_scores = outputs[0]
    """

    def __init__(self, config, *inputs, **kwargs):
        super(TFAlbertForMultipleChoice, self).__init__(config, *inputs, **kwargs)

        self.albert = TFAlbertModel(config, name='albert')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(1,
                                                kernel_initializer=get_initializer(config.initializer_range),
                                                name='classifier')

    def call(self, inputs, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
             inputs_embeds=None, training=False):
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            token_type_ids = inputs[2] if len(inputs) > 2 else token_type_ids
            position_ids = inputs[3] if len(inputs) > 3 else position_ids
            head_mask = inputs[4] if len(inputs) > 4 else head_mask
            inputs_embeds = inputs[5] if len(inputs) > 5 else inputs_embeds
            assert len(inputs) <= 6, "Too many inputs."
        elif isinstance(inputs, dict):
            input_ids = inputs.get('input_ids')
            attention_mask = inputs.get('attention_mask', attention_mask)
            token_type_ids = inputs.get('token_type_ids', token_type_ids)
            position_ids = inputs.get('position_ids', position_ids)
            head_mask = inputs.get('head_mask', head_mask)
            inputs_embeds = inputs.get('inputs_embeds', inputs_embeds)
            assert len(inputs) <= 6, "Too many inputs."
        else:
            input_ids = inputs

        if input_ids is not None:
            num_choices = tf.shape(input_ids)[1]
            seq_length = tf.shape(input_ids)[2]
        else:
            num_choices = tf.shape(inputs_embeds)[1]
            seq_length = tf.shape(inputs_embeds)[2]

        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None

        flat_inputs = [flat_input_ids, flat_attention_mask, flat_token_type_ids, flat_position_ids, head_mask,
                       inputs_embeds]

        outputs = self.albert(flat_inputs, training=training)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # reshaped_logits, (hidden_states), (attentions)
