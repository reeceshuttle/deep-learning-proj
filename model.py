from auto_gptq.modeling import BaseGPTQForCausalLM

class OLMoGPTQForCausalLM(BaseGPTQForCausalLM):
    # chained attribute name of transformer layer block
    layers_block_name = "model.transformer.blocks"
    # chained attribute names of other nn modules that in the same level as the transformer layer block
    outside_layer_modules = [
        "model.transformer.wte" #, "model.transformer.ln_f", "model.transformer.emb_drop"
    ]
    # chained attribute names of linear layers in transformer layer module
    # normally, there are four sub lists, for each one the modules in it can be seen as one operation,
    # and the order should be the order when they are truly executed, in this case (and usually in most cases),
    # they are: attention q_k_v projection, attention output projection, MLP project input, MLP project output
    inside_layer_modules = [
        ["att_proj"],
        ["attn_out"],
        ["ff_proj"],
        ["ff_out"]
    ]