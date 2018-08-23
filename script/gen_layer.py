#!/usr/bin/env python

def generate_data_layer_str(label_source, data_source, local_dict,train_batch_size, test_batch_size,
                            channel, train_num_words, test_num_words, c_h, c_w):
    train_label = label_source+'/train_label.txt'
    train_data = data_source+'/train_data.txt'
    test_label = label_source+'/test_label.txt'
    test_data = data_source+'/test_data.txt'

    data_layer_str = '''name: "DenseAttentionNet"
layer {
  name: "data"
  type: "TextData"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  text_data_param {
    label_source: "%s"
    data_source: "%s"
    dict_source: "%s"
    batch_size: %d
    channel: %d
    num_words: %d
    crop_height: %d
    crop_width: %d
  }
}
layer {
  name: "data"
  type: "TextData"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  text_data_param {
    label_source: "%s"
    data_source: "%s"
    dict_source: "%s"
    batch_size: %d
    channel: %d
    num_words: %d
    crop_height: %d
    crop_width: %d
    shuffle: 0
  }
}\n'''%(train_label, train_data, local_dict, train_batch_size, channel, train_num_words, c_h, c_w,
        test_label, test_data, local_dict, test_batch_size, channel, test_num_words, c_h, c_w)
    return data_layer_str

def generate_conv_layer_str(name, bottom, top, num_output, kernel_h, kernel_w, pad_h, pad_w, num_groups=1):
    conv_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "Convolution"
  convolution_param {
    num_output: %d
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.058925565
    }
    bias_filler {
      type: "constant"
      value: 0
    }
    group: %d
    kernel_h: %d
    kernel_w: %d
    pad_h: %d
    pad_w: %d
  }
}\n'''%(name, bottom, top, num_output, num_groups, kernel_h, kernel_w, pad_h, pad_w)
    return conv_layer_str

def generate_bn_layer_str(bn_name, bottom, top):
    bn_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "BatchNorm"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
}\n'''%(bn_name, bottom, top)
    return bn_layer_str

def generate_activation_layer_str(name, bottom, act_type="ReLU"):
    act_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "%s"
}\n'''%(name, bottom, bottom, act_type)
    return act_layer_str

def generate_concat_layer_str(name, bottoms, top, concat_axis=1):
    bottom_str = ''
    for item in bottoms:
        bottom_str += '  bottom: "'+item+'"\n'
    concat_layer_str = '''layer {
  name: "%s"
%s  top: "%s"
  type: "Concat"
  concat_param {
    axis: %d
  }
}\n'''%(name, bottom_str, top, concat_axis)
    return concat_layer_str

def generate_slice_layer_str(name, bottom, top, num_slice, num_dimension, axis=1):
    top_str = ''
    slice_point_str = '    axis: '+str(axis)+'\n'
    for i in range(num_slice):
        top_str += '  top: "' + top + str(i)+'"\n'
    for i in range(1, num_slice):
        slice_point_str += '    slice_point: ' + str(i*num_dimension) + '\n'
    slice_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
%s  type: "Slice"
  slice_param {
%s    }
}\n'''%(name, bottom, top_str, slice_point_str)
    return slice_layer_str

def generate_permute_layer_str(name, bottom, top, permute_param_list):
    permute_param_str = 'permute_param {\n'
    for i in permute_param_list:
        permute_param_str += '      order: ' + str(i) + '\n'
    permute_param_str += '  }'

    permute_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "Permute"
  %s
}\n'''%(name, bottom, top, permute_param_str)
    return permute_layer_str

def generate_reduction_layer_str(name, bottom, top, reduction_axis):
    reduction_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "Reduction"
  reduction_param {
    axis: %d
    }
}\n'''%(name, bottom, top, reduction_axis)
    return reduction_layer_str

def generate_tile_layer_str(name, bottom, top, axis, tiles):
    tile_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "Tile"
  tile_param {
    axis: %d
    tiles: %d
  }
}\n'''%(name, bottom, top, axis, tiles)
    return tile_layer_str

def generate_reshape_layer_str(name, bottom, top, reshape_param_list):
    reshape_param_str = 'reshape_param {\n    shape {\n'
    for i in reshape_param_list:
        reshape_param_str += '      dim: ' + str(i) + '\n'
    reshape_param_str += '    }\n  }'

    reshape_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "Reshape"
  %s
}\n'''%(name, bottom, top, reshape_param_str)
    return reshape_layer_str

def generate_eltwise_layer_str(name, bottoms, top, operation):
    bottom_str = ''
    for item in bottoms:
        bottom_str += '  bottom: "'+item+'"\n'
    eltwise_layer_str = '''layer {
  name: "%s"
%s  top: "%s"
  type: "Eltwise"
  eltwise_param {
    operation: %s
  }
}\n'''%(name, bottom_str, top, operation)
    return eltwise_layer_str

def generate_pooling_layer_str(name, bottom, top, pool_type="AVE", kernel_h=3, kernel_w=1, stride_h=2, stride_w=1, pad_h=1, pad_w=0):
    if pool_type == 'global_pooling':
        pool_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "Pooling"
  pooling_param {
    global_pooling : true
  }
}\n'''%(name, bottom, top)
    else:
        pool_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "Pooling"
  pooling_param {
    pool: %s
    kernel_h: %d
    kernel_w: %d
    stride_h: %d
    stride_w: %d
    pad_h: %d
    pad_w: %d
  }
}\n'''%(name, bottom, top, pool_type, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)
    return pool_layer_str

def generate_fc_layer_str(name, bottom, top, num_output, param_list=[]):
    if param_list == []:
        fc_layer_str = '''layer {
  name: "%s"
  type: "InnerProduct"
  bottom: "%s"
  top: "%s"
  inner_product_param {
     num_output: %d
     weight_filler {
       type: "gaussian"
       std: 0.001
     }
     bias_filler {
       type: "constant"
       value: 0
     }
  }
}\n'''%(name, bottom, top, num_output)
    else:
        param_str = '''param {
    lr_mult: %d
    decay_mult: %d
  }
  param {
    lr_mult: %d
    decay_mult: %d
  }'''%(param_list[0], param_list[1], param_list[2], param_list[3])

        fc_layer_str = '''layer {
  name: "%s"
  type: "InnerProduct"
  bottom: "%s"
  top: "%s"
  %s
  inner_product_param {
     num_output: %d
     weight_filler {
       type: "gaussian"
       std: 0.001
     }
     bias_filler {
       type: "constant"
       value: 0
     }
  }
}\n'''%(name, bottom, top, param_str, num_output)

    return fc_layer_str

def generate_flatten_layer_str(name, bottom, top):
    flatten_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "Flatten"
}\n'''%(name, bottom, top)
    return flatten_layer_str


def generate_dropout_layer_str(name, bottom, top, dropout_ratio):
    dropout_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "Dropout"
  dropout_param {
    dropout_ratio: %f
  }
}\n'''%(name, bottom, top, dropout_ratio)
    return dropout_layer_str


def generate_softmax_layer_str(name, bottom, top):
    softmax_layer_str = '''layer {
  name: "%s"
  type: "Softmax"
  bottom: "%s"
  top: "%s"
}\n'''%(name, bottom, top)
    return softmax_layer_str

def generate_softmax_loss_str(name, bottom0, bottom1, top, loss_weight=1):
    softmax_loss_str = '''layer {
  name: "%s"
  type: "SoftmaxWithLoss"
  bottom: "%s"
  bottom: "%s"
  top: "%s"
  loss_weight: %f
}\n'''%(name, bottom0, bottom1, top, loss_weight)
    return softmax_loss_str

def generate_accuracy_str(name, bottom0, bottom1, top):
    accuracy_str = '''layer {
  name: "%s"
  type: "Accuracy"
  bottom: "%s"
  bottom: "%s"
  top: "%s"
}\n'''%(name, bottom0, bottom1, top)
    return accuracy_str

