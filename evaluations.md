# STDAN original run on 9th epoch
process: 0.25
process: 0.5
process: 0.75
valmse: 38.893956507441615
tensor([0.0578, 0.1147, 0.2049, 0.3070, 0.4163, 0.5300, 0.6455, 0.7629, 0.8828,
        1.0057, 1.1321, 1.2631, 1.3997, 1.5426, 1.6924, 1.8496, 2.0153, 2.1897,
        2.3738, 2.5675, 2.7704, 2.9823, 3.2038, 3.4397, 3.6997],
       device='cuda:0')


# Transformer instead of LSTM based 5 step or 1 s prediction: On top of th STDAN
exp_name = tf_enc_dec
tensor([0.0475, 0.0980, 0.1814, 0.2809, 0.3896], device='cuda:0')




# Transformer instead of LSTM based 25 step or 5 s prediction: Best Run
exp_name = tf_enc_dec
tensor([0.0508, 0.1087, 0.1973, 0.2992, 0.4082, 0.5208, 0.6347, 0.7490, 0.8653,
        0.9848, 1.1090, 1.2366, 1.3705, 1.5106, 1.6583, 1.8146, 1.9786, 2.1512,
        2.3328, 2.5223, 2.7212, 2.9308, 3.1488, 3.3765, 3.6199],
       device='cuda:0')

tensor([0.4082,  0.9848,  1.6583, 2.5223, 3.6199],
       device='cuda:0')

# Nadam otpimizer, 25 steps, transformer, loss equation: loss_g = loss_g1 + loss_gx

Expreiment Name: tf_enc_dec_Nadam_optim
begin................................. best
process: 0.25
process: 0.5
process: 0.75
valmse: 37.32304510249971
tensor([0.0514, 0.1086, 0.1961, 0.2969, 0.4053, 0.5188, 0.6339, 0.7492, 0.8658,
        0.9853, 1.1083, 1.2359, 1.3694, 1.5086, 1.6549, 1.8095, 1.9730, 2.1450,
        2.3257, 2.5159, 2.7153, 2.9249, 3.1444, 3.3746, 3.6158],
       device='cuda:0')

tensor([ 0.4053, 0.9853,  1.6549, 2.5159, 3.6158],
       device='cuda:0')
# Loss fusion customize, 25 steps, transformer, loss equation: loss_g = loss_g1 + 50 * loss_gx

begin................................. best
process: 0.25
process: 0.5
process: 0.75
valmse: 34.3431135848693
tensor([0.1133, 0.1542, 0.2305, 0.3245, 0.4284, 0.5369, 0.6476, 0.7591, 0.8728,
        0.9883, 1.1062, 1.2275, 1.3539, 1.4848, 1.6205, 1.7634, 1.9121, 2.0690,
        2.2336, 2.4056, 2.5878, 2.7774, 2.9765, 3.1901, 3.4207],
       device='cuda:0')


# Loss fusion customize, 25 steps, transformer, loss equation: loss_g = loss_g1 + 10 * loss_gx
Expreiment Name: lat_long_ce_loss_scale_10
process: 0.25
process: 0.5
process: 0.75
valmse: 39.27711481648168
tensor([0.0609, 0.1177, 0.2071, 0.3108, 0.4215, 0.5357, 0.6520, 0.7698, 0.8911,
        1.0164, 1.1461, 1.2795, 1.4178, 1.5623, 1.7124, 1.8696, 2.0335, 2.2048,
        2.3865, 2.5768, 2.7793, 2.9929, 3.2171, 3.4505, 3.6959],
       device='cuda:0')


# Loss fusion customize, 25 steps, transformer, loss equation: loss_g = loss_g1 + 100 * loss_gx

  gdEncoder.load_state_dict(t.load(l_path + '/models/best/' + 'epochbest_gd.tar', map_location='cuda:0'))
begin................................. best
process: 0.25
process: 0.5
process: 0.75
valmse: 34.26325508297221
tensor([0.0916, 0.1521, 0.2378, 0.3361, 0.4408, 0.5479, 0.6570, 0.7674, 0.8805,
        0.9963, 1.1147, 1.2367, 1.3624, 1.4926, 1.6283, 1.7705, 1.9167, 2.0678,
        2.2286, 2.3969, 2.5770, 2.7673, 2.9674, 3.1776, 3.4046],
       device='cuda:0')

# Loss fusion customize, 25 steps, transformer, loss equation: loss_g = loss_g1 + 200 * loss_gx

Expreiment Name: lat_long_ce_loss_scale_200
begin................................. best
process: 0.25
process: 0.5
process: 0.75
valmse: 32.10391499326843
tensor([0.1009, 0.1590, 0.2430, 0.3393, 0.4423, 0.5490, 0.6573, 0.7663, 0.8759,
        0.9870, 1.1001, 1.2165, 1.3363, 1.4600, 1.5889, 1.7223, 1.8610, 2.0054,
        2.1579, 2.3189, 2.4895, 2.6689, 2.8583, 3.0557, 3.2639],
       device='cuda:0')

tensor([ 0.4423, 0.9870,  1.5889,  2.3189,  3.2639],
       device='cuda:0')

# Loss fusion customize, 25 steps, transformer, loss equation: loss_g = loss_g1 + 80 * loss_gx

Expreiment Name: lat_long_ce_loss_scale_80

begin................................. best
process: 0.25
process: 0.5
process: 0.75
valmse: 36.695385363309796
tensor([0.0825, 0.1430, 0.2303, 0.3314, 0.4392, 0.5506, 0.6649, 0.7808, 0.9002,
        1.0228, 1.1482, 1.2759, 1.4084, 1.5456, 1.6879, 1.8343, 1.9881, 2.1471,
        2.3137, 2.4885, 2.6721, 2.8667, 3.0721, 3.2888, 3.5187],
       device='cuda:0')

tensor([ 0.4392, 1.0228,  1.6879,  2.4885,  3.5187],
       device='cuda:0')


# Loss fusion customize, 25 steps, transformer, loss equation: loss_g = loss_g1 + 400 * loss_gx
Expreiment Name: lat_long_ce_loss_scale_400
begin................................. best
process: 0.25
process: 0.5
process: 0.75
valmse: 31.934969464239792
tensor([0.1185, 0.1712, 0.2532, 0.3502, 0.4541, 0.5614, 0.6699, 0.7785, 0.8879,
        0.9990, 1.1118, 1.2267, 1.3450, 1.4677, 1.5943, 1.7260, 1.8626, 2.0055,
        2.1546, 2.3120, 2.4766, 2.6507, 2.8360, 3.0330, 3.2363],
       device='cuda:0')
(vtp) ➜  TF-STDAN git:(

# Loss fusion customize, 25 steps, transformer, loss equation: loss_g = loss_g1 + 1000 * loss_gx

  gdEncoder.load_state_dict(t.load(l_path + '/models/best/' + 'epochbest_gd.tar', map_location='cuda:0'))
begin................................. best
process: 0.25
process: 0.5
process: 0.75
valmse: 34.62751747734467
tensor([0.2375, 0.3063, 0.3526, 0.4230, 0.5095, 0.6051, 0.7043, 0.8078, 0.9153,
        1.0262, 1.1407, 1.2593, 1.3818, 1.5100, 1.6437, 1.7845, 1.9303, 2.0834,
        2.2432, 2.4099, 2.5848, 2.7702, 2.9631, 3.1617, 3.3626],
       device='cuda:0')

# Expreiment Name: no_scale_man_loss_use_spatial_adamw
begin................................. best
process: 0.25
process: 0.5
process: 0.75
valmse: 46.032121284378654
tensor([0.0475, 0.1083, 0.2000, 0.3068, 0.4209, 0.5375, 0.6562, 0.7736, 0.8975,
        1.0269, 1.1574, 1.2944, 1.4335, 1.5744, 1.7186, 1.8686, 2.0265, 2.2010,
        2.3947, 2.6228, 2.8933, 3.2121, 3.5846, 4.0136, 4.4950],
       device='cuda:0')