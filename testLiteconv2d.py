import numpy as np

def round_away_from_zero(x):
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))

calib_scale = 0.2261638045310974
inv_scale = np.float32(1.0) / np.float32(calib_scale)
input_data = np.load("output_hardwish_1.npy").astype("float32")

input_data = input_data * inv_scale

# 加载int8 权重
weight_data = np.load("conv2d_weights.npy").astype("int32") 

# 加载int8 缩放系数
weights_scale = np.array([0.00422132620587945, 0.007228214759379625, 0.03149915114045143, 0.015298280864953995, 0.004579391796141863, 0.005920222494751215, 0.00269253714941442, 0.0037343637086451054, 0.003853008383885026, 0.011956640519201756, 0.010337473824620247, 0.2536289691925049, 0.08170392364263535, 0.002079658443108201, 0.004489982035011053, 0.0688115581870079, 0.007074081804603338, 0.0010581788374111056, 0.013578112237155437, 0.007465363014489412, 0.05379762873053551, 0.006693805567920208, 0.005410909652709961, 0.003939296118915081]).astype("float32")

# 卷积后bias
bias_data = np.array([
    -0.1005251407623291,
    2.4909214973449707,
    11.85042667388916,
    -5.706641674041748,
    3.26167368888855,
    -0.4543790817260742,
    -0.40915486216545105,
    3.8443193435668945,
    0.7715367078781128,
    1.037082552909851,
    2.3108980655670166,
    -253.12091064453125,
    34.32855224609375,
    3.522444009780884,
    1.023667812347412,
    9.642375946044922,
    6.717805862426758,
    -3.3453750610351562,
    2.5333123207092285,
    0.6846879720687866,
    -17.461645126342773,
    -3.5966808795928955,
    3.763953685760498,
    4.103872776031494
]).astype("float32")

def hard_swish(x):
    return x * (np.clip(x + 3, 0, 6) / 6)

def conv2d(input_data1, weight_data, bias_data, weights_scale, calib_scale):
    batch_size, in_channels, in_height, in_width = np.array([1,16,208,208]).astype("int32")
    out_channels, _, kernel_height, kernel_width = np.array([24,16,1,1]).astype("int32")

    # print("input weights: ", weight_data)

    # 输出特征图的尺寸
    out_height = in_height - kernel_height + 1
    out_width = in_width - kernel_width + 1

    # 初始化输出特征图
    output_data = np.zeros((batch_size, out_channels, out_height, out_width), dtype=np.float32)
    output_data_single_channel_int32 = np.zeros((out_height, out_width), dtype=np.int32)

    # 卷积计算
    for oc in range(out_channels):
        output_data_single_channel_int32 = np.zeros((out_height, out_width), dtype=np.int32)
        for ic in range(in_channels):
            #output_data[0, oc] += input_data1[0, ic] * weight_data[oc, ic, 0, 0]
            output_data_single_channel_int32 += input_data1[0, ic] * weight_data[oc, ic, 0, 0]
        output_data[0, oc] = output_data_single_channel_int32.astype("float32")
        output_data[0, oc] *= weights_scale[oc] / inv_scale.astype("float32")
        # output_data[0, oc] = np.round(output_data[0, oc]).astype("float32")
        output_data[0, oc] += bias_data[oc].astype("float32")
        # output_data[0, oc] = np.clip(output_data[0, oc], -128, 127)

    return output_data


# 执行卷积计算
if __name__ == "__main__":
    # 四舍五入，转换到int8 nparray
#     input_data_int8 = np.where(
#     input_data >= 0,
#     np.floor(input_data + np.float32(0.5)),
#     np.ceil(input_data - np.float32(0.5))
#       ).astype(np.int32)
    input_data_int8 = np.round(input_data).astype("int32")

    input_data_int8 = np.clip(input_data_int8, -127, 127)

    output_data = conv2d(input_data_int8, weight_data, bias_data, weights_scale, calib_scale)

    # 应用hard swish激活函数
    output_data_hard_swish = hard_swish(output_data)
    # 保存输出结果
    np.save("output_hardwish_1_liteconv2d.npy", output_data_hard_swish)

    print("first 10 elements of the 1 channel:", output_data_hard_swish[0, 1, 0, :10])
