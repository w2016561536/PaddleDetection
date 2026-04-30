import os
import numpy as np

input_scale = 0.35018664598464966
inv_scale = np.float32(1.0) / np.float32(input_scale)

input_data = np.load("output_from_lite_runtime.npy").astype("float32")
input_data = input_data * inv_scale

# 加载int8 权重
weight_data = np.load("depthwise2d_weights.npy").astype("int32") 

# 加载int8 缩放系数
weights_scale = np.array([0.004361940082162619, 0.005932476371526718, 0.011012737639248371, 0.0004047783149871975, 0.0005917985690757632, 0.0024921002332121134, 0.011278200894594193, 0.00982583500444889, 0.008456436917185783, 0.0025821852032095194, 0.002195417182520032, 0.00016134792531374842, 0.0002372605522396043, 0.004676727578043938, 0.002779509639367461, 0.0003671871090773493, 0.0032303191255778074, 0.005764181725680828, 0.004389831330627203, 0.0005462115514092147, 0.0007484168745577335, 0.0016112369485199451, 0.0023101612459868193, 0.0014129136689007282]).astype("float32")

bias_data = np.array([
    4.405367851257324,
    2.3285253047943115,
    -1.1521813869476318,
    1.7974059581756592,
    3.0866289138793945,
    1.488497257232666,
    0.5528708696365356,
    4.002745151519775,
    2.0521140098571777,
    3.083024501800537,
    1.60580313205719,
    1.7898988723754883,
    2.372697353363037,
    4.260335922241211,
    -3.2369797229766846,
    2.30985951423645,
    2.737835168838501,
    3.8135101795196533,
    2.9685001373291016,
    2.0366764068603516,
    1.9386637210845947,
    0.01618748903274536,
    3.6546735763549805,
    2.1046996116638184]
).astype("float32")

padding = np.array([1,1])

stride = np.array([2,2])


def hard_swish(x):
    return x * (np.clip(x + 3, 0, 6) / 6)


def depthwise_conv2d(input_data1, weight_data, bias_data, weights_scale, calib_scale):
    batch_size, in_channels, in_height, in_width = np.array([1,24,208,208]).astype("int32")
    out_channels, _, kernel_height, kernel_width = np.array([24,1,3,3]).astype("int32")

    padding_h, padding_w = 1, 1
    stride_h, stride_w = 2, 2
    
    # 计算输出尺寸
    out_height = (in_height + 2 * padding_h - kernel_height) // stride_h + 1
    out_width = (in_width + 2 * padding_w - kernel_width) // stride_w + 1
    
    # 对输入进行 padding
    padded_input = np.pad(input_data1, ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w)), 
                          mode='constant', constant_values=0)
    
    # 初始化输出
    output_data = np.zeros((batch_size, out_channels, out_height, out_width), dtype="float32")
    
    # 深度卷积：每个输出通道只与一个输入通道相关（groups = out_channels）
    for c in range(out_channels):
        # 获取当前通道的权重 [1, 3, 3]
        weight = weight_data[c, 0, :, :].astype("int32")
        
        # 在高度和宽度上执行卷积
        for h_out in range(out_height):
            for w_out in range(out_width):
                h_in = h_out * stride_h
                w_in = w_out * stride_w
                # 提取输入patch
                patch = padded_input[0, c, h_in:h_in+kernel_height, w_in:w_in+kernel_width].astype("int32")
                # 计算卷积结果：点乘后求和，加上偏置
                output_data[0, c, h_out, w_out] = np.sum(patch * weight, dtype="int32")  # 使用更高精度的整数类型来避免溢出
        output_data[0, c, :, :] *=  weights_scale[c] / inv_scale
        output_data[0, c, :, :] += bias_data[c].astype("float32")

    
    return output_data

if __name__ == "__main__":
    input_data_int8 = np.round(input_data).astype("int32")
    # input_data_int8 = np.where(
    # input_data >= 0,
    # np.floor(input_data + np.float32(0.5)),
    # np.ceil(input_data - np.float32(0.5))
    #   ).astype(np.int32)

    input_data_int8 = np.clip(input_data_int8, -127, 127)

    output_data = depthwise_conv2d(input_data_int8, weight_data, bias_data, weights_scale, input_scale)

    # 应用hard swish激活函数
    output_data_hard_swish = hard_swish(output_data)

    # 保存输出结果
    np.save("output_hardwish_1_litedepthwise2d.npy", output_data_hard_swish)

    print("first 10 elements of the 1 channel:", output_data_hard_swish[0, 1, 0, :10])

