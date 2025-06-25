# -*- coding: utf-8 -*-

"""

@author: hmq

波形事件分类程序
数据范围：福建及周边
数据时间：2011年至2023年

使用说明：

1. 输入的波形为单通道SAC格式文件。
2. 模型输入数据为 (1, 4000, 1)，采样率为100Hz，对应40秒数据
3. 输入数据需截取事件P波开始时间前5秒至P波后35秒

"""
from tensorflow.keras.models import model_from_json
from obspy import read
import numpy as np


model_save_path=r'.\model_save\\'
model_name = 'model'
model_file=('%s%s.json'%(model_save_path,model_name))
model_weight=('%s%s.h5'%(model_save_path,model_name))
with open(model_file, 'r') as file:
    model_json = file.read()
inception_model = model_from_json(model_json)
inception_model.load_weights(model_weight)
#x为波形数据,采样率100Hz，时长40s，输入为(1,4000,1)，pre为预测结果，0爆破，1地震
sac_file_path = 'data/2016.127.11.38.51.0000.BU.LAY.00.BHE.D.SAC'
# 读取和处理波形数据
try:
    st = read(sac_file_path)
    st.detrend()
    st.filter("bandpass", freqmin=1, freqmax=25, corners=2, zerophase=True)
    st.normalize()

    starttime = st[0].stats.starttime
    st.trim(starttime + 150, starttime + 190)
    st.plot()

    data = st[0].data
    if len(data) < 4000:
        raise ValueError("数据长度不足4000点，无法预测。")

    x = data[0:4000].reshape(1, 4000, 1)

    # 模型预测
    pre = inception_model.predict(x)
    print("预测完成，结果为：")
    print(pre)
    print("分类结果：", "地震" if pre[0][0] <0.5 else "爆破")

except Exception as e:
    print(f"波形处理或预测失败：{e}")