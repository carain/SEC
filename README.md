# SEC
Seismic Event Classification



使用说明：

1. 输入的波形为单通道SAC格式文件。
2. 模型输入数据为 (1, 4000, 1)，采样率为100Hz，对应40秒数据
3. 输入数据需截取事件P波开始时间前5秒至P波后35秒
4. 输出为爆破与地震的概率，数组0为爆破的概率，数组1为地震的概率







