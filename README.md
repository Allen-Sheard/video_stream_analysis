# video_stream_analysis
分析视频，提取目标相关特征，得到目标的运动状态

- 利用OpenCV对视频进行二值化等处理，保存新的视频

```python video_process.py```

- `process.py`对图片进行二值化、边缘提取等操作，提取图片中目标的SIFT特征并进行两种图中的特征匹配、
  霍夫直线检测、扣取目标区域后进行二值化，比对黑白面积
  
- `test.py`调用`process.py`中的方法，对视频进行分析，检测目标整体的运动状态
