import pandas as pd
import matplotlib.pyplot as plt
import pathlib

import pandas.core.series

excel_path = [""]*10
excel_path[0] = "E:\DemoDatas\\runs\detect\\Adam_origin_yolov8_2000\F1_curve.csv"
excel_path[1] = "E:\DemoDatas\\runs\detect\\Adam_origin_yolov8-ca_2000\F1_curve.csv"
excel_path[2] = "E:\DemoDatas\\runs\detect\\Adam_origin_yolov8-ca-head_2000\F1_curve.csv"
excel_path[3] = "E:\DemoDatas\\runs\detect\\Adam_origin_yolov8-cbam_2000\F1_curve.csv"
excel_path[4] = "E:\DemoDatas\\runs\detect\\Adam_origin_yolov8-cbam-head_2000\F1_curve.csv"
excel_path[5] = "E:\DemoDatas\\runs\detect\\Adam_origin_yolov8-se_2000\F1_curve.csv"
excel_path[6] = "E:\DemoDatas\\runs\detect\\Adam_origin_yolov8-se-head_2000\F1_curve.csv"

excel_path[7] = "E:\DemoDatas\\runs\detect\\Adam_origin_yolov8-p2_2000\F1_curve.csv"
excel_path[8] = "E:\DemoDatas\\runs\detect\\Adam_origin_yolov8-ghost-p2_2000\F1_curve.csv"
excel_path[9] = "E:\DemoDatas\\runs\detect\\Adam_origin_yolov5_2000\F1_curve.csv"
# excel_path[9] = "E:\DemoDatas\\runs\detect\\Adam_origin_yolov3_2000\F1_curve.csv"
import pandas as pd
# pd.set_option('display.notebook_repr_html',False)


# 定义一个函数来处理特定列的数据
def convert_value(value):
    if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
        return float(value[1:-1])
    else:
        return value

paths = excel_path # [excel_path[i] for i in [4,7,8]]

xname = "px"
yname = "Large"
# yname = "py"
print(paths)
for excel_path in paths:
    print(excel_path)
    # excel_path = excel_path.replace("F1_curve","PR_curve")
    # if not pathlib.Path.is_file(excel_path):
    #     continue
    # 读取 csv文件
    df = pd.read_csv(excel_path)
    # print(df)
    # 假设两列数据的列名为 'Column1' 和 'Column2'
    x = df[xname]
    df[yname] = df[yname].apply(convert_value)
    y = df[yname]
    # print(x)
    # print(type(y))
    # 绘制图线
    plt.plot(x, y,label=excel_path.split("\\")[-2])
plt.legend()
# 设置图形大小
# plt.xlim(0.8, max(x))
plt.figure(figsize=(10, 6))
plt.xlabel('Column1')
plt.ylabel('Column2')
plt.title('Line Plot of Two Columns')
plt.show()