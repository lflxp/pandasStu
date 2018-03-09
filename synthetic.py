import math

# 在此第一个单元格中，我们将加载必要的库。
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
# from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

# 加载数据集。
# 我们将对数据进行随机化处理，以确保不会出现任何病态排序结果（可能会损害随机梯度下降法的效果）。此外，我们会将 median_house_value 调整为以千为单位，这样，模型就能够以常用范围内的学习速率较为轻松地学习这些数据。
cali_data = pd.read_csv("./c.csv",sep=",")
cali_data = cali_data.reindex(np.random.permutation(cali_data.index))
cali_data["median_house_value"] /= 1000.0

# 检查数据
# print cali_data
print cali_data.describe()

# 第 1 步：定义特征并配置特征列
# 为了将我们的训练数据导入 TensorFlow，我们需要指定每个特征包含的数据类型。在本练习及今后的练习中，我们主要会使用以下两类数据：
# 分类数据：一种文字数据。在本练习中，我们的住房数据集不包含任何分类特征，但您可能会看到的示例包括家居风格以及房地产广告词。
# 数值数据：一种数字（整数或浮点数）数据以及您希望视为数字的数据。有时您可能会希望将数值数据（例如邮政编码）视为分类数据（我们将在稍后的部分对此进行详细说明）。
# 在 TensorFlow 中，我们使用一种称为“特征列”的结构来表示特征的数据类型。特征列仅存储对特征数据的描述；不包含特征数据本身。
# 一开始，我们只使用一个数值输入特征 total_rooms。以下代码会从 california_housing_dataframe 中提取 total_rooms 数据，并使用 numeric_column 定义特征列，这样会将其数据指定为数值：
# https://colab.research.google.com/notebooks/mlcc/first_steps_with_tensor_flow.ipynb?hl=zh-cn#scrollTo=EL8-9d4ZJNR7
my_feature = cali_data[["total_rooms"]]
feature_columns = [tf.feature_columns.numeric_column("total_rooms")]