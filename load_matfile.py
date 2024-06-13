from scipy.io import loadmat

# 加载.mat文件
data = loadmat('/Users/lvangge/Desktop/Archive/project1_release/codes/digits.mat')

# 查看.mat文件中包含的所有变量名
print(data.keys())

# 获取特定变量的值
print(data['y'].shape)

