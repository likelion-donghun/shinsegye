from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt


X = np.array([258.0, 270.0, 294.0, 320.0, 342.0, 368.0, 396.0, 446.0, 480.0, 586.0])[:, np.newaxis]
y = np.array([236.4, 234.4, 252.8, 298.6, 314.2, 342.2, 360.8, 368.0, 391.2, 390.8])
lr = LinearRegression()
pr = LinearRegression()

# 2차항 추가
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)

# 비교를 위해 심플 선형 회귀 모델 피팅
lr.fit(X, y) # train data 학습
X_fit = np.arange(250, 600, 10)[:, np.newaxis]
y_lin_fit = lr.predict(X_fit) # test data에 대한 심플 선형 회위

# 다중회귀 모델 피팅
pr.fit(X_quad, y) # 2차항을 가진 인풋을 통해 학습
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit)) # test data를 2차로 변형 및 학습한 다중 선형 회귀
plt.scatter(X, y, label = 'training points') # train data를 산점도로 표시
plt.plot(X_fit, y_lin_fit, label = 'linear fit', linestyle = '--') # test data에 대한 심플 선형 회귀 그래프
plt.plot(X_fit, y_quad_fit, label='quadratic fit') # test data에 대한 다중 선형 회귀 그래프
plt.legend(loc='upper left')
plt.show()

y_lin_pred = lr.predict(X) # train data에 대한 심플 선형 회귀
y_quad_pred = pr.predict(X_quad) # train data에 대한 다중 선형 회귀
print('Training MSE linear: %.3f, quadratic: %.3f' %(mean_squared_error(y, y_lin_pred), mean_squared_error(y, y_quad_pred)))

print('Training R^2 linear: %.3f, quadratic: %.3f'%(r2_score(y, y_lin_pred), r2_score(y, y_quad_pred)))
