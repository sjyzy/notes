# LightDepth: Single-View Depth Self-Supervision from Illumination Decline

论文目标：提出了一种自监督训练方法，对结肠镜进行深度估计。

贡献：自监督超过了有监督的效果

方法：通过光照判断距离，越亮说明越近。通过神经网络预测反照率，深度图和法向量图，通过这三个图渲染出原始图像，将渲染的原始图像与真实图像计算loss。

1.Photometric Model

光照率的计算方法借鉴了SLS方法

![image-20250708203146109](/Users/shanjingyang/Library/Application Support/typora-user-images/image-20250708203146109.png)

一个像素点的光照，可以通过其位置$x_i$和相机方向的夹角$\psi_i$决定，$\sigma_0$ 表示最大光照。

实际计算过程中$x_i$是不知道的，但是可以通过深度信息$d_i$和相机射线$r_i$得到。$x_i=d_ir_i$

而相机射线可以通过法线方向$u_i$得到，$r_i=\pi^{-1}(u_i)$得到，

$p_i$表示反射率，g表示相机的数码增强

![image-20250708204451506](/Users/shanjingyang/Library/Application Support/typora-user-images/image-20250708204451506.png)

模型假定Lambertian反射