高斯密度函数为：
$$
\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi} \sigma} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
$$
对于本题我们有 $\sigma_1 = \sigma_2 = 1$，所以高斯密度化简为：
$$
\mathcal{N}(x \mid \mu, 1) = \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{(x - \mu)^2}{2} \right)
$$
因此，对于后验概率：
$$
P(Z = 1 \mid X = x) = \frac{0.25 \cdot \exp\left( -\frac{(x - 0)^2}{2} \right)}{0.25 \cdot \exp\left( -\frac{(x - 0)^2}{2} \right) + 0.75 \cdot \exp\left( -\frac{(x - 2)^2}{2} \right)}
$$

------

### 🔷 示例：X = 1

$$
P(Z = 1 \mid X = 1) = \frac{0.25 \cdot \exp(-0.5)}{0.25 \cdot \exp(-0.5) + 0.75 \cdot \exp(-0.5)} = \frac{0.25}{0.25 + 0.75} = 0.25
$$

------

### 🔷 示例：X = -5

$$
P(Z = 1 \mid X = -5) = \frac{0.25 \cdot \exp\left( -\frac{25}{2} \right)}{0.25 \cdot \exp\left( -\frac{25}{2} \right) + 0.75 \cdot \exp\left( -\frac{49}{2} \right)} 
$$

因为 $e^{-24.5} \ll e^{-12.5}$

------

### 🔷 示例：X = 5

$$
P(Z = 2 \mid X = 5) = \frac{0.75 \cdot \exp\left( -\frac{9}{2} \right)}{0.25 \cdot \exp\left( -\frac{25}{2} \right) + 0.75 \cdot \exp\left( -\frac{9}{2} \right)}
$$