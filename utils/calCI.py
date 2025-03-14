import numpy as np
import scipy.stats as st

data = []
for _ in range(5):
    num = float(input())
    data.append(num)

mean = np.mean(data)
stderr = np.std(data, ddof=1) / np.sqrt(len(data))

ci = st.t.interval(0.95, len(data)-1, loc=mean, scale=stderr)

lower, upper = ci

print(f"result: {mean:.3f}({lower:.3f}-{upper:.3f})")
