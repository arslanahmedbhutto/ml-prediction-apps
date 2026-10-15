# Model comparison

5-fold CV on the training split (80%); test = held-out 20%. Best model chosen by CV RMSE.


## Student GPA (2,392 rows), best: **Linear regression**

| Model | CV RMSE | CV R² | Test RMSE | Test MAE | Test R² |
|---|---|---|---|---|---|
| **Linear regression** | 0.197 ± 0.007 | 0.953 | 0.197 | 0.155 | 0.953 |
| Ridge | 0.197 ± 0.007 | 0.953 | 0.197 | 0.155 | 0.953 |
| Gradient boosting | 0.210 ± 0.004 | 0.947 | 0.209 | 0.163 | 0.947 |
| Random forest | 0.242 ± 0.005 | 0.930 | 0.242 | 0.188 | 0.929 |
| Mean baseline | 0.916 ± 0.020 | -0.000 | 0.910 | 0.766 | -0.001 |

## House price (489 rows), best: **Linear regression**

| Model | CV RMSE | CV R² | Test RMSE | Test MAE | Test R² |
|---|---|---|---|---|---|
| **Linear regression** | 29.265 ± 1.319 | 0.999 | 28.547 | 24.511 | 0.999 |
| Ridge | 29.265 ± 1.317 | 0.999 | 28.544 | 24.508 | 0.999 |
| Gradient boosting | 91.319 ± 9.063 | 0.991 | 74.143 | 57.418 | 0.995 |
| Random forest | 142.940 ± 5.645 | 0.979 | 127.694 | 99.945 | 0.985 |
| Mean baseline | 1,011.255 ± 29.619 | -0.032 | 1,042.611 | 883.158 | -0.014 |

## Salary (35 rows), best: **Ridge**

| Model | CV RMSE | CV R² | Test RMSE | Test MAE | Test R² |
|---|---|---|---|---|---|
| **Ridge** | 5,815.884 ± 1,331.627 | 0.958 | 7,511.103 | 6,755.977 | 0.890 |
| Linear regression | 5,870.700 ± 1,369.816 | 0.958 | 7,467.382 | 6,692.364 | 0.891 |
| Gradient boosting | 7,839.852 ± 1,743.173 | 0.923 | 5,487.217 | 3,983.010 | 0.941 |
| Random forest | 7,886.887 ± 909.672 | 0.924 | 7,258.508 | 5,177.220 | 0.897 |
| Mean baseline | 33,852.604 ± 2,788.098 | -0.266 | 26,589.538 | 24,567.714 | -0.377 |
