def func(a, b):
    res = 1
    for i in range(a - b + 1, a + 1):
        res *= i
    return res

num1 = func(44, 20) / func(64, 20)
num2 = pow(2, 20) - 1
print(func(44, 20))
print(func(64, 20))
print(num1 * num2)