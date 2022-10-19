def factorial(x):
    """This is a recursive function
    to find the factorial of an integer"""

    if x == 1:
        return 1
    else:
        # recursive call to the function
        return (x * factorial(x-1))

def even_fac_sum(*list):
    sum = 0
    for i in range(len(list)):
        if type(list[i]) is int and list[i] % 2 == 0:
            sum += 1 / factorial(list[i])
        else:
            continue

    return sum


print(even_fac_sum(3,4,6,12.12, 'tracy'))