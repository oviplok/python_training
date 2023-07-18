def minSubLen(target, nums):
    nums.sort(reverse=True)
    turn = 0
    result = 100
    sum = 0
    for i in range(len(nums)):
        for j in range(len(nums)):
            if sum >= target and turn <= result:
                result = turn
                sum = 0
                turn = 0
                break
            if i != j:
                sum += nums[j]
                turn += 1
    return result if result != 100 else 0


if __name__ == '__main__':
    print(minSubLen(213, [12, 28, 83, 4, 25, 26, 25, 2, 25, 25, 25, 12]))
