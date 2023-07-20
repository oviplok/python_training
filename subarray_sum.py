def minSubLen_V3(target, nums):
    total = left = 0
    result = len(nums) + 1
    for right, n in enumerate(nums):
        total += n
        while total >= target:
            result = min(result, right - left + 1)
            total -= nums[left]
            left += 1
    return result if result <= len(nums) else 0


def minSubLen_V2(target, nums):
    nums.sort(reverse=True)
    turn, sum = 0, 0
    result = 100
    for i in range(len(nums)):
        if sum >= target and turn <= result:
            result = turn
            sum = 0
            turn = 0
            break
        sum += nums[i]
        turn += 1
    return result if result != 100 else 0


def minSubLen_V1(target, nums):
    # nums.sort(reverse=True)
    turn, sum, w = 0, 0, 0
    result = 100
    for i in range(len(nums)):
        for i in range(len(nums)):
            if sum >= target and turn <= result:
                result = turn
                sum = 0
                turn = 0
                w += 1
                break
            if i <= w:
                sum += nums[i]
                turn += 1

    return result if result != 100 else 0


if __name__ == '__main__':
    print(minSubLen_V1(7, [2, 3, 1, 2, 4, 3]))
    print(minSubLen_V2(7, [2, 3, 1, 2, 4, 3]))
    print(minSubLen_V3(7, [2, 3, 1, 2, 4, 3]))
