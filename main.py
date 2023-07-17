from unittest import case


def isPalindrome(x):
    st = str(x)
    if st == st[::-1]:
        return 1
    else:
        return 0


def removeDuplicates(nums):
    duplicates = 0
    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1]:
            duplicates += 1
        else:
            nums[i - duplicates] = nums[i]
    return len(nums) - duplicates


# работает на 3.10(((
# def romanToInt(s):
#     # c = list(s)
#     s = s.replace("IV", "IIII").replace("IX", "VIIII")
#     s = s.replace("XL", "XXXX").replace("XC", "LXXXX")
#     s = s.replace("CD", "CCCC").replace("CM", "DCCCC")
#     summ = 0
#     for i in s:
#         match i:
#             case "I":
#                 summ = summ + 1
#             case "V":
#                 summ = summ + 5
#             case "X":
#                 summ = summ + 10
#             case "L":
#                 summ = summ + 50
#             case "C":
#                 summ = summ + 100
#             case "D":
#                 summ = summ + 500
#             case "M":
#                 summ = summ + 1000
#     return summ


def reverse(x):
    reversed_num = ""
    xx = str(x)
    if x < 0:
        reversed_num += "-"
        for i in range(len(str(x)) - 1):
            reversed_num += xx[len(xx) - 1 - i]
    else:
        for i in range(len(str(x))):
            reversed_num += xx[len(xx) - 1 - i]
    return int(reversed_num)


def longestCommonPrefix(strs):
    if "" in strs or strs == []:
        return ""
    preix = strs[0]
    for i in range(1, len(strs)):
        while preix != "":
            try:
                if str.index(str(strs[i]), preix) == 0:
                    break
                else:
                    preix = preix[:-1]
            except:
                preix = preix[:-1]
    return preix


if __name__ == '__main__':
    # removeDuplicates([0, 0, 1, 1, 1, 2, 2, 3, 3, 4])

    reverse(2281)

    # isPalindrome(-121)

    # strs = ["brak", "dom", "semya"]
    # longestCommonPrefix(strs)

    # romanToInt("LVIII")
