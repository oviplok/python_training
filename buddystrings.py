def buddyStrings(s, goal):
    sg =[]
    t = 0
    if len(s) != len(goal):
        return False
    if s == goal:
        return True if len(s) - len(set(s)) >= 1 else False
    for i in range(len(goal)):
        if s[i] != goal[i]:
            sg.append(i)
            t += 1
    if len(sg) != 2: return False

    if s[sg[0]] == goal[sg[1]] and s[sg[1]] == goal[sg[0]]:
        return True
    return False


if __name__ == '__main__':
    print(buddyStrings("aaaaaaabc", "aaaaaaacb"))
