flag = True
while flag:
    TP = int(input())
    FP = int(input())
    # TN = input("TN：")
    FN = int(input())
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = (2*precision*recall)/(precision+recall)
    print(f1)