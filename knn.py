from collections import Counter
def edistance(v1,v2):
    y = 0
    for x1,x2 in zip(v1,v2):
        y += pow(x1-x2, 2)
    return y
def sort_vector(vec):
    return sorted(vec, key=lambda l:l[1])
def train_test(data, k_neighbours, instance):
    new_data = []
    for x in data:
        new_data.append((x,edistance(x,instance)))
    elect = sort_vector(new_data)[:k_neighbours]
    xlst=[]
    for v in elect:
        xlst.append(v[0][2])
    return Counter(xlst).most_common(1)[0][0]

def main():
    ## 0 -> Bad, 1 -> Good
    with open("knn_input.txt", "r", encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(tuple([int(x) for x in line.strip().split()]))
    # data = [(7,7,0),(7,4,0),(3,4,1),(1,4,1)]
    test = (3,7)
    k_neighbour = 3
    print(train_test(data,k_neighbour,test))

if __name__ == "__main__":
    main()