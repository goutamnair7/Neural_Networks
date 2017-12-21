import sys
import random
import math

def sigmoid(x):
    exp = math.exp(-x)
    return 1 / (1 + exp)

def main():

    train_file = sys.argv[1]
    train_labels = sys.argv[2]
    dev_file = sys.argv[3]

    f = open(train_file, "r")
    lines = f.readlines()
    f.close()

    del lines[0]

    f = open(train_labels, "r")
    labels = f.readlines()
    f.close()

    train_examples = []

    for i in range(0, len(lines)):
        ex_line = lines[i].strip().split(",")
        ex = [1.0] * 3
        ex[1] = (float(ex_line[0]) - 1900)/2000.0
        ex[2] = (float(ex_line[1]) - 0.0)/7.0
        
        if ex_line[2] == "yes":
            ex.append(1)
        else:
            ex.append(0)

        if ex_line[3] == "yes":
            ex.append(1)
        else:
            ex.append(0)

        label = labels[i].strip()
        if label == "yes":
            ex.append(1)
        else:
            ex.append(0)

        train_examples.append(ex)

    h_units = 3+1
    attr = 4+1

    w_input = []
    rand = random.sample(range(-100, 100), attr*h_units)
    for i in range(0, attr): 
        w = []
        for j in range(0, h_units):
            w.append(rand[i*h_units+j]/1000.0)
        w_input.append(w)

    w_hidden = []
    rand = random.sample(range(-100, 100), h_units)
    for i in range(0, h_units):
        w_hidden.append(rand[i]/1000.0)

    hidden = [1.0] * h_units
    eta = 0.1
    count = 1

    while count <= 40000:
        error = 0 

        for ex in train_examples:
            train = ex[:-1]
            label = ex[-1]
            
            for j in range(1, h_units):
                hid = 0 
                for i in range(0, attr): 
                    hid += (train[i] * w_input[i][j])
                hidden[j] = sigmoid(hid)
            
            out = 0 
            for i in range(0, h_units):
                out += (hidden[i] * w_hidden[i])
            out = sigmoid(out)
            
            error += (label - out) ** 2

            delta_k = out * (1.0 - out) * (label - out)
            delta_h = [0.0] * h_units

            delta_h[0] = hidden[i] * w_hidden[i] * delta_k
            for i in range(1, h_units):
                delta_h[i] = hidden[i] * (1.0 - hidden[i]) * w_hidden[i] * delta_k
                w_hidden[i] += (eta * delta_k * hidden[i])

            for j in range(0, h_units):
                for i in range(0, attr):
                    w_input[i][j] += (eta * delta_h[j] * train[i])
        
        print str(error/2.0)
        count += 1

    print "TRAINING COMPLETED! NOW PREDICTING."

    f = open(dev_file, "r")
    lines = f.readlines()
    f.close()

    del lines[0]

#    f = open("music_dev_keys.txt", "r")
#    labels = f.readlines()
#    f.close()

#    lab = []
#    for line in labels:
#        l = line.strip()
#        if l == "yes":
#        	lab.append(1)
#        else:
#            lab.append(0)
    
#    count = 0
#    k = 0

    for line in lines:
        ex_line = line.strip().split(",")
		
        ex = [1.0] * 3
        ex[1] = (float(ex_line[0]) - 1900)/2000.0
        ex[2] = (float(ex_line[1]) - 0.0)/7.0
		
        if ex_line[2] == "yes":
            ex.append(1)
        else:
            ex.append(0)

        if ex_line[3] == "yes":
            ex.append(1)
        else:
            ex.append(0)

        hidden = [1.0] * h_units
        for j in range(1, h_units):
            hid = 0
            for i in range(0, attr):
                hid += (ex[i] * w_input[i][j])
            hidden[j] = sigmoid(hid)

        out = 0
        for i in range(0, h_units):
            out += (hidden[i] * w_hidden[i])
        out = sigmoid(out)
        
        if out >= 0.5:
#            if lab[k] == 1:
#                count += 1
            print "yes"
        else:
#            if lab[k] == 0:
#                count += 1
            print "no"
#        k += 1
    
#    print "Acc: ", (count*1.0) / (len(lab))

if __name__ == "__main__":
	main()
