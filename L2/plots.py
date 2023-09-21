import os
import matplotlib.pyplot as plt

## Rerun with align option

directory = "C:/Users/Erik Bjerned/Documents/Git_repos/ABP/L2/raw"
count = 0
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    print(f)
    if os.path.isfile(f):

        f = f.replace("\\", "/")

        data = []
        
        
        with open(f, "r") as file:
            for line in file:
                split_line = line.split()
                data.append(split_line)
            
            file.close()
        #print(data) 
        size_vec = [float(line[4]) for line in data]
        mem_vec = [float(line[-2]) for line in data]
        gflops_vec = [float(line[-5]) for line in data]
        styles = ["-", "-", "-", "-", "-", "-", "k--"]
        #print(mem_vec)
        plt.xscale("log")
        
        plt.plot(size_vec, mem_vec, styles[count], linewidth=1)
        
        plt.title("Matrix-vector multiplication")
        #plt.legend(["1", "256", "512", "768", "1024"])
        plt.ylabel("Memory throughput [GB/s]")
        plt.xlabel("Size N [-]")
        #plt.axes((512, 1e8, 0, 256))
        if count == 6:
            plt.legend(["32", "64", "128", "256", "512", "1024", "cuBLAS"])
            plt.grid()
            plt.savefig("Matvec.png")
            plt.show()
            
    
            
        
            
            
            
    count += 1